//! Jacobian Lens — measuring how hidden-state dimensions map to output logits.
//!
//! The Jacobian is not a module; it is a *measurement lens* applied to the hidden state.
//! Each row of J tells us: "if I nudge hidden dimension d, how does output token t change?"
//!
//! This is the "key" Jason described — the emergent thing that turns clusters into perm-addresses.
//! We don't build the cathedral; we lay one brick and see if it holds.

use candle_core::{Device, Result, Tensor};
use crate::hooks::HookSite;
use rand::rng;
use rand::Rng;
use rand::seq::SliceRandom;

/// A Jacobian measurement session.
///
/// Perturbs hidden state along selected dimensions, measures output changes,
/// and returns a sensitivity map: which hidden dimensions drive which outputs.
pub struct JacobianLens {
    /// Perturbation magnitude for finite difference.
    /// Start with 1e-4, sweep 1e-5 to 1e-3 for stability.
    pub epsilon: f32,
    
    /// Which hook sites to measure at (FinalNorm, PostMlp, etc.).
    /// FinalNorm is the primary — this is where "emergent thinking" lives.
    pub sites: Vec<HookSite>,
    
    /// Only measure sensitivity for top-k output tokens (by absolute J value).
    /// Reduces output dimensionality from vocab_size to top_k.
    pub top_k: usize,
    
    /// Subsample hidden dimensions for efficiency.
    /// 0 = all dimensions; >0 = measure only this many (randomly sampled).
    pub max_dims: usize,
    
    /// Optional persistent trace log.
    pub trace: Option<TraceWriter>,
}

/// Where in the model a measurement was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementLocation {
    pub site: HookSite,
    pub layer_idx: usize,
}

/// A single sensitivity measurement at one hook site.
///
/// Contains the full Jacobian matrix (or subsampled version) and derived statistics.
pub struct JacobianMeasurement {
    /// Where this measurement was taken.
    pub location: MeasurementLocation,
    
    /// Shape: (top_k, measured_dims)
    /// J[i][j] = sensitivity of output token i to perturbation of hidden dimension j.
    pub sensitivity: Tensor,
    
    /// Frobenius norm of the full sensitivity matrix (summary statistic).
    pub norm: f32,
    
    /// Top-N dimensions by mean |J|, sorted descending.
    /// Each entry: (dimension_index, mean_absolute_sensitivity).
    pub top_dimensions: Vec<(usize, f32)>,
    
    /// Top-N tokens by mean |J|, sorted descending.
    /// Each entry: (token_id, mean_absolute_sensitivity).
    pub top_tokens: Vec<(usize, f32)>,
}

impl Clone for JacobianMeasurement {
    fn clone(&self) -> Self {
        Self {
            location: self.location,
            sensitivity: self.sensitivity.clone(),
            norm: self.norm,
            top_dimensions: self.top_dimensions.clone(),
            top_tokens: self.top_tokens.clone(),
        }
    }
}

/// Summary statistics across all measured sites.
pub struct JacobianReport {
    /// Individual measurements at each site.
    pub measurements: Vec<JacobianMeasurement>,
    
    /// Mean Frobenius norm across all sites (global sensitivity).
    pub global_sensitivity: f32,
    
    /// Dimensions with highest average |J| across all sites.
    /// These are the "dominant" dimensions that drive most output changes.
    pub dominant_dimensions: Vec<usize>,
    
    /// Tokens most sensitive to perturbations across all sites.
    pub dominant_tokens: Vec<usize>,
}

impl Default for JacobianReport {
    fn default() -> Self {
        Self {
            measurements: Vec::new(),
            global_sensitivity: 0.0,
            dominant_dimensions: Vec::new(),
            dominant_tokens: Vec::new(),
        }
    }
}

/// Optional trace writer for persistent Jacobian logs.
pub struct TraceWriter {
    path: std::path::PathBuf,
    file: std::fs::File,
}

impl TraceWriter {
    pub fn new(path: &str) -> Result<Self> {
        let path = std::path::Path::new(path);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }
    
    pub fn write(&mut self, data: &str) -> std::io::Result<()> {
        use std::io::Write;
        self.file.write_all(data.as_bytes())?;
        self.file.write_all(b"\n")?;
        self.file.flush()
    }
}

impl JacobianLens {
    /// Create a new Jacobian lens.
    ///
    /// # Arguments
    /// * `epsilon` — perturbation magnitude for finite difference
    /// * `sites` — which hook sites to measure at
    /// * `top_k` — only track top-k output tokens by sensitivity
    /// * `max_dims` — subsample hidden dimensions (0 = all)
    pub fn new(epsilon: f32, sites: Vec<HookSite>, top_k: usize, max_dims: usize) -> Self {
        Self {
            epsilon,
            sites,
            top_k,
            max_dims,
            trace: None,
        }
    }
    
    /// Enable optional trace logging.
    pub fn with_trace(mut self, path: &str) -> Result<Self> {
        self.trace = Some(TraceWriter::new(path)?);
        Ok(self)
    }
    
    /// Measure Jacobian at a single hidden state.
    ///
    /// # Arguments
    /// * `hidden` — the hidden state tensor, shape (batch, hidden_dim)
    /// * `project_to_logits` — closure that maps hidden state to logits
    ///
    /// # Returns
    /// JacobianReport with sensitivity measurements at each requested site.
    pub fn measure(
        &mut self,
        hidden: &Tensor,
        project_to_logits: impl Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<JacobianReport> {
        let batch_size = hidden.dims()[0];
        let hidden_dim = hidden.dims()[1];
        let dev = hidden.device();
        
        // Subsample dimensions if requested
        let dims_to_measure: Vec<usize> = if self.max_dims > 0 && self.max_dims < hidden_dim {
            let mut rng = rng();
            let mut dims: Vec<usize> = (0..hidden_dim).collect();
            dims.shuffle(&mut rng);
            dims.truncate(self.max_dims);
            dims.sort();
            dims
        } else {
            (0..hidden_dim).collect()
        };
        
        let measured_dim_count = dims_to_measure.len();
        let mut all_measurements: Vec<JacobianMeasurement> = Vec::new();
        
        // For each hook site, measure sensitivity
        for &site in &self.sites {
            // We measure at FinalNorm by taking the hidden state as-is
            // (the hidden state IS the pre-lm_head residual stream)
            let measurement = self.measure_site(
                hidden,
                &dims_to_measure,
                &project_to_logits,
                MeasurementLocation { site, layer_idx: 0 },
            )?;
            all_measurements.push(measurement);
        }
        
        // Compute summary statistics
        let report = self.compute_report(&all_measurements, dev)?;
        
        // Log to trace if enabled
        if let Some(trace) = self.trace.as_mut() {
            let summary = format!(
                "JacobianReport: global_sensitivity={:.6}, top_dims={:?}, top_tokens={:?}",
                report.global_sensitivity,
                report.dominant_dimensions,
                report.dominant_tokens
            );
            let _ = trace.write(&summary);
        }
        
        Ok(report)
    }
    
    /// Measure sensitivity at a single hook site.
    fn measure_site(
        &self,
        hidden: &Tensor,
        dims_to_measure: &[usize],
        project_to_logits: &impl Fn(&Tensor) -> Result<Tensor>,
        location: MeasurementLocation,
    ) -> Result<JacobianMeasurement> {
        let hidden_dim = hidden.dims()[1];
        let dev = hidden.device();
        
        // Get baseline logits
        let baseline_logits = project_to_logits(hidden)?;
        
        // Determine top-k tokens (by absolute value of baseline logits)
        let top_k_tokens = self.select_top_k_tokens(&baseline_logits)?;
        
        // For each dimension to measure, compute finite difference
        // J[token][dim] = (logits(hidden + ε·e_dim) - logits(hidden - ε·e_dim)) / (2ε)
        let mut sensitivity_matrix: Vec<f32> = vec![0.0; top_k_tokens.len() * dims_to_measure.len()];
        
        for (dim_idx, &dim) in dims_to_measure.iter().enumerate() {
            // One-hot perturbation of magnitude ε at `dim` and nowhere else.
            //
            // This used to be `Tensor::zeros(...).add(&scalar)`, which broadcasts ε
            // across *every* dimension; combined with `hidden.sub(&neg_perturbation)`
            // where neg_perturbation was all -ε, both sides came out as `hidden + ε`
            // and the whole sensitivity matrix was identically zero.
            let mut onehot = vec![0f32; hidden_dim];
            onehot[dim] = self.epsilon;
            let delta = Tensor::from_vec(onehot, (1, hidden_dim), dev)?
                .to_dtype(hidden.dtype())?;

            // Compute perturbed hidden states
            let pos_hidden = hidden.broadcast_add(&delta)?;
            let neg_hidden = hidden.broadcast_sub(&delta)?;

            // Get perturbed logits
            let pos_logits = project_to_logits(&pos_hidden)?;
            let neg_logits = project_to_logits(&neg_hidden)?;
            
            // Compute finite difference: (pos - neg) / (2ε)
            let diff = pos_logits.sub(&neg_logits)?;
            let two_eps = Tensor::new(&[2.0 * self.epsilon], dev)?;
            let jacobian = diff.div(&two_eps)?;
            
            // Extract top-k token sensitivities for this dimension
            for (token_idx, &token_id) in top_k_tokens.iter().enumerate() {
                let sensitivity = jacobian.get(0)?.get(token_id)?;
                let sensitivity_val: f32 = sensitivity.to_vec1()?.first().copied().unwrap_or(0.0);
                sensitivity_matrix[token_idx * dims_to_measure.len() + dim_idx] = sensitivity_val;
            }
        }
        
        // Build sensitivity tensor
        let sensitivity_tensor = Tensor::from_vec(
            sensitivity_matrix,
            (top_k_tokens.len(), dims_to_measure.len()),
            dev,
        )?;
        
        // Compute summary statistics
        let norm = self.compute_frobenius_norm(&sensitivity_tensor)?;
        let top_dimensions = self.compute_top_dimensions(&sensitivity_tensor, dims_to_measure)?;
        let top_tokens = self.compute_top_tokens(&sensitivity_tensor, &top_k_tokens)?;
        
        Ok(JacobianMeasurement {
            location,
            sensitivity: sensitivity_tensor,
            norm,
            top_dimensions,
            top_tokens,
        })
    }
    
    /// Select top-k tokens by absolute baseline logit value.
    fn select_top_k_tokens(&self, logits: &Tensor) -> Result<Vec<usize>> {
        let vocab_size = logits.dims()[0];
        let abs_logits = logits.abs()?;
        
        // Get top-k indices via arg_sort_last_dim (descending)
        let top_k = std::cmp::min(self.top_k, vocab_size);
        
        // arg_sort_last_dim returns indices in ascending order, so we reverse
        let sorted_indices = abs_logits.arg_sort_last_dim(false)?;
        let top_indices: Vec<usize> = sorted_indices
            .to_vec1::<i64>()?
            .iter()
            .rev()
            .take(top_k)
            .map(|&x| x as usize)
            .collect();
        
        Ok(top_indices)
    }
    
    /// Compute Frobenius norm of the sensitivity matrix.
    fn compute_frobenius_norm(&self, sensitivity: &Tensor) -> Result<f32> {
        let squared = sensitivity.sqr()?;
        let sum = squared.sum_all()?;
        let sum_val: f32 = sum.to_scalar()?;
        Ok(sum_val.sqrt())
    }
    
    /// Compute top dimensions by mean absolute sensitivity.
    fn compute_top_dimensions(
        &self,
        sensitivity: &Tensor,
        dims_to_measure: &[usize],
    ) -> Result<Vec<(usize, f32)>> {
        let (top_k_count, measured_dims) = (sensitivity.dims()[0], sensitivity.dims()[1]);
        
        // Compute mean |J| across tokens for each dimension
        let abs_sensitivity = sensitivity.abs()?;
        let mean_per_dim = abs_sensitivity.mean(0)?;
        
        let mut dim_sensitivities: Vec<(usize, f32)> = (0..measured_dims)
            .map(|i| {
                let val: f32 = mean_per_dim.get(i).unwrap().to_scalar().unwrap_or(0.0);
                (dims_to_measure[i], val)
            })
            .collect();
        
        // Sort descending by sensitivity
        dim_sensitivities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(dim_sensitivities)
    }
    
    /// Compute top tokens by mean absolute sensitivity.
    fn compute_top_tokens(
        &self,
        sensitivity: &Tensor,
        top_k_tokens: &[usize],
    ) -> Result<Vec<(usize, f32)>> {
        let (top_k_count, measured_dims) = (sensitivity.dims()[0], sensitivity.dims()[1]);
        
        // Compute mean |J| across dimensions for each token
        let abs_sensitivity = sensitivity.abs()?;
        let mean_per_token = abs_sensitivity.mean(1)?;
        
        let mut token_sensitivities: Vec<(usize, f32)> = (0..top_k_count)
            .map(|i| {
                let val: f32 = mean_per_token.get(i).unwrap().to_scalar().unwrap_or(0.0);
                (top_k_tokens[i], val)
            })
            .collect();
        
        // Sort descending by sensitivity
        token_sensitivities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(token_sensitivities)
    }
    
    /// Compute summary report across all measurements.
    fn compute_report(
        &self,
        measurements: &[JacobianMeasurement],
        dev: &Device,
    ) -> Result<JacobianReport> {
        if measurements.is_empty() {
            return Ok(JacobianReport::default());
        }
        
        // Global sensitivity: mean Frobenius norm across sites
        let global_sensitivity: f32 = measurements.iter()
            .map(|m| m.norm)
            .sum::<f32>() / measurements.len() as f32;
        
        // Dominant dimensions: aggregate across all sites
        // (simplified: take union of top dimensions from all sites)
        let mut dim_sensitivity_sum: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        let mut dim_sensitivity_count: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        
        for measurement in measurements {
            for &(dim, sens) in &measurement.top_dimensions {
                *dim_sensitivity_sum.entry(dim).or_insert(0.0) += sens;
                *dim_sensitivity_count.entry(dim).or_insert(0.0) += 1.0;
            }
        }
        
        let mut dominant_dimensions: Vec<(usize, f32)> = dim_sensitivity_sum.iter()
            .map(|(dim, sum)| {
                let avg = sum / dim_sensitivity_count[dim];
                (*dim, avg)
            })
            .collect();
        
        // Sort descending by average sensitivity
        dominant_dimensions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Dominant tokens: aggregate across all sites (simplified)
        let mut token_sensitivity_sum: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        let mut token_sensitivity_count: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        
        for measurement in measurements {
            for &(token_id, sens) in &measurement.top_tokens {
                *token_sensitivity_sum.entry(token_id).or_insert(0.0) += sens;
                *token_sensitivity_count.entry(token_id).or_insert(0.0) += 1.0;
            }
        }
        
        let mut dominant_tokens: Vec<(usize, f32)> = token_sensitivity_sum.iter()
            .map(|(token_id, sum)| {
                let avg = sum / token_sensitivity_count[token_id];
                (*token_id, avg)
            })
            .collect();
        
        dominant_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(JacobianReport {
            measurements: measurements.to_vec(),
            global_sensitivity,
            dominant_dimensions: dominant_dimensions.into_iter().map(|(dim, _)| dim).collect(),
            dominant_tokens: dominant_tokens.into_iter().map(|(token_id, _)| token_id).collect(),
        })
    }
}

// ─── Multi-key picker addresses ─────────────────────────────────────────────
//
// Jacobian lens → instructional / first-thought keys. One episode can emit
// several keys (first commit / revise onset / settle). Keys cluster into a
// multi-packet pick set (k≈8 bet). Semantics only nudge; residual state is
// the load. Never inject raw 64D into wrong residual_d — text bridge only.
//
// Hook points (measure):
//   - main.rs: measure_jacobian_step on jacobian.interval (periodic proxy)
//   - Prefer event-driven: first content token (answer), revise flip, settle
//   - From report.top_dimensions / dominant_dimensions → DimSignature → JacobianKey

/// Generation phase tag for a key sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyPhase {
    /// First commit of an answer (instructional / first-thought key).
    Answer,
    /// Self-reg revise onset (wait-loop, thrash, text cue).
    Revise,
    /// Settle / EOS / clamp end of episode turn.
    Settle,
}

impl KeyPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            KeyPhase::Answer => "answer",
            KeyPhase::Revise => "revise",
            KeyPhase::Settle => "settle",
        }
    }

    /// Parse self_reg phase strings used in main.rs probe path.
    pub fn from_str_lossy(s: &str) -> Option<Self> {
        match s {
            "answer" => Some(KeyPhase::Answer),
            "revise" => Some(KeyPhase::Revise),
            "settle" => Some(KeyPhase::Settle),
            _ => None,
        }
    }
}

/// Sparse dim-signature: top-k driving hidden dims + non-negative weights.
/// Weights are typically mean |J| (or normalized). Used as the perm-address body.
#[derive(Debug, Clone, PartialEq)]
pub struct DimSignature {
    /// (dimension_index, weight) sorted by weight descending; dims unique.
    pub dims: Vec<(usize, f32)>,
}

impl DimSignature {
    pub fn new(mut dims: Vec<(usize, f32)>) -> Self {
        // Drop non-positive weights; sort descending weight; keep first of each dim.
        dims.retain(|(_, w)| *w > 0.0 && w.is_finite());
        dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut seen = std::collections::HashSet::new();
        dims.retain(|(d, _)| seen.insert(*d));
        Self { dims }
    }

    /// From a JacobianMeasurement's top_dimensions (already sorted).
    pub fn from_top_dimensions(top: &[(usize, f32)], top_k: usize) -> Self {
        let take = if top_k == 0 { top.len() } else { top_k.min(top.len()) };
        Self::new(top.iter().take(take).copied().collect())
    }

    /// From a report's dominant dims with uniform weight 1.0 (proxy when |J| lost).
    pub fn from_dominant_dims(dims: &[usize], top_k: usize) -> Self {
        let take = if top_k == 0 { dims.len() } else { top_k.min(dims.len()) };
        Self::new(
            dims.iter()
                .take(take)
                .enumerate()
                .map(|(i, &d)| (d, 1.0 / (1.0 + i as f32)))
                .collect(),
        )
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn dim_set(&self) -> std::collections::HashSet<usize> {
        self.dims.iter().map(|(d, _)| *d).collect()
    }

    /// L2 norm of the sparse weight vector.
    pub fn l2_norm(&self) -> f32 {
        self.dims
            .iter()
            .map(|(_, w)| w * w)
            .sum::<f32>()
            .sqrt()
    }
}

/// One instructional key sample for storage/retrieval addressing.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobianKey {
    /// Sparse dim signature (perm-address body).
    pub signature: DimSignature,
    /// When in the turn this was measured.
    pub phase: KeyPhase,
    /// Decode step within the turn (0-based or absolute — caller convention).
    pub step: usize,
    /// Optional turn index within the episode/session.
    pub turn: Option<usize>,
    /// Optional FNV-1a (or other) hash of a short text bridge string — not raw embed.
    pub text_bridge_hash: Option<u64>,
    /// Host residual width this signature was measured in (must match at inject time).
    pub residual_d: usize,
    /// Optional Frobenius / global sensitivity at measure time.
    pub sensitivity_norm: Option<f32>,
}

impl JacobianKey {
    pub fn new(
        signature: DimSignature,
        phase: KeyPhase,
        step: usize,
        residual_d: usize,
    ) -> Self {
        Self {
            signature,
            phase,
            step,
            turn: None,
            text_bridge_hash: None,
            residual_d,
            sensitivity_norm: None,
        }
    }

    pub fn with_turn(mut self, turn: usize) -> Self {
        self.turn = Some(turn);
        self
    }

    pub fn with_text_bridge_hash(mut self, h: u64) -> Self {
        self.text_bridge_hash = Some(h);
        self
    }

    pub fn with_sensitivity_norm(mut self, n: f32) -> Self {
        self.sensitivity_norm = Some(n);
        self
    }

    /// Build a key from a live JacobianReport (dominant dims + global norm).
    pub fn from_report(
        report: &JacobianReport,
        phase: KeyPhase,
        step: usize,
        residual_d: usize,
        top_k: usize,
    ) -> Self {
        // Prefer weighted dims from first measurement if present.
        let signature = if let Some(m) = report.measurements.first() {
            DimSignature::from_top_dimensions(&m.top_dimensions, top_k)
        } else {
            DimSignature::from_dominant_dims(&report.dominant_dimensions, top_k)
        };
        Self::new(signature, phase, step, residual_d)
            .with_sensitivity_norm(report.global_sensitivity)
    }
}

/// FNV-1a 64-bit hash for optional text-bridge strings (stable, no deps).
pub fn text_bridge_hash(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Weighted Jaccard on sparse dim weights:
///   sum min(w_a, w_b) / sum max(w_a, w_b) over the union of dims.
/// Returns 0 if both empty or denominator 0; 1 if identical support+weights.
pub fn weighted_jaccard(a: &DimSignature, b: &DimSignature) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut map_a: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
    for &(d, w) in &a.dims {
        map_a.insert(d, w);
    }
    let mut map_b: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
    for &(d, w) in &b.dims {
        map_b.insert(d, w);
    }
    let mut dims: std::collections::HashSet<usize> = map_a.keys().copied().collect();
    dims.extend(map_b.keys().copied());
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for d in dims {
        let wa = map_a.get(&d).copied().unwrap_or(0.0);
        let wb = map_b.get(&d).copied().unwrap_or(0.0);
        num += wa.min(wb);
        den += wa.max(wb);
    }
    if den <= 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Cosine similarity on sparse dim weights (0 if either zero-norm).
pub fn sparse_cosine(a: &DimSignature, b: &DimSignature) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let map_b: std::collections::HashMap<usize, f32> =
        b.dims.iter().copied().collect();
    let mut dot = 0.0f32;
    for &(d, wa) in &a.dims {
        if let Some(&wb) = map_b.get(&d) {
            dot += wa * wb;
        }
    }
    let na = a.l2_norm();
    let nb = b.l2_norm();
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// Distance in [0, 1]: 1 - weighted_jaccard (0 = identical).
pub fn signature_distance(a: &DimSignature, b: &DimSignature) -> f32 {
    1.0 - weighted_jaccard(a, b)
}

/// Single-linkage style clustering of signatures by distance threshold.
/// Returns cluster_id per input index (0..n-1), dense ids from 0.
pub fn cluster_signatures(sigs: &[&DimSignature], threshold: f32) -> Vec<usize> {
    let n = sigs.len();
    if n == 0 {
        return Vec::new();
    }
    // Union-find
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0; n];
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let mut ra = find(parent, a);
        let mut rb = find(parent, b);
        if ra == rb {
            return;
        }
        if rank[ra] < rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        parent[rb] = ra;
        if rank[ra] == rank[rb] {
            rank[ra] += 1;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if signature_distance(sigs[i], sigs[j]) <= threshold {
                union(&mut parent, &mut rank, i, j);
            }
        }
    }
    // Dense remap
    let roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next = 0usize;
    for r in &roots {
        if !map.contains_key(r) {
            map.insert(*r, next);
            next += 1;
        }
    }
    roots.iter().map(|r| map[r]).collect()
}

/// One pick-query item: a cluster of keys that should co-load as a packet set.
#[derive(Debug, Clone)]
pub struct PickCluster {
    pub cluster_id: usize,
    /// Indices into the parent MultiKeyAddress.keys list.
    pub key_indices: Vec<usize>,
    /// Representative signature (first key in cluster, or later: centroid).
    pub representative: DimSignature,
}

/// Query emitted for the picker: top clusters / top-k packets.
#[derive(Debug, Clone)]
pub struct PickQuery {
    pub clusters: Vec<PickCluster>,
    /// Max packets requested (k≈8 bet default).
    pub top_k: usize,
    /// residual_d of the episode (host must match).
    pub residual_d: usize,
}

/// Multi-key address for one episode: 1..N keys → pick-query (clustered).
#[derive(Debug, Clone, Default)]
pub struct MultiKeyAddress {
    pub keys: Vec<JacobianKey>,
    /// Clustering distance threshold (weighted-Jaccard distance). Default 0.5.
    pub cluster_threshold: f32,
}

impl MultiKeyAddress {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            cluster_threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.cluster_threshold = t.clamp(0.0, 1.0);
        self
    }

    pub fn push(&mut self, key: JacobianKey) {
        self.keys.push(key);
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Cluster keys by signature distance; emit pick-query with up to `top_k` clusters.
    /// Clusters ordered by size (desc) then by earliest step in cluster.
    pub fn emit_pick_query(&self, top_k: usize) -> PickQuery {
        let residual_d = self.keys.first().map(|k| k.residual_d).unwrap_or(0);
        if self.keys.is_empty() {
            return PickQuery {
                clusters: Vec::new(),
                top_k,
                residual_d,
            };
        }
        let sigs: Vec<&DimSignature> = self.keys.iter().map(|k| &k.signature).collect();
        let ids = cluster_signatures(&sigs, self.cluster_threshold);

        let mut by_cluster: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, cid) in ids.iter().enumerate() {
            by_cluster.entry(*cid).or_default().push(i);
        }

        let mut clusters: Vec<PickCluster> = by_cluster
            .into_iter()
            .map(|(cluster_id, mut key_indices)| {
                key_indices.sort_by_key(|&i| self.keys[i].step);
                let representative = self.keys[key_indices[0]].signature.clone();
                PickCluster {
                    cluster_id,
                    key_indices,
                    representative,
                }
            })
            .collect();

        // Larger clusters first; tie-break by earliest step.
        clusters.sort_by(|a, b| {
            b.key_indices
                .len()
                .cmp(&a.key_indices.len())
                .then_with(|| {
                    let sa = a.key_indices.first().map(|&i| self.keys[i].step).unwrap_or(0);
                    let sb = b.key_indices.first().map(|&i| self.keys[i].step).unwrap_or(0);
                    sa.cmp(&sb)
                })
        });

        let k = if top_k == 0 { clusters.len() } else { top_k.min(clusters.len()) };
        clusters.truncate(k);

        PickQuery {
            clusters,
            top_k,
            residual_d,
        }
    }
}

#[cfg(test)]
mod multi_key_tests {
    use super::*;

    fn sig(pairs: &[(usize, f32)]) -> DimSignature {
        DimSignature::new(pairs.to_vec())
    }

    #[test]
    fn weighted_jaccard_identical_is_one() {
        let a = sig(&[(10, 0.5), (20, 0.3), (30, 0.2)]);
        let b = sig(&[(10, 0.5), (20, 0.3), (30, 0.2)]);
        assert!((weighted_jaccard(&a, &b) - 1.0).abs() < 1e-5);
        assert!(signature_distance(&a, &b) < 1e-5);
    }

    #[test]
    fn weighted_jaccard_disjoint_is_zero() {
        let a = sig(&[(1, 1.0), (2, 0.5)]);
        let b = sig(&[(99, 1.0), (100, 0.5)]);
        assert!(weighted_jaccard(&a, &b) < 1e-5);
        assert!((signature_distance(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_jaccard_partial_overlap() {
        // a: 1→1, 2→1; b: 1→1, 3→1 → min sum=1, max sum=3 → 1/3
        let a = sig(&[(1, 1.0), (2, 1.0)]);
        let b = sig(&[(1, 1.0), (3, 1.0)]);
        let j = weighted_jaccard(&a, &b);
        assert!((j - 1.0 / 3.0).abs() < 1e-5, "j={j}");
    }

    #[test]
    fn sparse_cosine_orthogonal_near_zero() {
        let a = sig(&[(1, 1.0)]);
        let b = sig(&[(2, 1.0)]);
        assert!(sparse_cosine(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn sparse_cosine_parallel_is_one() {
        let a = sig(&[(5, 2.0), (7, 0.0)]); // 0 weight dropped
        let b = sig(&[(5, 4.0)]);
        assert!((sparse_cosine(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cluster_merges_near_sigs() {
        let s0 = sig(&[(1, 1.0), (2, 0.8), (3, 0.5)]);
        let s1 = sig(&[(1, 0.9), (2, 0.7), (3, 0.4)]); // near s0
        let s2 = sig(&[(100, 1.0), (101, 0.5)]); // far
        let ids = cluster_signatures(&[&s0, &s1, &s2], 0.4);
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], ids[1], "near signatures should share cluster");
        assert_ne!(ids[0], ids[2], "far signature should be separate");
    }

    #[test]
    fn multi_key_emit_pick_query_nonempty() {
        let mut addr = MultiKeyAddress::new().with_threshold(0.45);
        // Episode: first-thought answer key + revise key (similar dims) + settle (different)
        addr.push(
            JacobianKey::new(
                sig(&[(10, 0.9), (11, 0.7), (12, 0.4)]),
                KeyPhase::Answer,
                2,
                3840,
            )
            .with_turn(0)
            .with_text_bridge_hash(text_bridge_hash("first thought: spell cat")),
        );
        addr.push(
            JacobianKey::new(
                sig(&[(10, 0.85), (11, 0.65), (12, 0.35)]),
                KeyPhase::Revise,
                40,
                3840,
            )
            .with_turn(0),
        );
        addr.push(
            JacobianKey::new(
                sig(&[(200, 1.0), (201, 0.5)]),
                KeyPhase::Settle,
                48,
                3840,
            )
            .with_turn(0),
        );

        assert_eq!(addr.len(), 3);
        let q = addr.emit_pick_query(8);
        assert!(!q.clusters.is_empty(), "pick set must be non-empty");
        assert!(q.clusters.len() <= 8);
        assert_eq!(q.residual_d, 3840);
        // answer+revise near → same cluster; settle separate → 2 clusters
        assert_eq!(q.clusters.len(), 2, "expected 2 clusters, got {}", q.clusters.len());
        let total_keys: usize = q.clusters.iter().map(|c| c.key_indices.len()).sum();
        assert_eq!(total_keys, 3);
        // top cluster should be the 2-key answer/revise group
        assert_eq!(q.clusters[0].key_indices.len(), 2);
    }

    #[test]
    fn key_phase_from_self_reg_strings() {
        assert_eq!(KeyPhase::from_str_lossy("answer"), Some(KeyPhase::Answer));
        assert_eq!(KeyPhase::from_str_lossy("revise"), Some(KeyPhase::Revise));
        assert_eq!(KeyPhase::from_str_lossy("settle"), Some(KeyPhase::Settle));
        assert_eq!(KeyPhase::from_str_lossy("other"), None);
    }

    #[test]
    fn dim_signature_drops_nonpositive() {
        let s = DimSignature::new(vec![(1, 0.5), (2, 0.0), (3, -1.0), (1, 0.9)]);
        // dim 1 kept once (higher weight first after sort… actually both positive only dim1 and we uniq)
        assert!(s.dims.iter().all(|(_, w)| *w > 0.0));
        assert_eq!(s.dims.iter().filter(|(d, _)| *d == 1).count(), 1);
    }
}