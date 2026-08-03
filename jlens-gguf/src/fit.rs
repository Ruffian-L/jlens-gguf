//! Fitting the Jacobian lens by forward-mode finite differences.
//!
//! Port of `jlens/fitting.py`, with reverse mode replaced by the identity derived in
//! `docs/jlens-gguf/DESIGN.md` §3. jlens computes, per source layer:
//!
//! ```text
//! J_l[i,:] = mean_{p∈V}  ∂/∂h_l[p]  Σ_{p'∈V} h_tgt[p', i]
//! ```
//!
//! by placing a one-hot cotangent at output dim `i` across every valid target position and
//! backpropagating. Candle has no gradient through quantised matmul, so instead: perturb
//! `h_l[p] += εv` at **every** `p` in the band at once and sum the target-side change over
//! `p' ∈ V`. Causality zeroes `J[p'←p]` for `p' < p`, so
//!
//! ```text
//! J_l v = ( Σ_{p'∈V} Δh_tgt[p'] ) / (ε·|band|)
//! ```
//!
//! is the same estimator, not an approximation of it — the cross-terms that would normally
//! make simultaneous perturbation useless are exactly the terms being summed.
//!
//! Cost is one batched prefill per probe direction (two, with central differences), where
//! batch element `b` carries direction `b`.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use gguf_hooks::loader::Model;
use tokenizers::Tokenizer;

use crate::basis::{orthonormalize, random_basis, Basis, BasisKind};
use crate::lens::{BandId, Lens, TransportBlock};
use crate::probe::{band_mask, CaptureHook, ProbeHook, Site};

/// Positions before this index are excluded from the average; early positions act as
/// attention sinks and have atypical residual statistics. Same constant and rationale as
/// `jlens.fitting.SKIP_FIRST_N_POSITIONS`.
pub const SKIP_FIRST_N_POSITIONS: usize = 16;

/// Sequence positions to include, port of `jlens.fitting.valid_position_mask`.
///
/// Early positions are dominated by attention-sink behaviour and the final position has no
/// next-token target, so both are excluded.
pub fn valid_positions(seq_len: usize, skip_first: usize) -> Result<Vec<usize>> {
    if seq_len <= skip_first + 1 {
        bail!("prompt too short: seq_len={seq_len}, need > {}", skip_first + 1);
    }
    Ok((skip_first..seq_len - 1).collect())
}

/// How valid positions are partitioned into separately-fitted transports.
///
/// `Paper` reproduces jlens exactly. The others exist because averaging over all source
/// positions collapses first-thought / revise / settle into one transport, which is
/// precisely the distinction phase-edge keys need — see `DESIGN.md` §5.
#[derive(Debug, Clone)]
pub enum PositionGroups {
    /// One band covering every valid position.
    Paper,
    /// Three equal positional bands. A cheap proxy for phase structure — it assumes phases
    /// fall in sequence order, which is an assumption, not a measurement.
    Thirds,
    /// Explicit per-token phase labels, one `Vec<String>` per prompt.
    Labels(Vec<Vec<String>>),
}

impl PositionGroups {
    /// Parse `paper` | `thirds` | `labels:<file.json>`.
    ///
    /// The labels file is a JSON array with one entry per prompt, each entry an array of
    /// per-token phase names.
    pub fn parse(spec: &str) -> Result<Self> {
        match spec {
            "paper" => Ok(Self::Paper),
            "thirds" => Ok(Self::Thirds),
            other => {
                let path = other.strip_prefix("labels:").ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown --position-groups {other:?}; expected paper, thirds, or labels:<file.json>"
                    )
                })?;
                let raw = std::fs::read_to_string(path)
                    .with_context(|| format!("reading position labels from {path}"))?;
                let labels: Vec<Vec<String>> = serde_json::from_str(&raw)
                    .with_context(|| format!("parsing {path} as [[String]] (one array per prompt)"))?;
                if labels.is_empty() {
                    bail!("{path} contains no label rows");
                }
                Ok(Self::Labels(labels))
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Paper => "paper",
            Self::Thirds => "thirds",
            Self::Labels(_) => "labels",
        }
    }

    /// Split `valid` into named bands for `prompt_idx`.
    ///
    /// Bands that come out empty are dropped: a band with no positions has no estimator,
    /// and silently dividing by zero would produce a transport of infinities.
    fn bands(&self, valid: &[usize], prompt_idx: usize) -> Result<Vec<(BandId, Vec<usize>)>> {
        let bands = match self {
            Self::Paper => vec![(BandId::all(), valid.to_vec())],
            Self::Thirds => {
                let n = valid.len();
                let cut = [0, n / 3, 2 * n / 3, n];
                ["early", "middle", "late"]
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        (BandId::named(name), valid[cut[i]..cut[i + 1]].to_vec())
                    })
                    .collect()
            }
            Self::Labels(labels) => {
                let row = labels.get(prompt_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "position labels have {} rows but prompt {prompt_idx} was requested",
                        labels.len()
                    )
                })?;
                let mut grouped: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
                for &p in valid {
                    let label = row.get(p).ok_or_else(|| {
                        anyhow::anyhow!(
                            "prompt {prompt_idx} has {} labels but the tokenised prompt reaches \
                             position {p}; labels must cover every token",
                            row.len()
                        )
                    })?;
                    grouped.entry(label.as_str()).or_default().push(p);
                }
                grouped
                    .into_iter()
                    .map(|(name, ps)| (BandId::named(name), ps))
                    .collect()
            }
        };
        Ok(bands.into_iter().filter(|(_, ps)| !ps.is_empty()).collect())
    }
}

/// Everything the fit needs that isn't the model or the corpus.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Source layers to fit. Block outputs, `0..n_layers`.
    pub source_layers: Vec<usize>,
    /// Probe directions per layer. `None` means the exact fit (`rank == d_model`).
    pub rank: Option<usize>,
    pub basis_kind: BasisKind,
    /// Probe directions carried per forward pass — the batch axis.
    pub probe_batch: usize,
    /// ε as a fraction of the source residual's RMS. Absolute ε is meaningless across
    /// sites and models because Gemma scales embeddings by √d before layer 0.
    pub eps_rel: f32,
    pub max_seq_len: usize,
    pub skip_first: usize,
    pub groups: PositionGroups,
    pub seed: u64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            source_layers: Vec::new(),
            rank: Some(256),
            basis_kind: BasisKind::ResidualSpan,
            probe_batch: 8,
            eps_rel: 1e-2,
            max_seq_len: 128,
            skip_first: SKIP_FIRST_N_POSITIONS,
            groups: PositionGroups::Paper,
            seed: 0,
        }
    }
}

/// Per-prompt diagnostics, mirroring the ones `jlens.fitting.fit` logs.
#[derive(Debug, Clone, Copy)]
pub struct PromptStats {
    pub seq_len: usize,
    pub n_valid: usize,
    /// `max_l ‖J_l‖ / √d` — flags heavy-tailed prompts that dominate the mean.
    pub norm_over_sqrt_d: f32,
    pub seconds: f64,
}

/// Tokenise and truncate, returning `[1, seq]` on `device`.
fn encode(tokenizer: &Tokenizer, prompt: &str, max_seq_len: usize, device: &Device) -> Result<Tensor> {
    let encoded = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
    let mut ids = encoded.get_ids().to_vec();
    ids.truncate(max_seq_len);
    if ids.is_empty() {
        bail!("prompt tokenised to nothing");
    }
    Ok(Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?)
}

/// One prefill with `hook` installed. `index_pos = 0` makes the model ignore any stale KV,
/// so every probe sees the same clean prefill.
fn prefill(model: &mut Model, tokens: &Tensor, hook: &mut dyn gguf_hooks::hooks::LayerHook) -> Result<()> {
    model.clear_kv_cache();
    model.forward_with_hidden_hooked(tokens, 0, Some(hook))?;
    Ok(())
}

/// Root-mean-square of `h[0, positions, :]` — the scale relative ε is a fraction of.
fn residual_rms(h: &Tensor, positions: &[usize], device: &Device) -> Result<f32> {
    let idx = Tensor::new(
        positions.iter().map(|&p| p as u32).collect::<Vec<_>>().as_slice(),
        device,
    )?;
    let selected = h.narrow(0, 0, 1)?.index_select(&idx, 1)?.to_dtype(DType::F32)?;
    Ok(selected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
}

/// One chunk of probe directions → the corresponding columns of `J`.
///
/// This is the estimator itself, and everything else in this module is bookkeeping around
/// it. Each row of `dirs` is perturbed into every position of the band at once; the
/// target-side change is summed over `target_idx` and divided by `2ε·|band|`.
///
/// Returns `[n_dirs, d_model]`, row `i` being `J · dirs[i]`.
#[allow(clippy::too_many_arguments)]
pub fn probe_columns(
    model: &mut Model,
    tokens: &Tensor,
    dirs: &Tensor,
    eps: f32,
    source: Site,
    target: Site,
    band_mask: &Tensor,
    band_len: usize,
    target_idx: &Tensor,
) -> Result<Tensor> {
    if band_len == 0 {
        bail!("cannot probe an empty position band");
    }
    let n_dirs = dirs.dim(0)?;
    let batched = tokens.repeat((n_dirs, 1))?;

    let plus = (dirs * eps as f64)?;
    let mut hook = ProbeHook::new(source, target, &plus, band_mask);
    prefill(model, &batched, &mut hook)?;
    let h_plus = hook
        .take()
        .ok_or_else(|| anyhow::anyhow!("probe captured no activation at the target site"))?;

    let minus = (dirs * (-(eps as f64)))?;
    let mut hook = ProbeHook::new(source, target, &minus, band_mask);
    prefill(model, &batched, &mut hook)?;
    let h_minus = hook
        .take()
        .ok_or_else(|| anyhow::anyhow!("probe captured no activation at the target site"))?;

    let diff = (h_plus.to_dtype(DType::F32)? - h_minus.to_dtype(DType::F32)?)?;
    let summed = diff.index_select(target_idx, 1)?.sum(1)?;
    let scale = 1.0 / (2.0 * eps as f64 * band_len as f64);
    Ok((summed * scale)?)
}

/// Build the probe basis for each source layer from residuals observed on `prompts`.
///
/// Samples are taken from the baseline pass only — no probes are spent here. `Identity`
/// needs no samples at all and short-circuits.
pub fn build_bases(
    model: &mut Model,
    tokenizer: &Tokenizer,
    prompts: &[String],
    cfg: &FitConfig,
    device: &Device,
    verbose: bool,
) -> Result<BTreeMap<usize, Basis>> {
    let d_model = model.token_embeddings().dim(1)?;
    let rank = cfg.rank.unwrap_or(d_model).min(d_model);
    let mut bases = BTreeMap::new();

    if cfg.basis_kind == BasisKind::Identity || rank == d_model {
        for &layer in &cfg.source_layers {
            bases.insert(layer, Basis::identity(d_model));
        }
        return Ok(bases);
    }

    if cfg.basis_kind == BasisKind::Random {
        for &layer in &cfg.source_layers {
            let rows = random_basis(d_model, rank, cfg.seed ^ layer as u64)?;
            bases.insert(layer, Basis::from_rows(BasisKind::Random, d_model, rows)?);
        }
        return Ok(bases);
    }

    // Residual-span: collect samples, then orthonormalise per layer. Over-sample by 2× so
    // dependent rows can be dropped and still leave a full-rank basis.
    let target_samples = rank * 2;
    let mut samples: BTreeMap<usize, Vec<f32>> = cfg
        .source_layers
        .iter()
        .map(|&l| (l, Vec::<f32>::with_capacity(target_samples * d_model)))
        .collect();
    let sites: Vec<Site> = cfg.source_layers.iter().map(|&l| Site::block_out(l)).collect();

    for (i, prompt) in prompts.iter().enumerate() {
        if samples.values().all(|s| s.len() >= target_samples * d_model) {
            break;
        }
        let tokens = match encode(tokenizer, prompt, cfg.max_seq_len, device) {
            Ok(t) => t,
            Err(e) => {
                if verbose {
                    eprintln!("  basis: skipping prompt {i}: {e}");
                }
                continue;
            }
        };
        let seq_len = tokens.dim(1)?;
        let valid = match valid_positions(seq_len, cfg.skip_first) {
            Ok(v) => v,
            Err(e) => {
                if verbose {
                    eprintln!("  basis: skipping prompt {i}: {e}");
                }
                continue;
            }
        };

        let mut hook = CaptureHook::new(sites.clone());
        prefill(model, &tokens, &mut hook)?;

        for &layer in &cfg.source_layers {
            let slot = samples.get_mut(&layer).expect("layer was inserted above");
            if slot.len() >= target_samples * d_model {
                continue;
            }
            let h = hook
                .take(Site::block_out(layer))
                .ok_or_else(|| anyhow::anyhow!("no activation captured at layer {layer}"))?;
            // Spread samples across the sequence rather than taking a contiguous run, so a
            // single prompt does not contribute one narrow region of the residual space.
            let stride = (valid.len() / 8).max(1);
            for &p in valid.iter().step_by(stride) {
                if slot.len() >= target_samples * d_model {
                    break;
                }
                let row = h.i((0, p))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                slot.extend_from_slice(&row);
            }
        }
    }

    for (&layer, rows) in samples.iter() {
        if rows.is_empty() {
            bail!("no residual samples collected at layer {layer}");
        }
        let basis = orthonormalize(rows, d_model, rank)?;
        let got = basis.len() / d_model;
        if verbose {
            println!(
                "  basis L{layer}: rank {got}/{rank} from {} samples",
                rows.len() / d_model
            );
        }
        bases.insert(
            layer,
            Basis::from_rows(BasisKind::ResidualSpan, d_model, basis)?,
        );
    }
    Ok(bases)
}

/// Fit `J` over `prompts`, accumulating a running mean exactly as `jlens.fitting.fit` does.
///
/// `on_prompt` is called after each prompt with its diagnostics so callers can log or
/// checkpoint without this function owning either concern.
pub fn fit(
    model: &mut Model,
    tokenizer: &Tokenizer,
    prompts: &[String],
    cfg: &FitConfig,
    bases: &BTreeMap<usize, Basis>,
    device: &Device,
    mut on_prompt: impl FnMut(usize, &PromptStats),
) -> Result<Lens> {
    let d_model = model.token_embeddings().dim(1)?;
    let n_layers = model.n_layers();
    for &layer in &cfg.source_layers {
        if layer >= n_layers {
            bail!("source layer {layer} out of range for a {n_layers}-layer model");
        }
    }
    let target = Site::block_out(n_layers - 1);

    // Running sums, keyed by (layer, band). Shape [d_model, rank]: column i is J·u_i.
    let mut sums: BTreeMap<(usize, BandId), Vec<f64>> = BTreeMap::new();
    let mut counts: BTreeMap<(usize, BandId), usize> = BTreeMap::new();
    let mut n_done = 0usize;
    let sqrt_d = (d_model as f32).sqrt();

    let capture_sites: Vec<Site> = cfg.source_layers.iter().map(|&l| Site::block_out(l)).collect();

    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        let started = std::time::Instant::now();
        let tokens = match encode(tokenizer, prompt, cfg.max_seq_len, device) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  skipping prompt {prompt_idx}: {e}");
                continue;
            }
        };
        let seq_len = tokens.dim(1)?;
        let valid = match valid_positions(seq_len, cfg.skip_first) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  skipping prompt {prompt_idx}: {e}");
                continue;
            }
        };
        let bands = cfg.groups.bands(&valid, prompt_idx)?;

        // Target positions are always the full valid set, independent of the source band —
        // the estimator sums cotangents over every valid target.
        let target_idx = Tensor::new(
            valid.iter().map(|&p| p as u32).collect::<Vec<_>>().as_slice(),
            device,
        )?;

        // One baseline prefill supplies the ε scale for every source layer.
        let mut baseline = CaptureHook::new(capture_sites.clone());
        prefill(model, &tokens, &mut baseline)?;
        let mut eps_for_layer: BTreeMap<usize, f32> = BTreeMap::new();
        for &layer in &cfg.source_layers {
            let h = baseline
                .take(Site::block_out(layer))
                .ok_or_else(|| anyhow::anyhow!("no activation captured at layer {layer}"))?;
            let rms = residual_rms(&h, &valid, device)?;
            if !rms.is_finite() || rms <= 0.0 {
                bail!("layer {layer} residual RMS is {rms} on prompt {prompt_idx}");
            }
            eps_for_layer.insert(layer, cfg.eps_rel * rms);
        }

        let mut prompt_max_norm = 0f32;

        for &layer in &cfg.source_layers {
            let basis = bases
                .get(&layer)
                .ok_or_else(|| anyhow::anyhow!("no probe basis for layer {layer}"))?;
            let eps = eps_for_layer[&layer];
            let rank = basis.rank();
            let source = Site::block_out(layer);

            for (band_id, positions) in &bands {
                let mask = band_mask(seq_len, positions, device)?;
                let key = (layer, band_id.clone());
                let sum = sums
                    .entry(key.clone())
                    .or_insert_with(|| vec![0f64; d_model * rank]);

                let mut start = 0usize;
                while start < rank {
                    let len = cfg.probe_batch.min(rank - start);
                    let dirs = basis.chunk(start, len, device)?; // [len, d]
                    let cols: Vec<Vec<f32>> = probe_columns(
                        model,
                        &tokens,
                        &dirs,
                        eps,
                        source,
                        target,
                        &mask,
                        positions.len(),
                        &target_idx,
                    )?
                    .to_vec2()?;

                    // Store column-major within the [d_model, rank] block: probe i fills
                    // column i, because forward mode yields J·u_i, not a row of J.
                    for (row, col) in cols.iter().enumerate() {
                        let i = start + row;
                        for (r, &value) in col.iter().enumerate() {
                            sum[r * rank + i] += value as f64;
                        }
                    }
                    start += len;
                }

                *counts.entry(key).or_insert(0) += 1;

                let frob = sum
                    .iter()
                    .map(|v| (v / counts[&(layer, band_id.clone())] as f64).powi(2))
                    .sum::<f64>()
                    .sqrt() as f32;
                prompt_max_norm = prompt_max_norm.max(frob / sqrt_d);
            }
        }

        n_done += 1;
        let stats = PromptStats {
            seq_len,
            n_valid: valid.len(),
            norm_over_sqrt_d: prompt_max_norm,
            seconds: started.elapsed().as_secs_f64(),
        };
        on_prompt(prompt_idx, &stats);
    }

    if n_done == 0 {
        bail!("no prompts were long enough to fit on");
    }

    let mut blocks = BTreeMap::new();
    for ((layer, band), sum) in sums {
        let n = counts[&(layer, band.clone())].max(1);
        let rank = bases[&layer].rank();
        let y: Vec<f32> = sum.iter().map(|v| (v / n as f64) as f32).collect();
        let basis = &bases[&layer];
        blocks.insert(
            (layer, band),
            TransportBlock {
                y,
                u: if basis.is_identity() {
                    None
                } else {
                    Some(basis.rows().to_vec())
                },
                rank,
            },
        );
    }

    Ok(Lens {
        d_model,
        n_prompts: n_done,
        n_layers,
        basis_kind: cfg.basis_kind,
        groups: cfg.groups.as_str().to_string(),
        eps_rel: cfg.eps_rel,
        skip_first: cfg.skip_first,
        max_seq_len: cfg.max_seq_len,
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_positions_match_the_reference_mask() {
        // jlens: mask[skip_first : seq_len - 1] = True
        assert_eq!(valid_positions(20, 16).unwrap(), vec![16, 17, 18]);
        assert!(valid_positions(17, 16).is_err(), "seq_len 17 leaves no valid positions");
    }

    #[test]
    fn thirds_partition_without_overlap_or_loss() {
        let valid: Vec<usize> = (16..40).collect();
        let bands = PositionGroups::Thirds.bands(&valid, 0).unwrap();
        assert_eq!(bands.len(), 3);
        let mut all: Vec<usize> = bands.iter().flat_map(|(_, ps)| ps.clone()).collect();
        all.sort_unstable();
        assert_eq!(all, valid, "thirds must cover every valid position exactly once");
    }

    #[test]
    fn paper_group_is_one_band_over_everything() {
        let valid: Vec<usize> = (16..30).collect();
        let bands = PositionGroups::Paper.bands(&valid, 0).unwrap();
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0].1, valid);
    }

    #[test]
    fn labels_group_by_name_and_reject_short_rows() {
        let mut row = vec!["answer".to_string(); 20];
        row[18] = "revise".to_string();
        let groups = PositionGroups::Labels(vec![row]);
        let valid: Vec<usize> = (16..19).collect();
        let bands = groups.bands(&valid, 0).unwrap();
        assert_eq!(bands.len(), 2);

        let short = PositionGroups::Labels(vec![vec!["answer".to_string(); 3]]);
        assert!(short.bands(&valid, 0).is_err(), "labels must cover every token");
    }
}
