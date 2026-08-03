//! Per-layer residual baselines — what to subtract before a key means anything.
//!
//! ## Why this exists
//!
//! The first telemetry batch keyed on the top-magnitude dimensions of the raw residual.
//! Measured across two paraphrases and one unrelated prompt, the signatures were
//! indistinguishable — weighted Jaccard 0.836 for the paraphrase pair against 0.854 for the
//! unrelated pair at L24, i.e. the *unrelated* prompt scored higher. The text bridges were
//! the same multilingual soup for "currency of Italy" and "why is the sky blue".
//!
//! That is not a bug in the capture; it is a property of transformer residual streams. A
//! handful of dimensions carry enormous, content-independent magnitude (rogue / outlier
//! dimensions, attention-sink structure). Ranking by `|h_i|` finds those every time, so the
//! key describes the model's furniture rather than the thought sitting in it.
//!
//! The fix is to measure what varies. Collect per-dimension mean and standard deviation
//! over a corpus, then key on the standardised deviation `z = (h - μ) / σ`. Constant
//! structure cancels; what is left is how *this* activation differs from what the model
//! usually does at this layer.
//!
//! ## Two different uses, deliberately different treatments
//!
//! - **`dim_signature`** uses the full z-score. It is a ranking, so rescaling per dimension
//!   is exactly the intent — a dimension that always sits at 200 and now sits at 201 should
//!   rank below one that always sits at 0.1 and now sits at 3.
//! - **`text_bridge`** unembeds the *centred* residual `h - μ` only, never the z-score.
//!   Dividing by σ would rotate the residual out of the basis the unembedding matrix
//!   expects, and the decoded tokens would stop meaning anything. Centring alone is
//!   basis-preserving.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use safetensors::tensor::{Dtype, TensorView};

/// Per-layer mean and standard deviation over a corpus.
#[derive(Debug, Clone)]
pub struct Baseline {
    pub d_model: usize,
    /// Positions accumulated per layer — small counts make a baseline untrustworthy.
    pub n_samples: usize,
    pub stats: BTreeMap<usize, (Vec<f32>, Vec<f32>)>,
}

/// Streaming mean/variance accumulator (Welford), one per layer.
pub struct Accumulator {
    d_model: usize,
    n: BTreeMap<usize, usize>,
    mean: BTreeMap<usize, Vec<f64>>,
    m2: BTreeMap<usize, Vec<f64>>,
}

impl Accumulator {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            n: BTreeMap::new(),
            mean: BTreeMap::new(),
            m2: BTreeMap::new(),
        }
    }

    /// Accumulate every row of `[n_positions, d_model]`.
    pub fn push(&mut self, layer: usize, rows: &[f32]) -> Result<()> {
        if rows.len() % self.d_model != 0 {
            bail!(
                "rows ({}) are not a whole multiple of d_model ({})",
                rows.len(),
                self.d_model
            );
        }
        let mean = self
            .mean
            .entry(layer)
            .or_insert_with(|| vec![0f64; self.d_model]);
        let m2 = self
            .m2
            .entry(layer)
            .or_insert_with(|| vec![0f64; self.d_model]);
        let n = self.n.entry(layer).or_insert(0);

        for row in rows.chunks_exact(self.d_model) {
            *n += 1;
            let count = *n as f64;
            for (i, &value) in row.iter().enumerate() {
                let value = value as f64;
                let delta = value - mean[i];
                mean[i] += delta / count;
                m2[i] += delta * (value - mean[i]);
            }
        }
        Ok(())
    }

    /// Finish, converting M2 into a standard deviation.
    ///
    /// Dimensions with no observed variance get σ = 1 rather than 0: dividing by their
    /// (zero) spread would produce infinities, and a dimension that never moves carries no
    /// information about this activation anyway.
    pub fn finish(self) -> Result<Baseline> {
        if self.n.is_empty() {
            bail!("baseline accumulated no samples");
        }
        let mut stats = BTreeMap::new();
        let mut total = 0usize;
        for (layer, n) in self.n.iter() {
            if *n < 2 {
                bail!("layer {layer} saw only {n} positions; a baseline needs at least 2");
            }
            total = total.max(*n);
            let mean = &self.mean[layer];
            let m2 = &self.m2[layer];
            let sd: Vec<f32> = m2
                .iter()
                .map(|v| {
                    let var = v / (*n as f64 - 1.0);
                    let sd = var.sqrt();
                    if sd.is_finite() && sd > 1e-12 {
                        sd as f32
                    } else {
                        1.0
                    }
                })
                .collect();
            stats.insert(
                *layer,
                (mean.iter().map(|v| *v as f32).collect::<Vec<f32>>(), sd),
            );
        }
        Ok(Baseline {
            d_model: self.d_model,
            n_samples: total,
            stats,
        })
    }
}

impl Baseline {
    pub fn layers(&self) -> Vec<usize> {
        self.stats.keys().copied().collect()
    }

    /// `(h - μ)` — basis-preserving, safe to unembed.
    pub fn center(&self, residual: &Tensor, layer: usize, device: &Device) -> Result<Tensor> {
        let (mean, _) = self.require(layer)?;
        let mean = Tensor::from_vec(mean.clone(), self.d_model, device)?;
        Ok(residual.broadcast_sub(&mean)?)
    }

    /// `(h - μ) / σ` — a ranking space, **not** a basis the unembedding understands.
    pub fn standardize(&self, residual: &Tensor, layer: usize, device: &Device) -> Result<Tensor> {
        let (mean, sd) = self.require(layer)?;
        let mean = Tensor::from_vec(mean.clone(), self.d_model, device)?;
        let sd = Tensor::from_vec(sd.clone(), self.d_model, device)?;
        Ok(residual.broadcast_sub(&mean)?.broadcast_div(&sd)?)
    }

    fn require(&self, layer: usize) -> Result<&(Vec<f32>, Vec<f32>)> {
        self.stats.get(&layer).ok_or_else(|| {
            anyhow::anyhow!(
                "baseline has no statistics for layer {layer}; it covers {:?}",
                self.layers()
            )
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut tensors: Vec<(String, TensorView)> = Vec::new();
        for (layer, (mean, sd)) in &self.stats {
            tensors.push((
                format!("mean/{layer}"),
                TensorView::new(Dtype::F32, vec![self.d_model], as_bytes(mean))?,
            ));
            tensors.push((
                format!("sd/{layer}"),
                TensorView::new(Dtype::F32, vec![self.d_model], as_bytes(sd))?,
            ));
        }
        let mut header = std::collections::HashMap::new();
        header.insert("d_model".to_string(), self.d_model.to_string());
        header.insert("n_samples".to_string(), self.n_samples.to_string());
        header.insert(
            "layers".to_string(),
            serde_json::to_string(&self.layers())?,
        );

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        safetensors::serialize_to_file(tensors, Some(header), &tmp)
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Baseline> {
        let buffer = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let (_, header) = safetensors::SafeTensors::read_metadata(&buffer)?;
        let meta = header
            .metadata()
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("{} has no header", path.display()))?;
        let d_model: usize = meta
            .get("d_model")
            .ok_or_else(|| anyhow::anyhow!("{} has no d_model", path.display()))?
            .parse()?;
        let n_samples: usize = meta
            .get("n_samples")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(0);
        let layers: Vec<usize> = serde_json::from_str(
            meta.get("layers")
                .ok_or_else(|| anyhow::anyhow!("{} has no layer list", path.display()))?,
        )?;

        let file = safetensors::SafeTensors::deserialize(&buffer)?;
        let mut stats = BTreeMap::new();
        for layer in layers {
            stats.insert(
                layer,
                (
                    read_f32(&file, &format!("mean/{layer}"))?,
                    read_f32(&file, &format!("sd/{layer}"))?,
                ),
            );
        }
        Ok(Baseline {
            d_model,
            n_samples,
            stats,
        })
    }
}

fn read_f32(file: &safetensors::SafeTensors, name: &str) -> Result<Vec<f32>> {
    let view = file
        .tensor(name)
        .with_context(|| format!("baseline is missing {name}"))?;
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn as_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns; the slice borrows `values`.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_recovers_mean_and_sd() {
        let mut acc = Accumulator::new(2);
        // dim 0: constant 100 (an "outlier" dimension). dim 1: varies.
        acc.push(0, &[100.0, 1.0, 100.0, 3.0, 100.0, 5.0]).unwrap();
        let baseline = acc.finish().unwrap();
        let (mean, sd) = &baseline.stats[&0];
        assert!((mean[0] - 100.0).abs() < 1e-5);
        assert!((mean[1] - 3.0).abs() < 1e-5);
        // A dimension that never moves gets sd = 1, not 0 — no infinities downstream.
        assert_eq!(sd[0], 1.0);
        assert!((sd[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn standardizing_demotes_a_constant_outlier_dimension() {
        let mut acc = Accumulator::new(2);
        acc.push(0, &[100.0, 1.0, 100.0, 3.0, 100.0, 5.0]).unwrap();
        let baseline = acc.finish().unwrap();

        // Raw magnitude says dim 0 dominates; it is the same for every input.
        let h = Tensor::from_vec(vec![100.0f32, 7.0], 2, &Device::Cpu).unwrap();
        let raw: Vec<f32> = h.to_vec1().unwrap();
        assert!(raw[0].abs() > raw[1].abs(), "raw magnitude picks the outlier");

        // Standardised, the dimension that actually moved wins.
        let z: Vec<f32> = baseline
            .standardize(&h, 0, &Device::Cpu)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            z[1].abs() > z[0].abs(),
            "z-score must rank the varying dim above the constant one, got {z:?}"
        );
    }
}
