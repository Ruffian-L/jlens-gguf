//! A fitted lens: the per-layer transports, the readout, and on-disk storage.
//!
//! Port of `jlens/lens.py`. Two differences from the reference, both deliberate:
//!
//! - **Storage is safetensors, not `.pt`.** Pickle has no good Rust reader and the format
//!   is not worth reimplementing. `scripts/lens_pt_to_safetensors.py` converts in both
//!   directions so a Python fit and a Rust fit stay comparable.
//! - **A transport is factored, not a dense `J`.** Forward mode produces `J·u` per probe,
//!   so a rank-`r` fit stores `Y = [d, r]` (`Y[:, i] = J u_i`) alongside its basis
//!   `U = [r, d]`, and transports as `J h ≈ Y (U h)`. The exact fit has `U = I`, which is
//!   stored implicitly.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use gguf_hooks::loader::Model;
use safetensors::tensor::{Dtype, TensorView};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::basis::BasisKind;
use crate::probe::{CaptureHook, Site};

/// Name of a source-position band. `BandId::all()` is the paper's single band.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BandId(String);

impl BandId {
    pub fn all() -> Self {
        Self("all".to_string())
    }

    pub fn named(name: &str) -> Self {
        Self(name.to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BandId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One `(layer, band)` transport.
#[derive(Debug, Clone)]
pub struct TransportBlock {
    /// `[d_model, rank]`, row-major. Column `i` is `J · u_i`.
    pub y: Vec<f32>,
    /// `[rank, d_model]`, row-major. `None` for the exact fit, where the basis is `I`.
    pub u: Option<Vec<f32>>,
    pub rank: usize,
}

/// Fit provenance, written into the safetensors header so a lens file is self-describing.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LensMeta {
    d_model: usize,
    n_prompts: usize,
    n_layers: usize,
    basis_kind: String,
    groups: String,
    eps_rel: f32,
    skip_first: usize,
    max_seq_len: usize,
    /// `(layer, band, rank, has_basis)` — lets a reader check the header without the data.
    blocks: Vec<(usize, String, usize, bool)>,
}

/// A fitted Jacobian lens.
#[derive(Debug, Clone)]
pub struct Lens {
    pub d_model: usize,
    pub n_prompts: usize,
    pub n_layers: usize,
    pub basis_kind: BasisKind,
    pub groups: String,
    pub eps_rel: f32,
    pub skip_first: usize,
    pub max_seq_len: usize,
    pub blocks: BTreeMap<(usize, BandId), TransportBlock>,
}

impl Lens {
    /// Layers this lens was fitted at, ascending.
    pub fn source_layers(&self) -> Vec<usize> {
        let mut layers: Vec<usize> = self.blocks.keys().map(|(l, _)| *l).collect();
        layers.dedup();
        layers
    }

    pub fn bands(&self) -> Vec<BandId> {
        let mut bands: Vec<BandId> = self.blocks.keys().map(|(_, b)| b.clone()).collect();
        bands.sort();
        bands.dedup();
        bands
    }

    /// Map residuals at `layer` into the target basis: `J h`, as `Y (U h)`.
    ///
    /// `residual` is `[n, d_model]`. Anything outside the probe basis's span is dropped —
    /// that is the rank approximation, and it is silent by nature, so compare against an
    /// exact fit before trusting a low rank.
    pub fn transport(&self, residual: &Tensor, layer: usize, band: &BandId) -> Result<Tensor> {
        let block = self
            .blocks
            .get(&(layer, band.clone()))
            .ok_or_else(|| anyhow::anyhow!("lens has no transport for layer {layer} band {band}"))?;
        let device = residual.device();
        let residual = residual.to_dtype(DType::F32)?;

        let coeffs = match &block.u {
            Some(u) => {
                let u = Tensor::from_vec(u.clone(), (block.rank, self.d_model), device)?;
                residual.matmul(&u.t()?.contiguous()?)?
            }
            None => residual,
        };
        let y = Tensor::from_vec(block.y.clone(), (self.d_model, block.rank), device)?;
        Ok(coeffs.matmul(&y.t()?.contiguous()?)?)
    }

    /// Run `model` on `prompt` and read out lens logits at the requested layers.
    ///
    /// Returns `(lens_logits, model_logits, token_ids)`. `lens_logits` maps each layer to
    /// `[n_positions, vocab]`; `model_logits` is the model's own final-layer logits at the
    /// same positions. Mirrors `jlens.lens.JacobianLens.apply`.
    ///
    /// `positions` uses Python-style indexing — negatives count from the end — so the
    /// reference's `positions=[-2]` examples transfer unchanged.
    pub fn apply(
        &self,
        model: &mut Model,
        tokenizer: &Tokenizer,
        prompt: &str,
        layers: &[usize],
        positions: Option<&[i64]>,
        band: &BandId,
        device: &Device,
    ) -> Result<(BTreeMap<usize, Tensor>, Tensor, Vec<u32>)> {
        let n_layers = model.n_layers();
        for &layer in layers {
            if layer >= n_layers {
                bail!("layer {layer} out of range for a {n_layers}-layer model");
            }
            if !self.blocks.contains_key(&(layer, band.clone())) {
                bail!(
                    "layer {layer} was not fitted for band {band}; fitted layers are {:?}",
                    self.source_layers()
                );
            }
        }

        let encoded = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
        let mut ids = encoded.get_ids().to_vec();
        ids.truncate(self.max_seq_len);
        if ids.is_empty() {
            bail!("prompt tokenised to nothing");
        }
        let seq_len = ids.len();
        let tokens = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;

        let final_site = Site::block_out(n_layers - 1);
        let mut sites: Vec<Site> = layers.iter().map(|&l| Site::block_out(l)).collect();
        if !sites.contains(&final_site) {
            sites.push(final_site);
        }

        let mut hook = CaptureHook::new(sites);
        model.clear_kv_cache();
        model.forward_with_hidden_hooked(&tokens, 0, Some(&mut hook))?;

        let wanted: Vec<usize> = match positions {
            None => (0..seq_len).collect(),
            Some(ps) => ps
                .iter()
                .map(|&p| {
                    let resolved = if p < 0 { seq_len as i64 + p } else { p };
                    if resolved < 0 || resolved >= seq_len as i64 {
                        bail!("position {p} out of range for a {seq_len}-token prompt");
                    }
                    Ok(resolved as usize)
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let idx = Tensor::new(
            wanted.iter().map(|&p| p as u32).collect::<Vec<_>>().as_slice(),
            device,
        )?;

        let select = |captured: Option<Tensor>, what: &str| -> Result<Tensor> {
            let h = captured
                .ok_or_else(|| anyhow::anyhow!("no activation captured for {what}"))?;
            Ok(h.narrow(0, 0, 1)?
                .index_select(&idx, 1)?
                .squeeze(0)?
                .to_dtype(DType::F32)?)
        };

        let mut captured: BTreeMap<usize, Tensor> = BTreeMap::new();
        for &layer in layers {
            captured.insert(
                layer,
                select(hook.take(Site::block_out(layer)), &format!("layer {layer}"))?,
            );
        }
        let final_residual = match captured.get(&(n_layers - 1)) {
            Some(h) => h.clone(),
            None => select(hook.take(final_site), "the final layer")?,
        };

        let mut lens_logits = BTreeMap::new();
        for (&layer, residual) in captured.iter() {
            let transported = self.transport(residual, layer, band)?;
            lens_logits.insert(layer, model.unembed(&transported)?.to_dtype(DType::F32)?);
        }
        let model_logits = model.unembed(&final_residual)?.to_dtype(DType::F32)?;

        Ok((lens_logits, model_logits, ids))
    }

    /// Combine lenses fitted on disjoint prompt subsets — an `n_prompts`-weighted mean, as
    /// `jlens.lens.JacobianLens.merge`.
    ///
    /// Inputs must agree on `d_model` and on which `(layer, band)` blocks they hold; a
    /// weighted mean over a *different* set of transports would be a silent lie about what
    /// the result covers.
    pub fn merge(lenses: &[Lens]) -> Result<Lens> {
        let Some(first) = lenses.first() else {
            bail!("merge() needs at least one lens");
        };
        let keys: Vec<_> = first.blocks.keys().cloned().collect();
        for other in &lenses[1..] {
            if other.d_model != first.d_model {
                bail!(
                    "lenses disagree on d_model ({} vs {})",
                    first.d_model,
                    other.d_model
                );
            }
            let other_keys: Vec<_> = other.blocks.keys().cloned().collect();
            if other_keys != keys {
                bail!("lenses disagree on which (layer, band) transports they hold");
            }
        }

        let total: usize = lenses.iter().map(|l| l.n_prompts).sum();
        if total == 0 {
            bail!("every lens reports n_prompts = 0");
        }

        let mut blocks = BTreeMap::new();
        for key in keys {
            let rank = first.blocks[&key].rank;
            let mut y = vec![0f64; first.blocks[&key].y.len()];
            for lens in lenses {
                let block = &lens.blocks[&key];
                if block.rank != rank {
                    bail!("lenses disagree on rank at layer {} band {}", key.0, key.1);
                }
                let w = lens.n_prompts as f64;
                for (slot, value) in y.iter_mut().zip(&block.y) {
                    *slot += w * *value as f64;
                }
            }
            blocks.insert(
                key.clone(),
                TransportBlock {
                    y: y.iter().map(|v| (v / total as f64) as f32).collect(),
                    // The basis is a property of the fit, not of the averaging; all inputs
                    // must already share it for the mean to mean anything.
                    u: first.blocks[&key].u.clone(),
                    rank,
                },
            );
        }

        Ok(Lens {
            n_prompts: total,
            blocks,
            ..first.clone()
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut tensors: Vec<(String, TensorView)> = Vec::new();
        let mut meta_blocks = Vec::new();

        for ((layer, band), block) in &self.blocks {
            if band.as_str().contains('/') {
                bail!("band name {band:?} contains '/', which collides with tensor naming");
            }
            meta_blocks.push((*layer, band.to_string(), block.rank, block.u.is_some()));
            tensors.push((
                format!("J/{layer}/{band}"),
                TensorView::new(
                    Dtype::F32,
                    vec![self.d_model, block.rank],
                    bytemuck_cast(&block.y),
                )?,
            ));
            if let Some(u) = &block.u {
                tensors.push((
                    format!("U/{layer}/{band}"),
                    TensorView::new(
                        Dtype::F32,
                        vec![block.rank, self.d_model],
                        bytemuck_cast(u),
                    )?,
                ));
            }
        }

        let meta = LensMeta {
            d_model: self.d_model,
            n_prompts: self.n_prompts,
            n_layers: self.n_layers,
            basis_kind: self.basis_kind.as_str().to_string(),
            groups: self.groups.clone(),
            eps_rel: self.eps_rel,
            skip_first: self.skip_first,
            max_seq_len: self.max_seq_len,
            blocks: meta_blocks,
        };
        let mut header = HashMap::new();
        header.insert("jlens".to_string(), serde_json::to_string(&meta)?);

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        // Write beside the target then rename, so an interrupted save never leaves a
        // half-written lens that looks loadable.
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        safetensors::serialize_to_file(tensors, Some(header), &tmp)
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Lens> {
        let buffer = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let (_, header) = safetensors::SafeTensors::read_metadata(&buffer)
            .with_context(|| format!("{} is not a safetensors file", path.display()))?;
        let raw = header
            .metadata()
            .as_ref()
            .and_then(|m| m.get("jlens"))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} has no `jlens` header — is it a lens file, or a plain safetensors?",
                    path.display()
                )
            })?;
        let meta: LensMeta = serde_json::from_str(raw)
            .with_context(|| format!("parsing the jlens header in {}", path.display()))?;

        let file = safetensors::SafeTensors::deserialize(&buffer)?;
        let mut blocks = BTreeMap::new();
        for (layer, band, rank, has_basis) in &meta.blocks {
            let y = read_f32(&file, &format!("J/{layer}/{band}"))?;
            let u = if *has_basis {
                Some(read_f32(&file, &format!("U/{layer}/{band}"))?)
            } else {
                None
            };
            blocks.insert(
                (*layer, BandId::named(band)),
                TransportBlock {
                    y,
                    u,
                    rank: *rank,
                },
            );
        }

        Ok(Lens {
            d_model: meta.d_model,
            n_prompts: meta.n_prompts,
            n_layers: meta.n_layers,
            basis_kind: match meta.basis_kind.as_str() {
                "identity" => BasisKind::Identity,
                "random" => BasisKind::Random,
                _ => BasisKind::ResidualSpan,
            },
            groups: meta.groups,
            eps_rel: meta.eps_rel,
            skip_first: meta.skip_first,
            max_seq_len: meta.max_seq_len,
            blocks,
        })
    }
}

fn read_f32(file: &safetensors::SafeTensors, name: &str) -> Result<Vec<f32>> {
    let view = file
        .tensor(name)
        .with_context(|| format!("lens file is missing tensor {name}"))?;
    if view.dtype() != Dtype::F32 {
        bail!("{name} is {:?}, expected F32", view.dtype());
    }
    let data = view.data();
    Ok(data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// `&[f32] -> &[u8]` without a `bytemuck` dependency. Safe: f32 has no padding or
/// invalid bit patterns, and the result borrows the same lifetime.
fn bytemuck_cast(values: &[f32]) -> &[u8] {
    // SAFETY: f32 is Copy with no niches; every bit pattern is a valid u8 sequence, and
    // the returned slice cannot outlive `values`.
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_lens() -> Lens {
        // d_model = 2, exact fit, J = [[1, 2], [3, 4]] stored row-major.
        let mut blocks = BTreeMap::new();
        blocks.insert(
            (0, BandId::all()),
            TransportBlock {
                y: vec![1.0, 2.0, 3.0, 4.0],
                u: None,
                rank: 2,
            },
        );
        Lens {
            d_model: 2,
            n_prompts: 3,
            n_layers: 4,
            basis_kind: BasisKind::Identity,
            groups: "paper".to_string(),
            eps_rel: 1e-2,
            skip_first: 16,
            max_seq_len: 128,
            blocks,
        }
    }

    #[test]
    fn transport_applies_j_not_j_transpose() {
        let lens = tiny_lens();
        // h = [1, 0] must select the first *column* of J: J·e_0 = [1, 3].
        let h = Tensor::from_vec(vec![1f32, 0.0], (1, 2), &Device::Cpu).unwrap();
        let out: Vec<Vec<f32>> = lens
            .transport(&h, 0, &BandId::all())
            .unwrap()
            .to_vec2()
            .unwrap();
        assert_eq!(out[0], vec![1.0, 3.0]);
    }

    #[test]
    fn factored_transport_matches_the_dense_one() {
        // Same map expressed in a rotated rank-2 basis must transport identically.
        let lens = tiny_lens();
        let h = Tensor::from_vec(vec![0.5f32, -1.5], (1, 2), &Device::Cpu).unwrap();
        let dense: Vec<Vec<f32>> = lens
            .transport(&h, 0, &BandId::all())
            .unwrap()
            .to_vec2()
            .unwrap();

        let s = std::f32::consts::FRAC_1_SQRT_2;
        let u = vec![s, s, s, -s]; // orthonormal [2, 2]
        // Y[:, i] = J u_i
        let y = vec![
            1.0 * s + 2.0 * s,
            1.0 * s - 2.0 * s,
            3.0 * s + 4.0 * s,
            3.0 * s - 4.0 * s,
        ];
        let mut factored = lens.clone();
        factored.blocks.insert(
            (0, BandId::all()),
            TransportBlock {
                y,
                u: Some(u),
                rank: 2,
            },
        );
        let got: Vec<Vec<f32>> = factored
            .transport(&h, 0, &BandId::all())
            .unwrap()
            .to_vec2()
            .unwrap();
        for (a, b) in dense[0].iter().zip(&got[0]) {
            assert!((a - b).abs() < 1e-5, "dense {a} vs factored {b}");
        }
    }

    #[test]
    fn merge_is_an_n_prompts_weighted_mean() {
        let mut a = tiny_lens();
        a.n_prompts = 1;
        let mut b = tiny_lens();
        b.n_prompts = 3;
        b.blocks.get_mut(&(0, BandId::all())).unwrap().y = vec![5.0, 6.0, 7.0, 8.0];

        let merged = Lens::merge(&[a, b]).unwrap();
        assert_eq!(merged.n_prompts, 4);
        // (1·1 + 3·5) / 4 = 4
        assert_eq!(merged.blocks[&(0, BandId::all())].y[0], 4.0);
    }

    #[test]
    fn merge_rejects_mismatched_block_sets() {
        let a = tiny_lens();
        let mut b = tiny_lens();
        b.blocks.insert(
            (1, BandId::all()),
            TransportBlock {
                y: vec![0.0; 4],
                u: None,
                rank: 2,
            },
        );
        assert!(Lens::merge(&[a, b]).is_err());
    }

    #[test]
    fn save_then_load_round_trips() {
        let lens = tiny_lens();
        let dir = std::env::temp_dir().join(format!("jlens-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lens.safetensors");
        lens.save(&path).unwrap();
        let back = Lens::load(&path).unwrap();
        assert_eq!(back.d_model, lens.d_model);
        assert_eq!(back.n_prompts, lens.n_prompts);
        assert_eq!(back.groups, lens.groups);
        assert_eq!(
            back.blocks[&(0, BandId::all())].y,
            lens.blocks[&(0, BandId::all())].y
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
