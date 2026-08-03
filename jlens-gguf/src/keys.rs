//! Bridging lens readouts into hydro's multi-key address schema.
//!
//! `research_logs/2026-08-02_jacobian_lens_repo_vs_hydro_fd.md` sets out what this lane is
//! for: fitted `J_l` → transport a mid-stack residual → unembed → top tokens → verbalizable
//! label and text-bridge content for the SplatRAG picker. Two things follow from that.
//!
//! **The bridge rule.** A pick carries *text* (or token ids); the host embeds it in **its**
//! residual dim. Nothing here ever hands a `d_model` vector across the boundary, which is
//! why the key stores a hash of the decoded tokens rather than the transported residual.
//!
//! **The signature is a transport fingerprint, not the FD proxy's.** Hydro's
//! `jacobian::DimSignature` is normally built from local `∂logits/∂h` sensitivity. Here it
//! comes from the dominant dimensions of `J_l h` — a different measurement in the same
//! schema. Both index the same `MultiKeyAddress`; which is more stable is an open
//! experiment, so keys record which one produced them.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};
use gguf_hooks::jacobian::{text_bridge_hash, DimSignature, JacobianKey, KeyPhase};
use tokenizers::Tokenizer;

use crate::lens::BandId;

/// What the lens says an activation is disposed to make the model say.
#[derive(Debug, Clone)]
pub struct Readout {
    /// Top vocabulary tokens by lens logit, most likely first.
    pub tokens: Vec<(u32, String, f32)>,
    /// Dominant dimensions of the transported residual, `(dim, |value|)` descending.
    pub top_dims: Vec<(usize, f32)>,
    /// L2 norm of the transported residual.
    pub transport_norm: f32,
}

impl Readout {
    /// The decoded top tokens joined into one short string — the verbalizable label, and
    /// the text the bridge hash is taken over.
    pub fn label(&self) -> String {
        self.tokens
            .iter()
            .map(|(_, text, _)| text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Read out one transported residual: top tokens and dominant dims.
///
/// `transported` is `[d_model]` or `[1, d_model]`; `logits` is the matching `[vocab]` or
/// `[1, vocab]` from `Model::unembed`.
pub fn read_out(
    transported: &Tensor,
    logits: &Tensor,
    tokenizer: &Tokenizer,
    top_k: usize,
    top_dims: usize,
) -> Result<Readout> {
    let vector = flatten(transported)?;
    let scores = flatten(logits)?;
    if top_k == 0 {
        bail!("top_k must be at least 1 for a readout to say anything");
    }

    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(top_k);

    let tokens = ranked
        .into_iter()
        .map(|(id, score)| {
            let text = tokenizer
                .decode(&[id as u32], false)
                .unwrap_or_else(|_| format!("<{id}>"));
            (id as u32, text, score)
        })
        .collect();

    let mut dims: Vec<(usize, f32)> = vector
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (i, v.abs()))
        .filter(|(_, v)| v.is_finite() && *v > 0.0)
        .collect();
    dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    dims.truncate(top_dims);

    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();

    Ok(Readout {
        tokens,
        top_dims: dims,
        transport_norm: norm,
    })
}

/// Which generation phase a band corresponds to.
///
/// Only bands that were *named* for a phase map to one. Positional bands (`all`, `early`,
/// `middle`, `late`) return `None`: assuming "late positions are the settle phase" would
/// put a guess into the key where a measurement belongs.
pub fn phase_for_band(band: &BandId) -> Option<KeyPhase> {
    KeyPhase::from_str_lossy(band.as_str())
}

/// Build a `JacobianKey` from a lens readout.
///
/// The signature is the transport fingerprint; the text bridge is the decoded top tokens.
/// `residual_d` must be the *host's* live residual width — a key measured at one width
/// cannot address a host at another.
pub fn key_from_readout(
    readout: &Readout,
    phase: KeyPhase,
    step: usize,
    turn: Option<usize>,
    residual_d: usize,
    top_k: usize,
) -> Result<JacobianKey> {
    let signature = DimSignature::from_top_dimensions(&readout.top_dims, top_k);
    if signature.is_empty() {
        bail!("transported residual had no positive dimensions — nothing to address with");
    }
    let mut key = JacobianKey::new(signature, phase, step, residual_d)
        .with_text_bridge_hash(text_bridge_hash(&readout.label()))
        .with_sensitivity_norm(readout.transport_norm);
    if let Some(turn) = turn {
        key = key.with_turn(turn);
    }
    Ok(key)
}

fn flatten(t: &Tensor) -> Result<Vec<f32>> {
    let t = t.to_dtype(DType::F32)?;
    let n = t.elem_count();
    Ok(t.reshape(n)?.to_vec1::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn only_phase_named_bands_map_to_a_phase() {
        assert_eq!(phase_for_band(&BandId::named("revise")), Some(KeyPhase::Revise));
        assert_eq!(phase_for_band(&BandId::named("settle")), Some(KeyPhase::Settle));
        assert_eq!(phase_for_band(&BandId::named("answer")), Some(KeyPhase::Answer));
        // Positional bands carry no phase claim.
        assert_eq!(phase_for_band(&BandId::all()), None);
        assert_eq!(phase_for_band(&BandId::named("late")), None);
    }

    #[test]
    fn top_dims_rank_by_magnitude_not_sign() {
        let transported =
            Tensor::from_vec(vec![0.1f32, -5.0, 2.0, 0.0], 4, &Device::Cpu).unwrap();
        let logits = Tensor::from_vec(vec![1.0f32, 3.0, 2.0], 3, &Device::Cpu).unwrap();
        let tokenizer_missing = read_out_dims(&transported, &logits);
        assert_eq!(tokenizer_missing[0].0, 1, "-5.0 is the largest magnitude");
        assert_eq!(tokenizer_missing[1].0, 2);
        // The zero dimension is dropped rather than ranked last.
        assert_eq!(tokenizer_missing.len(), 3);
    }

    /// `read_out` minus the tokenizer, so the dim ranking can be tested on its own.
    fn read_out_dims(transported: &Tensor, _logits: &Tensor) -> Vec<(usize, f32)> {
        let vector = flatten(transported).unwrap();
        let mut dims: Vec<(usize, f32)> = vector
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (i, v.abs()))
            .filter(|(_, v)| v.is_finite() && *v > 0.0)
            .collect();
        dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        dims
    }
}
