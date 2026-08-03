//! Telemetry records: one memory object, several addresses.
//!
//! A record describes **how a thought opened** at one `(layer, position)` — the model's
//! disposition before it committed to speech — and carries three independent addresses for
//! it. They are not interchangeable, and conflating them is the mistake this module exists
//! to prevent:
//!
//! | door | field | scope | answers |
//! |------|-------|-------|---------|
//! | verbalizable | `text_bridge` | **cross-model** | "what was it leaning toward saying" |
//! | fingerprint | `dim_signature` | **within-model only** | "which internal directions were live" |
//! | rehydration | `state_ref` | within-model, exact | "put the model back in this stance" |
//!
//! `dim_signature` indexes raw residual dimensions. Dimension 1523 in Gemma has no
//! relationship to dimension 1523 in Qwen — residual bases are per-model and arbitrary. So
//! a dim signature can filter or cluster *within* one model's traces and nothing more. Any
//! basin that must hold across Claude / Grok / Gemini has to form on `text_bridge`, which
//! is basis-independent because it is text. This is the same bridge rule as the picker's
//! ("a pick carries text; the host re-embeds in its own residual dim"), arriving from the
//! other side.
//!
//! ## What produced the numbers
//!
//! `lens` records the readout method, and it is load-bearing:
//!
//! - `logit` — the mid-layer residual unembedded directly, no transport. Exact arithmetic,
//!   no fitting, no differencing. This is `jlens.apply(use_jacobian=False)`, the paper's
//!   own baseline, and it is what ships today.
//! - `jacobian` — transported through a fitted `J` first. Not yet available on GGUF; see
//!   `docs/jlens-gguf/CHANGELOG.md` for why finite differences are blocked.
//! - `secant` — a finite-difference response at large ε. **This is not the paper's `J`.**
//!   It is the deployed quantised model's response to a finite nudge. If one is ever
//!   emitted it says `secant`, never `jacobian`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use crate::keys::Readout;

/// Bumped whenever the record shape changes, so a picker reading a cold log can tell what
/// it is holding without guessing.
pub const SCHEMA: &str = "jlens.readout.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LensKind {
    /// Residual unembedded directly. No fit, no differencing.
    Logit,
    /// Transported through a fitted `J`.
    Jacobian,
    /// Finite-difference secant of the quantised model. NOT the paper's `J`.
    Secant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopToken {
    pub id: u32,
    pub text: String,
    pub logit: f32,
}

/// One `(layer, position)` disposition snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutRecord {
    pub schema: String,
    /// Which brain opened the door. A label on one shared field, not a separate store.
    pub source: String,
    pub model_path: String,
    pub lens: LensKind,

    pub layer: usize,
    pub position: usize,
    pub n_layers: usize,
    pub d_model: usize,
    /// The token actually sitting at `position`.
    pub token_id: u32,
    pub token: String,

    // ── door 1: verbalizable, cross-model ──────────────────────────────────────
    pub text_bridge: String,
    pub text_bridge_hash: u64,
    pub top_tokens: Vec<TopToken>,

    // ── door 2: fingerprint, within-model only ─────────────────────────────────
    pub dim_signature: Vec<(usize, f32)>,
    pub residual_norm: f32,

    // ── door 3: rehydration handle ─────────────────────────────────────────────
    /// Path to the saved residual slice, if `--state-dir` was set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_ref: Option<String>,

    /// Free-form tag from the caller — turn id, episode, phase, whatever the picker keys on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

/// Assemble a record from a readout. Does not write the state slice; see [`save_state`].
#[allow(clippy::too_many_arguments)]
pub fn record(
    readout: &Readout,
    lens: LensKind,
    source: &str,
    model_path: &str,
    layer: usize,
    position: usize,
    n_layers: usize,
    d_model: usize,
    token_id: u32,
    token: String,
    tag: Option<String>,
) -> ReadoutRecord {
    let text_bridge = readout.label();
    ReadoutRecord {
        schema: SCHEMA.to_string(),
        source: source.to_string(),
        model_path: model_path.to_string(),
        lens,
        layer,
        position,
        n_layers,
        d_model,
        token_id,
        token,
        text_bridge_hash: gguf_hooks::jacobian::text_bridge_hash(&text_bridge),
        text_bridge,
        top_tokens: readout
            .tokens
            .iter()
            .map(|(id, text, logit)| TopToken {
                id: *id,
                text: text.clone(),
                logit: *logit,
            })
            .collect(),
        dim_signature: readout.top_dims.clone(),
        residual_norm: readout.transport_norm,
        state_ref: None,
        tag,
    }
}

/// Write the residual slice so the stance can be rehydrated later, and return its path.
///
/// Named by content hash rather than by (prompt, layer, position): identical residuals
/// deduplicate, and the name stays stable if the same moment is captured twice.
pub fn save_state(residual: &Tensor, dir: &Path) -> Result<String> {
    let values: Vec<f32> = residual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let digest = fnv1a_bytes(&values);
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating state dir {}", dir.display()))?;
    let path: PathBuf = dir.join(format!("{digest:016x}.safetensors"));

    if !path.exists() {
        let bytes: &[u8] = unsafe {
            // SAFETY: f32 has no padding or invalid bit patterns, and the slice cannot
            // outlive `values`.
            std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                std::mem::size_of_val(values.as_slice()),
            )
        };
        let view = safetensors::tensor::TensorView::new(
            safetensors::tensor::Dtype::F32,
            vec![values.len()],
            bytes,
        )?;
        safetensors::serialize_to_file([("residual".to_string(), view)], None, &path)
            .with_context(|| format!("writing state slice {}", path.display()))?;
    }
    Ok(path.display().to_string())
}

/// FNV-1a over the raw float bytes. Same family as `jacobian::text_bridge_hash`, so both
/// doors hash with the same primitive.
fn fnv1a_bytes(values: &[f32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for value in values {
        for byte in value.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lens_kind_serialises_to_the_label_the_picker_reads() {
        assert_eq!(serde_json::to_string(&LensKind::Logit).unwrap(), "\"logit\"");
        assert_eq!(
            serde_json::to_string(&LensKind::Secant).unwrap(),
            "\"secant\"",
            "a secant must never serialise as `jacobian`"
        );
    }

    #[test]
    fn identical_residuals_hash_to_the_same_state_name() {
        let a = vec![1.0f32, -2.5, 3.25];
        let b = vec![1.0f32, -2.5, 3.25];
        let c = vec![1.0f32, -2.5, 3.26];
        assert_eq!(fnv1a_bytes(&a), fnv1a_bytes(&b));
        assert_ne!(fnv1a_bytes(&a), fnv1a_bytes(&c));
    }
}
