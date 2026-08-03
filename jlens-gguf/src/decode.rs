//! Decode-time capture — reading the disposition *inside* the generated thought stream.
//!
//! ## Why prefill was the wrong moment
//!
//! The stability gate failed at AUC ≈ 0.50 across every layer and position tested, and the
//! positive control explained it: at the prefill hinge a Gemma 4 IT model predicts
//! `<|channel>` for every prompt regardless of subject. It has committed to *opening a
//! thought block* and to nothing else. There is no subject-shaped thing to key on yet.
//!
//! So the commit moves inside the thought stream. This module generates into it and
//! captures the residual at a chosen **content depth**.
//!
//! ## Depth is counted in content tokens, not decode steps
//!
//! The channel header (`<|channel>`, `thought`, `<channel|>`, newlines) varies in length
//! with how the prompt ends. Counting raw decode steps would put N=4 at a different
//! functional depth for different prompts, and the gate would be comparing unlike moments.
//!
//! `N = 0` is anchored at the **first non-control token emitted inside the thought stream**:
//! after the channel header closes, skipping anything that is a marker (`<…>`) or
//! whitespace-only. Every sample then sits at the same depth into the actual thinking.
//!
//! ## Determinism
//!
//! Greedy decode, no sampling. The gate's criterion 3 requires identical input to give
//! identical output, and a sampled decode cannot satisfy that.

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use gguf_hooks::hooks::LayerHook;
use gguf_hooks::loader::Model;
use tokenizers::Tokenizer;

use crate::probe::{CaptureHook, Site};

/// What one generation produced.
pub struct Captured {
    /// `(layer, depth)` → residual `[d_model]` at that content depth.
    pub residuals: BTreeMap<(usize, usize), Tensor>,
    /// Decode step at which depth 0 was anchored, for diagnostics.
    pub anchor_step: Option<usize>,
    /// Everything generated, decoded — so a failed anchor can be eyeballed.
    pub text: String,
    /// Content tokens from the anchor onward.
    pub content: Vec<u32>,
}

/// Is this token structure rather than thought?
///
/// Markers are `<…>`-shaped in the Gemma 4 vocabulary (`<|channel>`, `<channel|>`,
/// `<|turn>`, `<turn|>`, `<eos>`, `<pad>`). Whitespace-only tokens carry no content either.
/// `thought` is *not* special-cased by name — it is skipped because it falls inside the
/// header, before the channel closes.
fn is_control(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    t.starts_with('<') && t.ends_with('>')
}

/// Does this token close the channel header, so that thinking starts after it?
fn closes_channel(text: &str) -> bool {
    text.contains("channel|")
}

/// Greedily generate from `prompt_ids` and capture residuals at the requested content
/// depths.
///
/// `depths` must be sorted ascending. Generation stops once the deepest is captured, EOS is
/// emitted, or `max_steps` is reached.
pub fn decode_and_capture(
    model: &mut Model,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    layers: &[usize],
    depths: &[usize],
    max_steps: usize,
    eos_ids: &[u32],
    // `header_prefilled`: true when the prompt already ends with the thought header, so the
    // model continues a thought rather than opening one. `main.rs:287` pre-fills the header
    // for exactly this reason — asked to open it itself under greedy decode, Gemma 4
    // thrashes into `.\n.\n.\n`.
    header_prefilled: bool,
    device: &Device,
) -> Result<Captured> {
    if prompt_ids.is_empty() {
        bail!("cannot decode from an empty prompt");
    }
    let Some(&max_depth) = depths.last() else {
        bail!("no capture depths requested");
    };
    let sites: Vec<Site> = layers.iter().map(|&l| Site::block_out(l)).collect();

    // Prefill. Nothing captured here — this is precisely the moment that failed the gate.
    model.clear_kv_cache();
    let tokens = Tensor::new(prompt_ids, device)?.unsqueeze(0)?;
    let logits = model.forward(&tokens, 0)?;
    let mut next = argmax(&logits)?;

    let mut residuals = BTreeMap::new();
    let mut anchor_step: Option<usize> = None;
    let mut channel_closed = header_prefilled;
    let mut content_index = 0usize;
    let mut text = String::new();
    let mut content = Vec::new();

    for step in 0..max_steps {
        if eos_ids.contains(&next) {
            break;
        }
        let piece = tokenizer
            .decode(&[next], false)
            .unwrap_or_else(|_| format!("<{next}>"));
        text.push_str(&piece);

        // Feed the token and capture the residual at its position: the disposition the
        // model holds *having just emitted* it.
        let mut hook = CaptureHook::new(sites.clone());
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let index_pos = prompt_ids.len() + step;
        let (logits, _) = model.forward_with_hidden_hooked(
            &input,
            index_pos,
            Some(&mut hook as &mut dyn LayerHook),
        )?;

        if closes_channel(&piece) {
            channel_closed = true;
        } else if !is_control(&piece) {
            // Anchor at the first content token. If the model never opened a channel we
            // still anchor, rather than silently capturing nothing.
            let past_header = channel_closed || step >= 8;
            if past_header {
                if anchor_step.is_none() {
                    anchor_step = Some(step);
                }
                if anchor_step.is_some() {
                    if depths.contains(&content_index) {
                        for &layer in layers {
                            let h = hook.take(Site::block_out(layer)).ok_or_else(|| {
                                anyhow::anyhow!("nothing captured at layer {layer}")
                            })?;
                            let residual =
                                h.i((0, 0))?.flatten_all()?.to_dtype(DType::F32)?;
                            residuals.insert((layer, content_index), residual);
                        }
                    }
                    content.push(next);
                    if content_index >= max_depth {
                        next = argmax(&logits)?;
                        break;
                    }
                    content_index += 1;
                }
            }
        }

        next = argmax(&logits)?;
    }

    Ok(Captured {
        residuals,
        anchor_step,
        text,
        content,
    })
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let vocab = logits.dim(logits.rank() - 1)?;
    let flat = logits.reshape(((), vocab))?;
    let last = flat.narrow(0, flat.dim(0)? - 1, 1)?.to_dtype(DType::F32)?;
    let values: Vec<f32> = last.flatten_all()?.to_vec1()?;
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_value {
            best_value = v;
            best = i;
        }
    }
    Ok(best as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_and_whitespace_are_control_content_is_not() {
        for marker in ["<|channel>", "<channel|>", "<turn|>", "<eos>", "<pad>"] {
            assert!(is_control(marker), "{marker} should be control");
        }
        assert!(is_control("   "));
        assert!(is_control("\n"));
        assert!(!is_control("The"));
        assert!(!is_control(" Italy"));
        // `thought` is content by shape; it is skipped by header position, not by name.
        assert!(!is_control("thought"));
    }

    #[test]
    fn channel_close_is_detected_but_channel_open_is_not() {
        assert!(closes_channel("<channel|>"));
        assert!(!closes_channel("<|channel>"));
        assert!(!closes_channel("thought"));
    }
}
