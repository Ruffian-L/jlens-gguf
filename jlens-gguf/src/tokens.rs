//! Prompt encoding, with the trailing-EOS strip every caller needs.
//!
//! Tokenizer post-processors differ in ways that silently ruin a prefill:
//!
//! | tokenizer | `post_processor.single` | effect on a prompt |
//! |---|---|---|
//! | Gemma 3 | `[<bos>, A, <eos>]` | BOS added **and a trailing `<eos>`** |
//! | Gemma 4 | `[A]` | nothing added — no BOS at all |
//!
//! So Gemma 4 prompts must carry `<bos>` explicitly (see
//! `research_logs/2026-08-02_gemma4_missing_bos.md`), and Gemma 3 prompts arrive with an end
//! token where the model is supposed to start generating. Reading a residual at "the last
//! prompt position" then reads the disposition *at `<eos>`*, which is not a thought about
//! anything.
//!
//! `hydrodynamic-swarm`'s `encode_prompt_no_trailing_eos` handles this for the swarm; this
//! is the same guard for the sidecar.

use anyhow::{bail, Result};
use tokenizers::Tokenizer;

/// Tokens that mean "the turn is over" across the model families this crate loads.
/// Matched on decoded text rather than id, so no variant plumbing is needed.
const TERMINATORS: &[&str] = &[
    "<eos>",
    "<end_of_turn>",
    "<turn|>",
    "<|end_of_text|>",
    "<|eot_id|>",
];

/// Encode `text`, truncate to `max_len`, and strip any trailing turn terminators.
///
/// At least one token is always kept: an all-terminator encode is degenerate, but an empty
/// prefill would be worse.
pub fn encode_prompt(tokenizer: &Tokenizer, text: &str, max_len: usize) -> Result<Vec<u32>> {
    let encoded = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
    let mut ids = encoded.get_ids().to_vec();
    ids.truncate(max_len);

    while ids.len() > 1 {
        let last = *ids.last().expect("checked non-empty");
        let text = tokenizer.decode(&[last], false).unwrap_or_default();
        if TERMINATORS.contains(&text.trim()) {
            ids.pop();
        } else {
            break;
        }
    }

    if ids.is_empty() {
        bail!("prompt tokenised to nothing");
    }
    Ok(ids)
}
