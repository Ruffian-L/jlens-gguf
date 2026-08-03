//! GGUF model loading, shared by the swarm binary and the `jlens-gguf` sidecar.
//!
//! Everything here was lifted verbatim out of `main.rs` (the `Model` dispatch enum and the
//! inline load block) so that a second crate can load the *same* weights through the *same*
//! path. Two loaders drift; `src/bin/field_audit.rs` already shows what that costs.
//!
//! Arch sniffing, the Gemma 3n bail, and the "Gemma 4 must not silently take the Gemma 3
//! tokenizer" rule are behaviour-preserving copies — do not simplify them without checking
//! why each one is there.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::hooks::LayerHook;
use crate::{gemma, gemma4, llama};

// ═══════════════════════════════════════════════════════════════════════════════
// Model: dispatch enum wrapping Llama and Gemma for physics steering
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified model interface for the Niodoo physics engine.
pub enum Model {
    Llama(llama::ModelWeights),
    Gemma(gemma::ModelWeights),
    Gemma4(gemma4::ModelWeights),
}

impl Model {
    pub fn forward(&mut self, tokens: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Model::Llama(m) => m.forward(tokens, index_pos),
            Model::Gemma(m) => m.forward(tokens, index_pos),
            Model::Gemma4(m) => m.forward(tokens, index_pos),
        }
    }

    pub fn forward_with_hidden(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        match self {
            Model::Llama(m) => m.forward_with_hidden(tokens, index_pos),
            Model::Gemma(m) => m.forward_with_hidden(tokens, index_pos),
            Model::Gemma4(m) => m.forward_with_hidden(tokens, index_pos),
        }
    }

    pub fn forward_with_hidden_hooked(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
        hook: Option<&mut dyn LayerHook>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        match self {
            Model::Llama(m) => m.forward_with_hidden_hooked(tokens, index_pos, hook),
            Model::Gemma(m) => m.forward_with_hidden_hooked(tokens, index_pos, hook),
            Model::Gemma4(m) => m.forward_with_hidden_hooked(tokens, index_pos, hook),
        }
    }

    pub fn project_to_logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Model::Llama(m) => m.project_to_logits(hidden),
            Model::Gemma(m) => m.project_to_logits(hidden),
            Model::Gemma4(m) => m.project_to_logits(hidden),
        }
    }

    /// Final norm + lm_head. This is the lens's `unembed`, and it is *not* the same as
    /// `project_to_logits`, which assumes the norm has already been applied.
    pub fn unembed(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Model::Llama(m) => m.unembed(hidden),
            Model::Gemma(m) => m.unembed(hidden),
            Model::Gemma4(m) => m.unembed(hidden),
        }
    }

    pub fn token_embeddings(&self) -> &Tensor {
        match self {
            Model::Llama(m) => m.token_embeddings(),
            Model::Gemma(m) => m.token_embeddings(),
            Model::Gemma4(m) => m.token_embeddings(),
        }
    }

    /// Number of transformer blocks — layer indices for the lens run `0..n_layers`.
    pub fn n_layers(&self) -> usize {
        match self {
            Model::Llama(m) => m.n_layers(),
            Model::Gemma(m) => m.n_layers(),
            Model::Gemma4(m) => m.n_layers(),
        }
    }

    /// Pre-layer embedding scale applied in forward (Gemma: √hidden_dim; Llama: 1).
    /// Same factor as `gemma.rs` / `gemma4.rs` run_layers — raw matrix rows are *not* residual-space.
    pub fn embedding_input_scale(&self) -> f64 {
        match self {
            Model::Gemma(m) => (m.hidden_dim as f64).sqrt(),
            Model::Gemma4(m) => (m.hidden_dim as f64).sqrt(),
            Model::Llama(_) => 1.0,
        }
    }

    pub fn variant_name(&self) -> &'static str {
        match self {
            Model::Llama(_) => "llama3.1",
            Model::Gemma(_) => "gemma3",
            Model::Gemma4(_) => "gemma4",
        }
    }

    pub fn is_gemma(&self) -> bool {
        matches!(self, Model::Gemma(_) | Model::Gemma4(_))
    }

    /// Drop layer KV before a full re-prefill (multi-turn chat / new one-shot).
    pub fn clear_kv_cache(&mut self) {
        match self {
            Model::Llama(m) => m.clear_kv_cache(),
            Model::Gemma(m) => m.clear_kv_cache(),
            Model::Gemma4(m) => m.clear_kv_cache(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Architecture sniffing
// ═══════════════════════════════════════════════════════════════════════════════

/// GGUF architecture string from metadata (lowercase).
pub fn gguf_architecture(ct: &gguf_file::Content) -> String {
    ct.metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_lowercase())
        .unwrap_or_default()
}

/// Heuristic when metadata is missing: path name.
pub fn path_looks_like_gemma(path: &str) -> bool {
    let p = path.to_lowercase();
    p.contains("gemma") && !p.contains("llama")
}

pub fn path_looks_like_gemma4(path: &str) -> bool {
    let p = path.to_lowercase();
    p.contains("gemma-4") || p.contains("gemma_4") || p.contains("gemma4")
}

pub fn find_existing_file(paths: &[&str]) -> Option<String> {
    paths
        .iter()
        .find(|path| Path::new(path).exists())
        .map(|path| (*path).to_string())
}

pub fn tokenizer_next_to_model(model_path: &str) -> Option<String> {
    let tokenizer_path: PathBuf = Path::new(model_path).parent()?.join("tokenizer.json");
    tokenizer_path
        .exists()
        .then(|| tokenizer_path.display().to_string())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Loading
// ═══════════════════════════════════════════════════════════════════════════════

/// A loaded model plus everything the caller learned while loading it.
pub struct Loaded {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub tokenizer_path: String,
    /// GGUF `general.architecture`, lowercased. Empty when the metadata omits it.
    pub arch: String,
    /// Whether the Gemma 4 loader was selected — decides tokenizer fallbacks and SWA checks.
    pub is_gemma4: bool,
    /// Whether the Gemma 3 loader was selected. Mutually exclusive with `is_gemma4`;
    /// both false means the Llama loader.
    pub is_gemma3: bool,
}

/// Load a GGUF model and its tokenizer.
///
/// `cli_tokenizer` overrides the search when it points at a file that exists. `verbose`
/// gates the progress prints so the sidecar can load quietly.
pub fn load_gguf(
    model_path: &str,
    cli_tokenizer: Option<String>,
    device: &Device,
    verbose: bool,
) -> Result<Loaded> {
    let mut file = std::fs::File::open(model_path)?;
    let mut reader = BufReader::new(&mut file);
    let ct = gguf_file::Content::read(&mut reader)?;
    let arch = gguf_architecture(&ct);
    if arch.contains("gemma3n") {
        anyhow::bail!(
            "GGUF architecture is '{arch}' (Gemma 3n E2B/E4B). Needs a dedicated loader \
             (AltUp + Laurel + per-layer emb). File is fine at data/google/google_gemma-3n-E4B-it-Q5_K_M.gguf \
             — not wired yet. Use gemma-3-4b-it-Q4_K_M.gguf or a gemma4 IT GGUF for today."
        );
    }

    let load_gemma4 =
        arch.contains("gemma4") || (arch.is_empty() && path_looks_like_gemma4(model_path));
    let load_gemma3 = !load_gemma4
        && (arch.contains("gemma3")
            || (arch.is_empty() && path_looks_like_gemma(model_path))
            || (arch.contains("gemma") && !arch.contains("gemma4")));

    let describe = || {
        if arch.is_empty() {
            "path-heuristic".to_string()
        } else {
            arch.clone()
        }
    };

    let model = if load_gemma4 {
        if verbose {
            println!(
                "    Architecture: {} → Gemma 4 loader (our Rust path; see gemma4.rs + NOTICE)",
                describe()
            );
        }
        Model::Gemma4(gemma4::ModelWeights::from_gguf(ct, &mut reader, device)?)
    } else if load_gemma3 {
        if verbose {
            println!("    Architecture: {} → Gemma 3 loader", describe());
        }
        let m = gemma::ModelWeights::from_gguf(ct, &mut reader, device)?;
        if verbose {
            println!("    Gemma 3 loaded (hidden_dim={})", m.hidden_dim);
        }
        Model::Gemma(m)
    } else {
        if verbose {
            println!(
                "    Architecture: {} → Llama loader",
                if arch.is_empty() {
                    "default".to_string()
                } else {
                    arch.clone()
                }
            );
        }
        let m = llama::ModelWeights::from_gguf(ct, &mut reader, device)?;
        if verbose {
            println!("    Llama loaded");
        }
        Model::Llama(m)
    };

    // Find tokenizer. Gemma 4 must not silently fall back to the Gemma 3 asset:
    // their special-token vocabularies and chat markers differ.
    let tokenizer_fallbacks: &[&str] = if load_gemma4 {
        &["data/google/gemma4_assets/tokenizer.json"]
    } else {
        &["data/google/tokenizer.json", "data/tokenizer.json"]
    };
    let tokenizer_path = cli_tokenizer
        .filter(|path| Path::new(path).exists())
        .or_else(|| {
            if load_gemma4 {
                // `data/google/tokenizer.json` is Gemma 3 in this worktree even
                // though the Gemma 4 GGUF is adjacent to it.
                find_existing_file(tokenizer_fallbacks)
                    .or_else(|| tokenizer_next_to_model(model_path))
            } else {
                tokenizer_next_to_model(model_path)
                    .or_else(|| find_existing_file(tokenizer_fallbacks))
            }
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Required tokenizer file not found.\n\
                 Pass --tokenizer /path/to/tokenizer.json, put tokenizer.json next to the model,\n\
                 or install the matching asset under data/google/. See SETUP.md."
            )
        })?;
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
    if verbose {
        println!("    Tokenizer loaded ({})", tokenizer_path);
    }

    Ok(Loaded {
        model,
        tokenizer,
        tokenizer_path,
        arch,
        is_gemma4: load_gemma4,
        is_gemma3: load_gemma3,
    })
}
