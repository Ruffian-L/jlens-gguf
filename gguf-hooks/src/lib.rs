//! Load a GGUF model with candle and hook every residual-stream site in its forward pass.
//!
//! Deliberately thin: load a GGUF model, run it with hooks installed, read out logits.
//! Nothing else. Extracted from the `hydrodynamic-swarm` research codebase, where the same
//! modules drive a physics-steered generation loop.
//!
//! The one capability that matters: [`hooks::LayerHook`] can both **read** and **replace**
//! an activation at `PreLayer` / `PostAttn` / `PostMlp` / `FinalNorm` for any layer. That is
//! enough to capture residuals, inject perturbations, or steer — without forking the model
//! files.
//!
//! The module graph is acyclic and shallow — `config` depends on nothing, `dim_assert` on
//! `config`, `hooks` on `dim_assert`, the three model forks and `jacobian` on `hooks`, and
//! `loader` on the model forks. Keep it that way: the point of the split is that a sidecar
//! can load hydro's models without dragging in the swarm.
//!
//! ## Licenses & attributions
//!
//! - Our code: MIT-0 (LICENSE)
//! - Candle loader code (`llama.rs`): Apache-2.0 OR MIT — NOT the same as model weights
//! - Model weights carry their own terms; see NOTICE in the repo root.

// The binary uses one subset of these APIs and the sidecar another, so items that look
// dead from inside the library are live across the workspace.
#![allow(dead_code)]

pub mod config;
pub mod dim_assert;
pub mod gemma;
pub mod gemma4;
pub mod hooks;
pub mod jacobian;
pub mod llama;
pub mod loader;
