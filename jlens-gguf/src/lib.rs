//! `jlens-gguf` — the Jacobian lens of [Verbalizable Representations Form a Global
//! Workspace in Language Models][paper], fitted against GGUF weights through hydro's
//! loader.
//!
//! [paper]: https://transformer-circuits.pub/2026/workspace/index.html
//!
//! The reference implementation is `~/jacobian-lens` (Python, package `jlens`). This crate
//! ports it with one substitution: candle has no gradient through quantised matmul, so the
//! reverse-mode estimator is replaced by an exactly equivalent forward-mode one. See
//! `docs/jlens-gguf/DESIGN.md` for the derivation and `PLAN.md` for the verification gates.
//!
//! This is **not** `gguf_hooks::jacobian`, which is a local finite-difference probe
//! at one decode step. Different algorithm, different object, deliberately different name.

pub mod baseline;
pub mod basis;
pub mod decode;
pub mod fit;
pub mod keys;
pub mod lens;
pub mod probe;
pub mod stability;
pub mod structure;
pub mod telemetry;
pub mod tokens;
