//! Structured residual-width checks + startup inventory.
//!
//! **Contract:** `expected_d` is always the live model residual width from GGUF
//! (`gemma4.embedding_length` / field.dim / model.hidden_dim). Never hardcode
//! 3840 / 4096 / 5376 — 12B and 31B differ; the header is the card.
//!
//! On failure we always print:
//! ```text
//! [RESIDUAL MISMATCH] expected <D> got <last> at <site> shape=<…>
//! ```
//! then return Err (callers may panic). Unit tests use residual_dim=0 to skip.

use candle_core::{Result, Tensor};

/// Last axis of `t` must equal live residual width `expected_d`.
#[inline]
pub fn assert_last_dim(t: &Tensor, expected_d: usize, site: &'static str) -> Result<()> {
    if expected_d == 0 {
        return Ok(()); // unit-test / unset guard
    }
    let dims = t.dims();
    let last = match dims.last() {
        Some(&d) => d,
        None => {
            log_mismatch(expected_d, 0, site, &dims, "empty shape");
            return Err(candle_core::Error::Msg(format!(
                "[RESIDUAL MISMATCH] expected {expected_d} got empty at {site}"
            )));
        }
    };
    if last != expected_d {
        log_mismatch(expected_d, last, site, &dims, "last-axis");
        return Err(candle_core::Error::Msg(format!(
            "[RESIDUAL MISMATCH] expected {expected_d} got {last} at {site} shape={dims:?}"
        )));
    }
    Ok(())
}

/// Hard panic variant — use on memory write paths where Result is awkward.
#[inline]
pub fn require_last_dim(t: &Tensor, expected_d: usize, site: &'static str) {
    if let Err(e) = assert_last_dim(t, expected_d, site) {
        panic!("{e}");
    }
}

/// Assert a raw last-axis length (safetensors / TCT row width before Tensor wrap).
#[inline]
pub fn assert_width(got: usize, expected_d: usize, site: &'static str) -> Result<()> {
    if expected_d == 0 {
        return Ok(());
    }
    if got != expected_d {
        eprintln!(
            "[RESIDUAL MISMATCH] expected {expected_d} got {got} at {site} (raw width)"
        );
        return Err(candle_core::Error::Msg(format!(
            "[RESIDUAL MISMATCH] expected {expected_d} got {got} at {site}"
        )));
    }
    Ok(())
}

#[inline]
fn log_mismatch(expected: usize, got: usize, site: &str, dims: &[usize], kind: &str) {
    eprintln!(
        "[RESIDUAL MISMATCH] expected {expected} got {got} at {site} shape={dims:?} ({kind}) \
         — live GGUF residual width; do not hardcode 3840/4096/5376"
    );
}

/// Last dim helper (0 if empty).
#[inline]
pub fn last_dim(t: &Tensor) -> usize {
    t.dims().last().copied().unwrap_or(0)
}

/// One-shot startup inventory so we stop guessing what is live.
pub fn log_startup_inventory(
    residual_d: usize,
    variant: &str,
    model_path: &str,
    physics: &crate::config::PhysicsConfig,
    logit: &crate::config::LogitPhysicsConfig,
    hooks: &crate::config::HooksConfig,
    jacobian: &crate::config::JacobianConfig,
) {
    eprintln!("[RESIDUAL CONFIG] ========================================");
    eprintln!("[RESIDUAL CONFIG] variant={variant}");
    eprintln!("[RESIDUAL CONFIG] model={model_path}");
    eprintln!(
        "[RESIDUAL CONFIG] hidden_size / residual_d = {residual_d}  (from GGUF field — source of truth)"
    );
    eprintln!(
        "[RESIDUAL CONFIG] known cards: Gemma4-12B=3840  Gemma4-31B=5376  Llama3.1-8B=4096  Gemma3-4B=2560"
    );
    eprintln!(
        "[RESIDUAL CONFIG] residual.force_cap={}  splat_force_scale={}  goal_force_scale={}  field_wake_scale={}",
        physics.force_cap, physics.splat_force_scale, physics.goal_force_scale, physics.field_wake_scale
    );
    eprintln!(
        "[RESIDUAL CONFIG] residual.enabled_path={}  (cap>0 && steer_hidden)",
        physics.force_cap > 1e-8 && physics.steer_hidden
    );
    eprintln!(
        "[RESIDUAL CONFIG] logit.field_alpha={}  splat_scale={}  governor={}",
        logit.field_alpha, logit.splat_scale, logit.governor_enabled
    );
    eprintln!(
        "[RESIDUAL CONFIG] hooks.enabled={}  site={}  norm_fraction={}",
        hooks.enabled, hooks.site, hooks.norm_fraction
    );
    eprintln!(
        "[RESIDUAL CONFIG] jacobian.interval={}  (0=off/read-only when measuring) epsilon={} max_dims={}",
        jacobian.interval, jacobian.epsilon, jacobian.max_dims
    );
    // No learned Linear out_features into residual in this crate — forces add in D.
    eprintln!(
        "[RESIDUAL CONFIG] custom_proj.out_features = none (niodoo/splat/jacobian add in residual_d; no Linear inject)"
    );
    eprintln!(
        "[RESIDUAL CONFIG] leftover hardcodes to watch: splat.rs used to default current_dim=4096 (now from mu); \
         main comment may mention 3840 for 12B memory"
    );
    eprintln!("[RESIDUAL CONFIG] ========================================");
}
