//! Forward-pass hooks: capture residuals, and inject probe directions.
//!
//! These are the whole mechanism by which the lens gets at the model. Hydro's
//! `LayerHook` can both read an activation and replace it, so a fit needs no changes to
//! `llama.rs` / `gemma.rs` / `gemma4.rs` beyond `unembed`.
//!
//! See `docs/jlens-gguf/DESIGN.md` §3 for why perturbing *every* band position at once
//! is the estimator rather than an approximation of it.

use candle_core::{Result, Tensor};
use gguf_hooks::hooks::{HookSite, LayerHook};

/// A point in the stack the lens reads from or writes to.
///
/// Note the layer convention: block sites (`PreLayer`, `PostAttn`, `PostMlp`) are indexed
/// `0..n_layers`, but `FinalNorm` is reported by the model forks with `layer_idx ==
/// n_layers`. [`Site::final_norm`] builds that correctly so callers don't have to remember.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Site {
    pub site: HookSite,
    pub layer: usize,
}

impl Site {
    /// Output of transformer block `layer` — the residual stream, and the lens's default.
    pub fn block_out(layer: usize) -> Self {
        Self {
            site: HookSite::PostMlp,
            layer,
        }
    }

    /// Post-final-norm, pre-`lm_head`. `n_layers` is the model's block count.
    pub fn final_norm(n_layers: usize) -> Self {
        Self {
            site: HookSite::FinalNorm,
            layer: n_layers,
        }
    }

    fn matches(&self, site: HookSite, layer_idx: usize) -> bool {
        site == self.site && layer_idx == self.layer
    }
}

/// Records activations at a set of sites and leaves the forward pass untouched.
///
/// Multi-site on purpose: the baseline pass needs the source residual at *every* layer
/// being fitted, and one pass that captures all of them costs one prefill instead of one
/// per layer. It supplies the residuals the probe basis is built from and the scale that
/// relative ε is measured against.
pub struct CaptureHook {
    sites: Vec<(Site, Option<Tensor>)>,
}

impl CaptureHook {
    pub fn new(sites: impl IntoIterator<Item = Site>) -> Self {
        Self {
            sites: sites.into_iter().map(|s| (s, None)).collect(),
        }
    }

    /// The activation captured at `at`, `[batch, seq, d]`. `None` means the site never
    /// fired — the site was misaddressed, not that the model had nothing there.
    pub fn take(&mut self, at: Site) -> Option<Tensor> {
        self.sites
            .iter_mut()
            .find(|(site, _)| *site == at)
            .and_then(|(_, captured)| captured.take())
    }
}

impl LayerHook for CaptureHook {
    fn wants(&self, site: HookSite, layer_idx: usize) -> bool {
        self.sites
            .iter()
            .any(|(want, _)| want.matches(site, layer_idx))
    }

    fn apply(&mut self, site: HookSite, layer_idx: usize, h: &Tensor) -> Result<Option<Tensor>> {
        for (want, slot) in self.sites.iter_mut() {
            if want.matches(site, layer_idx) {
                *slot = Some(h.clone());
            }
        }
        Ok(None)
    }

    fn begin_token(&mut self, _n_layers: usize) {
        for (_, slot) in self.sites.iter_mut() {
            *slot = None;
        }
    }
}

/// Injects one probe direction per batch element at the source site, and captures the
/// target site.
///
/// Batch element `b` carries direction `b` — the same batch-axis trick `jlens` uses to
/// compute several Jacobian rows per backward pass, run forwards instead. The prompt must
/// therefore be replicated along the batch axis before the forward call.
pub struct ProbeHook<'a> {
    source: Site,
    target: Site,
    /// `[batch, d]` — probe directions, already scaled by ±ε.
    deltas: &'a Tensor,
    /// `[1, seq, 1]` — 1.0 at the source band's positions, 0 elsewhere.
    band_mask: &'a Tensor,
    captured: Option<Tensor>,
}

impl<'a> ProbeHook<'a> {
    pub fn new(source: Site, target: Site, deltas: &'a Tensor, band_mask: &'a Tensor) -> Self {
        Self {
            source,
            target,
            deltas,
            band_mask,
            captured: None,
        }
    }

    /// The captured target activation, `[batch, seq, d]`.
    pub fn take(&mut self) -> Option<Tensor> {
        self.captured.take()
    }

    /// `h[b, p, :] += deltas[b, :]` for every `p` in the band.
    fn inject(&self, h: &Tensor) -> Result<Tensor> {
        let deltas = self.deltas.to_dtype(h.dtype())?;
        let mask = self.band_mask.to_dtype(h.dtype())?;
        // [batch, 1, d] × [1, seq, 1] -> [batch, seq, d]
        let spread = deltas.unsqueeze(1)?.broadcast_mul(&mask)?;
        h + spread
    }
}

impl LayerHook for ProbeHook<'_> {
    fn wants(&self, site: HookSite, layer_idx: usize) -> bool {
        self.source.matches(site, layer_idx) || self.target.matches(site, layer_idx)
    }

    fn apply(&mut self, site: HookSite, layer_idx: usize, h: &Tensor) -> Result<Option<Tensor>> {
        let is_source = self.source.matches(site, layer_idx);
        let is_target = self.target.matches(site, layer_idx);

        if is_source {
            let perturbed = self.inject(h)?;
            // When source == target the capture must see the *injected* value; that is what
            // makes the identity gate (fit with source == target) return J ≈ I rather than 0.
            if is_target {
                self.captured = Some(perturbed.clone());
            }
            return Ok(Some(perturbed));
        }

        if is_target {
            self.captured = Some(h.clone());
        }
        Ok(None)
    }

    fn begin_token(&mut self, _n_layers: usize) {
        self.captured = None;
    }
}

/// `[1, seq, 1]` mask that is 1.0 at `positions` and 0 elsewhere.
pub fn band_mask(seq_len: usize, positions: &[usize], device: &candle_core::Device) -> Result<Tensor> {
    let mut mask = vec![0f32; seq_len];
    for &p in positions {
        if p < seq_len {
            mask[p] = 1.0;
        }
    }
    Tensor::from_vec(mask, (1, seq_len, 1), device)
}
