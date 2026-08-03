//! Forward-pass surface: per-layer hooks inside the transformer stack.
//!
//! The residual engine acts once per token on the final pre-`lm_head` state; the logit
//! engines act after `lm_head`. This module is the third surface — physics applied
//! *between* layers, while the token is still being computed.
//!
//! All three model forks (`llama.rs`, `gemma.rs`, `gemma4.rs`) run their stack as a plain
//! `for` loop over an owned `Vec<Layer>`, so a hook is a few lines per loop. The hook is
//! threaded as `Option<&mut dyn LayerHook>` rather than parked in a global, so the
//! dependency stays visible in the signatures.
//!
//! ## Two constraints this module is built around
//!
//! **Shape.** `run_layers` sees `(1, S, D)` during prefill and `(1, 1, D)` during decode.
//! Only the last position is perturbed and the delta is zero-padded across the sequence,
//! matching what the residual engine does at the top level.
//!
//! **Geometry.** Gemma scales embeddings by `√D` before the stack and applies
//! `post_attn_norm` / `post_ffn_norm`, so mid-stack activations are *not* in the same space
//! as the pre-`lm_head` residual that `splat_sigma` and `force_cap` were tuned against
//! (see `research_logs/2026-07-11_diderot-field-geometry-divergence.md`). Absolute caps
//! ported from the residual surface would be meaningless here. Instead every hook force is
//! **scale-free**: the delta is renormalized to a fraction of `‖h‖` measured at that site,
//! so one knob behaves the same at every layer and every model size.

use candle_core::{DType, Result, Tensor};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

/// Where in a transformer block a hook fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookSite {
    /// Block input, before attention norm.
    PreLayer,
    /// After the attention residual add.
    PostAttn,
    /// After the MLP residual add (== block output).
    PostMlp,
    /// After the final norm, before `lm_head`.
    FinalNorm,
}

impl HookSite {
    pub fn as_str(self) -> &'static str {
        match self {
            HookSite::PreLayer => "pre_layer",
            HookSite::PostAttn => "post_attn",
            HookSite::PostMlp => "post_mlp",
            HookSite::FinalNorm => "final_norm",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pre_layer" | "pre" => Some(HookSite::PreLayer),
            "post_attn" | "attn" => Some(HookSite::PostAttn),
            "post_mlp" | "mlp" | "layer" => Some(HookSite::PostMlp),
            "final_norm" | "final" => Some(HookSite::FinalNorm),
            _ => None,
        }
    }
}

/// A physics engine acting inside the forward pass.
pub trait LayerHook {
    /// Cheap gate, called before any tensor work. Must not allocate.
    ///
    /// The hot path depends on this: `n_layers × n_sites` calls per token, so a hook that
    /// is off must cost nothing beyond this test.
    fn wants(&self, site: HookSite, layer_idx: usize) -> bool;

    /// Return a replacement activation, or `None` to pass through untouched.
    fn apply(&mut self, site: HookSite, layer_idx: usize, h: &Tensor) -> Result<Option<Tensor>>;

    /// Called once per token before the stack runs, so the hook can reset per-token state.
    fn begin_token(&mut self, _n_layers: usize) {}
}

/// Convenience for the model forks: apply a hook in place if one is installed and wants
/// this site. Keeps each call site to a single line and guarantees the `wants` gate is
/// always checked before any tensor work.
#[inline]
pub fn maybe_apply(
    hook: &mut Option<&mut dyn LayerHook>,
    site: HookSite,
    layer_idx: usize,
    h: Tensor,
) -> Result<Tensor> {
    if let Some(hk) = hook.as_deref_mut() {
        if hk.wants(site, layer_idx) {
            if let Some(replacement) = hk.apply(site, layer_idx, &h)? {
                return Ok(replacement);
            }
        }
    }
    Ok(h)
}

/// Layer band expressed as depth fractions, so one setting scales across 4B/12B/27B
/// rather than hardcoding "layers 16–31" for a 32-layer model.
#[derive(Debug, Clone, Copy)]
pub struct LayerBand {
    pub start_frac: f32,
    pub end_frac: f32,
}

impl LayerBand {
    pub fn new(start_frac: f32, end_frac: f32) -> Self {
        Self {
            start_frac: start_frac.clamp(0.0, 1.0),
            end_frac: end_frac.clamp(0.0, 1.0),
        }
    }

    /// Resolve to an inclusive `[start, end]` layer range for a stack of `n_layers`.
    pub fn resolve(&self, n_layers: usize) -> (usize, usize) {
        if n_layers == 0 {
            return (0, 0);
        }
        let last = n_layers - 1;
        let start = ((self.start_frac * n_layers as f32).floor() as usize).min(last);
        let end = ((self.end_frac * n_layers as f32).ceil() as usize).min(last);
        if start > end {
            (start, start)
        } else {
            (start, end)
        }
    }

    pub fn contains(&self, layer_idx: usize, n_layers: usize) -> bool {
        let (s, e) = self.resolve(n_layers);
        layer_idx >= s && layer_idx <= e
    }
}

/// Mutable operator controls shared by one-shot generation and the chat REPL.
#[derive(Debug, Clone)]
pub struct HookControls {
    pub enabled: bool,
    pub site: HookSite,
    pub band: LayerBand,
    pub norm_fraction: f32,
}

impl HookControls {
    pub fn new(
        enabled: bool,
        site: HookSite,
        start_frac: f32,
        end_frac: f32,
        norm_fraction: f32,
    ) -> Self {
        Self {
            enabled,
            site,
            band: LayerBand::new(start_frac, end_frac),
            norm_fraction: norm_fraction.clamp(0.0, 0.05),
        }
    }

    pub fn params(&self) -> Vec<(&'static str, f32, f32, f32)> {
        vec![
            ("hook.on", if self.enabled { 1.0 } else { 0.0 }, 0.0, 1.0),
            ("hook.fraction", self.norm_fraction, 0.0, 0.01),
            ("hook.start", self.band.start_frac, 0.0, 1.0),
            ("hook.end", self.band.end_frac, 0.0, 1.0),
            ("hook.site", hook_site_code(self.site), 0.0, 3.0),
        ]
    }

    pub fn set_param(&mut self, name: &str, value: f32) -> bool {
        match name {
            "hook.on" => self.enabled = value >= 0.5,
            "hook.fraction" => self.norm_fraction = value.clamp(0.0, 0.01),
            "hook.start" => {
                self.band.start_frac = value.clamp(0.0, self.band.end_frac);
            }
            "hook.end" => {
                self.band.end_frac = value.clamp(self.band.start_frac, 1.0);
            }
            "hook.site" => {
                self.site = hook_site_from_code(value);
            }
            _ => return false,
        }
        true
    }

    pub fn render_sliders(&self) -> String {
        let mut out = String::from("  forward hook\n");
        for (name, value, min, max) in self.params() {
            let span = (max - min).max(1e-8);
            let filled = (((value - min) / span).clamp(0.0, 1.0) * 24.0).round() as usize;
            let bar: String = (0..24)
                .map(|i| if i < filled { '#' } else { '.' })
                .collect();
            out.push_str(&format!(
                "    {:<16} [{}] {:>8.5}   ({} … {})\n",
                name, bar, value, min, max
            ));
        }
        out.push_str(&format!("    site name: {}\n", self.site.as_str()));
        out
    }
}

fn hook_site_code(site: HookSite) -> f32 {
    match site {
        HookSite::PreLayer => 0.0,
        HookSite::PostAttn => 1.0,
        HookSite::PostMlp => 2.0,
        HookSite::FinalNorm => 3.0,
    }
}

fn hook_site_from_code(value: f32) -> HookSite {
    match value.round().clamp(0.0, 3.0) as u8 {
        0 => HookSite::PreLayer,
        1 => HookSite::PostAttn,
        2 => HookSite::PostMlp,
        _ => HookSite::FinalNorm,
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct HookReport {
    pub applications: usize,
    pub delta_mean: f32,
    pub delta_max: f32,
}

#[derive(Serialize)]
struct TraceEntry<'a> {
    step: usize,
    model: &'a str,
    layer: usize,
    site: &'a str,
    activation_norm: f32,
    delta_norm: f32,
    norm_fraction: f32,
}

/// Optional append-only hook trace. Normal operation performs no per-layer host copy;
/// trace mode deliberately synchronizes the two norm scalars for calibration.
pub struct HookTrace {
    file: File,
}

impl HookTrace {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        Ok(Self { file })
    }

    fn write(&mut self, entry: &TraceEntry<'_>) -> Result<()> {
        let line = serde_json::to_string(entry)
            .map_err(|e| candle_core::Error::Msg(format!("hook trace serialize: {e}")))?;
        writeln!(self.file, "{line}")
            .map_err(|e| candle_core::Error::Msg(format!("hook trace write: {e}")))?;
        Ok(())
    }
}

/// Forward-pass adapter driven by the residual steering delta computed once per token.
///
/// Reusing this cached direction is important: calling `NiodooEngine::steer` inside the
/// layer loop would repeat the O(V·D) field search and host telemetry synchronizations.
pub struct NiodooLayerHook<'a> {
    controls: HookControls,
    unit_direction: Tensor,
    direction_live: bool,
    n_layers: usize,
    step: usize,
    model: &'a str,
    trace: Option<&'a mut HookTrace>,
    delta_norms: Vec<Tensor>,
}

impl<'a> NiodooLayerHook<'a> {
    pub fn new(
        controls: &HookControls,
        direction: &Tensor,
        step: usize,
        model: &'a str,
        trace: Option<&'a mut HookTrace>,
    ) -> Result<Self> {
        let direction = match direction.dims() {
            [d] => direction.reshape((1, *d))?,
            [1, _] => direction.clone(),
            dims => {
                return Err(candle_core::Error::Msg(format!(
                    "hook direction must be (D,) or (1,D), got {dims:?}"
                )))
            }
        };
        let controls_live = controls.enabled && controls.norm_fraction > 0.0;
        let direction = direction.to_dtype(DType::F32)?;
        // A disabled hook is allocation-only and performs no device synchronization.
        // Otherwise this is one scalar synchronization per token, never per layer.
        let mag = if controls_live {
            direction.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt()
        } else {
            0.0
        };
        let direction_live = controls_live && mag.is_finite() && mag > 1e-8;
        let unit_direction = if direction_live {
            direction.affine(1.0 / mag as f64, 0.0)?
        } else {
            Tensor::zeros(direction.dims(), DType::F32, direction.device())?
        };
        Ok(Self {
            controls: controls.clone(),
            unit_direction,
            direction_live,
            n_layers: 0,
            step,
            model,
            trace,
            delta_norms: Vec::new(),
        })
    }

    pub fn finish(self) -> Result<HookReport> {
        if self.delta_norms.is_empty() {
            return Ok(HookReport::default());
        }
        let refs: Vec<&Tensor> = self.delta_norms.iter().collect();
        // One device→host transfer after the full stack, for per-token telemetry.
        let values: Vec<f32> = Tensor::stack(&refs, 0)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1()?;
        let applications = values.len();
        let delta_mean = values.iter().sum::<f32>() / applications as f32;
        let delta_max = values.iter().copied().fold(0.0f32, f32::max);
        Ok(HookReport {
            applications,
            delta_mean,
            delta_max,
        })
    }
}

impl LayerHook for NiodooLayerHook<'_> {
    fn wants(&self, site: HookSite, layer_idx: usize) -> bool {
        if !self.controls.enabled
            || !self.direction_live
            || self.controls.norm_fraction <= 0.0
            || site != self.controls.site
            || self.n_layers == 0
        {
            return false;
        }
        let effective_idx = if site == HookSite::FinalNorm {
            self.n_layers - 1
        } else {
            layer_idx
        };
        self.controls.band.contains(effective_idx, self.n_layers)
    }

    fn begin_token(&mut self, n_layers: usize) {
        self.n_layers = n_layers;
        self.delta_norms.clear();
    }

    fn apply(&mut self, site: HookSite, layer_idx: usize, h: &Tensor) -> Result<Option<Tensor>> {
        let (batch, seq, dim) = h.dims3()?;
        if batch != 1 || seq == 0 {
            return Ok(None);
        }
        let dir_d = self.unit_direction.dim(1)?;
        // Hard fail on residual-width mismatch (was silent passthrough — hid 3840/5376 bugs).
        if dim != dir_d {
            return Err(candle_core::Error::Msg(format!(
                "dim mismatch at hooks.apply residual add: h.shape={:?} last={dim} \
                 != direction_d={dir_d} (live residual width)",
                h.dims()
            )));
        }

        let last = h.narrow(1, seq - 1, 1)?.squeeze(1)?;
        let local_norm = last.sqr()?.sum_all()?.sqrt()?;
        let delta = self
            .unit_direction
            .affine(self.controls.norm_fraction as f64, 0.0)?
            .broadcast_mul(&local_norm)?;
        crate::dim_assert::assert_last_dim(&delta, dim, "hooks.apply.delta")?;
        let delta_norm = delta.sqr()?.sum_all()?.sqrt()?;
        self.delta_norms.push(delta_norm.clone());

        if let Some(trace) = self.trace.as_deref_mut() {
            trace.write(&TraceEntry {
                step: self.step,
                model: self.model,
                layer: layer_idx,
                site: site.as_str(),
                activation_norm: local_norm.to_scalar::<f32>()?,
                delta_norm: delta_norm.to_scalar::<f32>()?,
                norm_fraction: self.controls.norm_fraction,
            })?;
        }

        let replacement_last = (&last + &delta)?.unsqueeze(1)?;
        crate::dim_assert::assert_last_dim(&replacement_last, dim, "hooks.apply.residual_add")?;
        let replacement = if seq == 1 {
            replacement_last
        } else {
            let prefix = h.narrow(1, 0, seq - 1)?;
            Tensor::cat(&[&prefix, &replacement_last], 1)?
        };
        Ok(Some(replacement))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    struct CountingHook {
        band: LayerBand,
        n_layers: usize,
        site: HookSite,
        seen: Vec<(HookSite, usize)>,
        applied: usize,
    }

    impl CountingHook {
        fn new(band: LayerBand, n_layers: usize, site: HookSite) -> Self {
            Self {
                band,
                n_layers,
                site,
                seen: Vec::new(),
                applied: 0,
            }
        }
    }

    impl LayerHook for CountingHook {
        fn wants(&self, site: HookSite, layer_idx: usize) -> bool {
            site == self.site && self.band.contains(layer_idx, self.n_layers)
        }
        fn apply(
            &mut self,
            site: HookSite,
            layer_idx: usize,
            h: &Tensor,
        ) -> Result<Option<Tensor>> {
            self.seen.push((site, layer_idx));
            self.applied += 1;
            Ok(Some(h.affine(2.0, 0.0)?))
        }
    }

    fn h(dev: &Device) -> Tensor {
        Tensor::ones((1, 1, 4), DType::F32, dev).unwrap()
    }

    #[test]
    fn band_resolves_by_depth_fraction() {
        // Middle-to-late on a 32-layer stack ≈ the reference's 16..31.
        let b = LayerBand::new(0.5, 1.0);
        assert_eq!(b.resolve(32), (16, 31));
        // Same fractions on a bigger stack scale automatically.
        assert_eq!(b.resolve(62), (31, 61));
    }

    #[test]
    fn band_handles_degenerate_inputs() {
        assert_eq!(LayerBand::new(0.0, 1.0).resolve(1), (0, 0));
        assert_eq!(LayerBand::new(0.0, 0.0).resolve(0), (0, 0));
        // Inverted range collapses instead of panicking.
        let inverted = LayerBand::new(0.9, 0.1);
        let (s, e) = inverted.resolve(10);
        assert!(s <= e);
    }

    #[test]
    fn no_hook_is_a_passthrough() {
        let dev = Device::Cpu;
        let mut none: Option<&mut dyn LayerHook> = None;
        let out = maybe_apply(&mut none, HookSite::PostMlp, 3, h(&dev)).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![1.0; 4]);
    }

    #[test]
    fn hook_outside_its_band_does_not_fire() {
        let dev = Device::Cpu;
        let mut hk = CountingHook::new(LayerBand::new(0.5, 1.0), 32, HookSite::PostMlp);
        {
            let mut opt: Option<&mut dyn LayerHook> = Some(&mut hk);
            // layer 2 is below the band start (16)
            let out = maybe_apply(&mut opt, HookSite::PostMlp, 2, h(&dev)).unwrap();
            let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(v, vec![1.0; 4], "out-of-band layer must pass through");
        }
        assert_eq!(hk.applied, 0);
    }

    #[test]
    fn hook_fires_inside_its_band_and_site() {
        let dev = Device::Cpu;
        let mut hk = CountingHook::new(LayerBand::new(0.5, 1.0), 32, HookSite::PostMlp);
        {
            let mut opt: Option<&mut dyn LayerHook> = Some(&mut hk);
            let out = maybe_apply(&mut opt, HookSite::PostMlp, 20, h(&dev)).unwrap();
            let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(v, vec![2.0; 4]);
            // Right layer, wrong site → no fire.
            let out2 = maybe_apply(&mut opt, HookSite::PreLayer, 20, h(&dev)).unwrap();
            let v2: Vec<f32> = out2.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(v2, vec![1.0; 4]);
        }
        assert_eq!(hk.applied, 1);
        assert_eq!(hk.seen, vec![(HookSite::PostMlp, 20)]);
    }

    #[test]
    fn site_names_round_trip() {
        for s in [
            HookSite::PreLayer,
            HookSite::PostAttn,
            HookSite::PostMlp,
            HookSite::FinalNorm,
        ] {
            assert_eq!(HookSite::parse(s.as_str()), Some(s));
        }
        assert_eq!(HookSite::parse("nonsense"), None);
    }

    #[test]
    fn niodoo_hook_scales_to_local_norm_and_only_changes_last_position() {
        let dev = Device::Cpu;
        let controls = HookControls::new(true, HookSite::PostMlp, 0.0, 1.0, 0.01);
        let direction = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &dev).unwrap();
        let mut hook = NiodooLayerHook::new(&controls, &direction, 3, "test", None).unwrap();
        hook.begin_token(2);

        let input = Tensor::ones((1, 2, 4), DType::F32, &dev).unwrap();
        let out = hook.apply(HookSite::PostMlp, 0, &input).unwrap().unwrap();
        let values: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(&values[..4], &[1.0; 4], "prefill prefix changed");
        // Last-position norm is 2, so a 1% hook adds exactly 0.02 along +x.
        assert!((values[4] - 1.02).abs() < 1e-5);
        assert_eq!(&values[5..], &[1.0; 3]);

        let report = hook.finish().unwrap();
        assert_eq!(report.applications, 1);
        assert!((report.delta_mean - 0.02).abs() < 1e-5);
        assert!((report.delta_max - 0.02).abs() < 1e-5);
    }

    #[test]
    fn niodoo_hook_zero_direction_is_passthrough() {
        let dev = Device::Cpu;
        let controls = HookControls::new(true, HookSite::PostMlp, 0.0, 1.0, 0.01);
        let direction = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        let mut hook = NiodooLayerHook::new(&controls, &direction, 0, "test", None).unwrap();
        hook.begin_token(2);
        assert!(!hook.wants(HookSite::PostMlp, 0));
        assert_eq!(hook.finish().unwrap().applications, 0);
    }

    #[test]
    fn live_controls_clamp_and_route() {
        let mut controls = HookControls::new(true, HookSite::PostMlp, 0.5, 1.0, 0.0005);
        assert!(controls.set_param("hook.fraction", 1.0));
        assert_eq!(controls.norm_fraction, 0.01);
        assert!(controls.set_param("hook.site", 1.0));
        assert_eq!(controls.site, HookSite::PostAttn);
        assert!(!controls.set_param("unknown", 1.0));
    }
}
