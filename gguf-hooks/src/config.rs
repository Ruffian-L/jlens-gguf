//! Configuration Module
//!
//! TOML-deserializable configuration for all physics parameters.
//! Supports loading from file with CLI overrides.
//! Falls back to sensible defaults when no config file exists.

use serde::Deserialize;
use std::path::Path;

/// Top-level configuration for the hydrodynamic swarm.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub physics: PhysicsConfig,
    pub logit_physics: LogitPhysicsConfig,
    pub hooks: HooksConfig,
    pub generation: GenerationConfig,
    pub memory: MemoryConfig,
    pub micro_dream: MicroDreamConfig,
    pub algo: AlgoConfig,
    pub jacobian: JacobianConfig,
    /// Self-reg phase observe / (later) force — see docs/SELF_REG_PHASES.md
    pub self_reg: SelfRegConfig,
}

/// Self-regulation phases: answer → revise → settle.
///
/// `observe` labels only. `force` reserved for residual schedule in revise.
/// Default off = settle clamps only (current multi-turn usability path).
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SelfRegConfig {
    /// off | observe | force
    pub mode: String,
    /// Min answer tokens before revise heuristics can fire.
    pub min_answer_tokens: usize,
    /// Entropy rise vs step-0 to flag revise (nats, top-k approx).
    pub revise_entropy_delta: f32,
    /// Margin below this (after min_answer_tokens) soft-flags revise.
    pub revise_margin_max: f32,
    /// Trailing identical non-empty lines ≥ this → label `revise` (0 = off).
    /// Catches confident low-entropy thrash (`17 × 10 = 170`×N) that entropy/margin miss.
    pub revise_line_repeat: usize,
    /// Trailing identical lines ≥ this → settle clamp stop (0 = off). Default 4.
    pub settle_line_repeat: usize,
    /// Min trimmed line length for line-repeat revise/settle (avoid "ok"/"yes" noise).
    pub line_repeat_min_chars: usize,
    /// Count of "try again" / Wait-loop blocks before settle clamp (0 = off). Default 3.
    /// Catches Spell-cat multi-line revise loops that are not identical-line thrash.
    pub settle_wait_loops: usize,
    // --- mode=force only: residual schedule while phase==revise; answer stays force-off ---
    /// Residual force_cap during revise (light for 12B; 0 = skip force application).
    pub force_cap: f32,
    pub force_goal_scale: f32,
    pub force_splat_scale: f32,
    pub force_field_scale: f32,
}

impl Default for SelfRegConfig {
    fn default() -> Self {
        Self {
            mode: "off".into(),
            min_answer_tokens: 3,
            revise_entropy_delta: 1.2,
            revise_margin_max: 0.08,
            revise_line_repeat: 2,
            settle_line_repeat: 4,
            line_repeat_min_chars: 6,
            settle_wait_loops: 3,
            // Light 12B-ish revise shove (physics_light neighborhood)
            force_cap: 0.6,
            force_goal_scale: 0.08,
            force_splat_scale: 0.05,
            force_field_scale: 0.05,
        }
    }
}

/// Model identity for the √-law scaling readout (`src/algo_scale.rs`).
///
/// Purely informational: it drives the HUD's "predicted" column and nothing
/// else. Physics knobs are never auto-set from it — you paste the generated
/// values yourself after reading them.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AlgoConfig {
    /// Model size in billions. 0 = infer from the weights filename.
    pub params_b: f32,
    /// standard | instruct | chat | thinking | coding. Empty = infer from path.
    pub model_type: String,
}

/// Physics engine parameters.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PhysicsConfig {
    pub dt: f32,
    pub viscosity_scale: f32,
    pub force_cap: f32,
    pub splat_sigma: f32,
    pub splat_alpha: f32,
    pub min_splat_dist: f32,
    pub splat_delta_threshold: f32,
    /// Top-K nearest points for gradient approximation (0 = exact gradient).
    pub gradient_topk: usize,
    /// Steer the hidden state (pre-lm_head) instead of logits.
    pub steer_hidden: bool,
    /// Per-step blend factor pulling steered state back toward baseline (0.0 = off, 0.15 = gentle).
    pub manifold_pullback: f32,
    pub bundle_min_dist: f32,
    pub splat_lambda_default: f32,
    pub pain_decay_factor: f32,
    pub dream_correction_threshold: f32,
    /// Scale applied to splat force before sum (0.05–0.15 = gentle for Gemma).
    pub splat_force_scale: f32,
    /// Soft max on ||F_s|| after scale (0 = disabled).
    pub splat_force_max: f32,
    /// Scale applied to goal attractor (prefill residual).
    pub goal_force_scale: f32,
    /// Soft max on ||F_a|| after scale (0 = disabled).
    pub goal_force_max: f32,
    /// Step index to begin late F_a attenuation (0 = off). Typical B4d: 48.
    pub goal_late_start: usize,
    /// Tokens to ramp from full F_a → goal_late_end.
    pub goal_late_span: usize,
    /// F_a multiplier at end of late attenuation (0–1). e.g. 0.35.
    pub goal_late_end: f32,
    /// Min steps between online splat deposits (anti-spam).
    pub online_splat_interval: usize,
    /// Field wake mode: "off" | "wake" | "blend" | "dist_weighted"
    /// See research_logs/*field-wake* ablation table.
    pub field_wake_mode: String,
    /// k nearest embeddings for wake pull.
    pub field_wake_k: usize,
    /// Strength of nearest-emb pull (before soft cap).
    pub field_wake_scale: f32,
    /// Soft max on ||wake force|| (0 = off).
    pub field_wake_max: f32,
    /// In blend mode: weight of pure ∇ρ vs wake (0=all wake, 1=all grad when alive).
    pub field_grad_blend: f32,
    /// Distance scale for dist_weighted: strength ∝ 1/(1+(d/τ)²).
    pub field_wake_dist_tau: f32,
    /// Force ramp: first N tokens scale total force from `force_ramp_start` → 1.0.
    /// Early-token gentler force (respect prefill residual geometry). 0 = off.
    pub force_ramp_tokens: usize,
    /// Multiplier at step 0 when ramping (e.g. 0.15).
    pub force_ramp_start: f32,
    /// If true, only deposit splats on high-signal steps (δ > thresh OR pain OR strong pleasure).
    /// If false, any non-Skip quality deposit (current default path).
    pub targeted_splat_only: bool,
    /// After prefill, run one micro-dream against goal (respect initial hidden / J-space).
    pub prefill_micro_dream: bool,
    /// On Pain, deposit a stronger ocean "recovery" packet (variant E).
    pub pain_recovery_ocean: bool,
    /// Deposit a scar at the **prefill residual** before save so death→reload
    /// can couple early F_s (fixes LOCALITY COLD: trail scars sit far from next prefill).
    pub prefill_bridge_scar: bool,
    /// Width of the prefill-bridge scar (residual L2). ~80–120 covers small prefill jitter.
    pub prefill_bridge_sigma: f32,
    /// |alpha| scale for bridge scar (sign follows success pleasure / short pain).
    pub prefill_bridge_alpha: f32,
    /// Evaporation λ for bridge (low = lasts across sessions). 0 = anchor (never decays).
    pub prefill_bridge_lambda: f32,
    /// Soft off-center: place bridge at goal + (offset_frac · σ) along a deterministic
    /// direction perpendicular to goal. 0 = on-center (F_s≈0 at start by gradient geometry).
    /// ~0.3–0.4 keeps high potential and non-zero step0 F_s.
    pub prefill_bridge_offset_frac: f32,
    /// Min steps between endocrine fires (path stays ON; this only rates it).
    pub endocrine_cooldown_steps: usize,
    /// High-δ endocrine needs top-k entropy above this (nats). Raise = fewer fires.
    pub endocrine_entropy_min: f32,
    /// Console: log +will/−will deposits every N steps (always log −will if true below).
    pub will_log_every: usize,
    /// If true, always print −will deposits; if false, use will_log_every only.
    pub will_log_neg_always: bool,
}

/// Additive physics at the vocabulary distribution.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogitPhysicsConfig {
    /// z += α · normalize(E û_g). 0 = off.
    pub field_alpha: f32,
    /// Scar-tissue vocab bias strength. 0 = off.
    pub splat_scale: f32,
    /// Max scars contributing in one step.
    pub splat_top_m: usize,
    /// Tokens biased per contributing scar.
    pub splat_top_k: usize,
    pub governor_enabled: bool,
    pub governor_velocity_threshold: f32,
    pub governor_brake: f32,
    pub governor_window: usize,
    pub governor_viscosity_threshold: f32,
    pub governor_viscosity_gain: f32,
    /// Hard ceiling on any single governor bias, in logit units.
    pub governor_max_bias: f32,
    /// Penalty subtracted from the `\` token logit to break the `\` loop collapse. 0 = off.
    pub backslash_penalty: f32,
}

/// Scale-free physics inside the transformer stack.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HooksConfig {
    pub enabled: bool,
    /// pre_layer | post_attn | post_mlp | final_norm
    pub site: String,
    /// Inclusive depth band expressed as fractions of total layer count.
    pub start_frac: f32,
    pub end_frac: f32,
    /// Per-application delta norm as a fraction of the local activation norm.
    pub norm_fraction: f32,
    /// Optional JSONL trace path. Empty disables tracing.
    pub trace_out: String,
}

/// Generation parameters.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub default_prompt: String,
    pub eos_token_ids: Vec<u32>,
    pub rep_penalty: f32,
    /// Top-k sampling; 0 = disabled (full vocab after temperature).
    pub top_k: usize,
    /// Nucleus sampling threshold in (0, 1]; 1.0 = disabled.
    pub top_p: f32,
    pub min_success_tokens: usize,
    pub pleasure_alpha: f32,
    pub pain_alpha: f32,
}

/// Splat memory management.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    pub max_splats: usize,
    pub consolidation_dist: f32,
    /// End-of-run / session wall-clock evaporation fallback (see `decay_step`).
    pub decay_rate: f32,
    pub prune_threshold: f32,
    /// Per-token scar strength multiply during generation (`decay_per_token`).
    /// `1.0` = off. Typical B4b: `0.97`–`0.99`. Controls mid-run F_s climb.
    pub online_decay_rate: f32,
    /// Cap on prefill-bridge scars (LRU by created_at). 0 = unlimited.
    pub max_prefill_bridges: usize,
    /// Max active **pain** trail scars (α < 0, non-bridge). 0 = unlimited.
    /// Soft ceiling — prefer dissipation; budget is a backstop.
    pub max_pain_splats: usize,
    /// Max sum of |α| over active pain scars. 0 = unlimited.
    pub max_pain_mass: f32,
    /// After this many pain deposits in a row, place a pleasure answer near goal.
    /// 0 = off. Heart of anti-snowball: pleasure answers pain.
    pub pleasure_answer_after: usize,
    /// |α| for the pleasure-answer scar (positive).
    pub pleasure_answer_alpha: f32,
    /// Width scale for pleasure-answer (multiplies splat_sigma).
    pub pleasure_answer_sigma_scale: f32,
    /// Scar force aggregation: `"soft"` (legacy sum-all) or `"ranked"` (Top-K picker).
    /// Soft remains the default so ablation baselines stay bit-identical until enabled.
    pub memory_force_mode: String,
    /// Top-K scars allowed to contribute force when `memory_force_mode = "ranked"`.
    pub memory_pick_k: usize,
    /// When true (default with ranked): hard-pick only if residual is unsettled;
    /// settled steps fall back to soft-sum.
    pub memory_pick_selective: bool,
    /// Unsettled if top-k entropy (nats) ≥ this.
    pub memory_pick_entropy_min: f32,
    /// Unsettled if confidence margin (p1−p2, or p_chosen proxy) ≤ this.
    pub memory_pick_margin_max: f32,
    /// Unsettled if ‖goal − pos‖ ≥ this. 0 = residual-L2 gate off.
    pub memory_pick_residual_l2_min: f32,
    /// Weight on quality-history term in pick score.
    pub memory_pick_quality_weight: f32,
    /// Weight on prompt_fp match term in pick score.
    pub memory_pick_fp_weight: f32,
}

/// Jacobian measurement lens configuration.
///
/// Measures how hidden-state dimensions map to output logits via finite-difference
/// perturbation. The "Jacobian lens" is Jason's key — it turns clusters into
/// perm-addresses by revealing which hidden dimensions drive which outputs.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JacobianConfig {
    /// Perturbation magnitude for finite difference. 1e-4 is the sweet spot.
    pub epsilon: f32,
    /// Which hook sites to measure at. "final_norm" is primary.
    pub sites: String,
    /// Only track top-k output tokens by sensitivity.
    pub top_k: usize,
    /// Subsample hidden dimensions (0 = all). >0 = random sample.
    pub max_dims: usize,
    /// How often to measure (every N decode steps). 0 = disabled.
    pub interval: usize,
    /// Optional trace log path.
    pub trace_path: String,
    /// Capture JacobianKey on phase edges (first answer content, revise flip, settle).
    /// Cheap when max_dims is set (recommend 64–128); full D is slow.
    pub phase_edge_keys: bool,
}

impl Default for JacobianConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            sites: "final_norm".into(),
            top_k: 16,
            max_dims: 0,
            interval: 0, // disabled by default
            trace_path: String::new(),
            phase_edge_keys: false,
        }
    }
}

/// Micro-dream consolidation tuning.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MicroDreamConfig {
    pub entropy_threshold: f32,
    pub fixed_interval: usize,
    pub adaptive_interval: usize,
    pub blend_normal: f64,
    pub blend_high_entropy: f64,
    pub topocot_threshold: f32,
}

impl Default for AlgoConfig {
    fn default() -> Self {
        Self {
            params_b: 0.0,
            model_type: String::new(),
        }
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            dt: 0.035,
            // Slightly more field influence now that sigma isn't dead
            viscosity_scale: 0.25,
            force_cap: 5.0,
            // Residual-space friendly; was 35 and treated every splat as global
            splat_sigma: 12.0,
            splat_alpha: 1.2,
            min_splat_dist: 25.0,
            // Was 12 — steering delta is routinely 100+, so that deposited every step
            splat_delta_threshold: 90.0,
            gradient_topk: 1024,
            steer_hidden: true,
            manifold_pullback: 0.20,
            bundle_min_dist: 0.05,
            splat_lambda_default: 0.02,
            pain_decay_factor: 0.7,
            dream_correction_threshold: 6.0,
            // Wake memory a bit; 0.08 left F_s≈0 mid-run
            splat_force_scale: 0.25,
            splat_force_max: 60.0,
            // Stop prefill goal from monopolizing (~450 uncapped)
            goal_force_scale: 0.15,
            goal_force_max: 60.0,
            goal_late_start: 0,
            goal_late_span: 30,
            goal_late_end: 0.4,
            online_splat_interval: 6,
            // Phase 1 default: nearest-emb wake (k=1)
            field_wake_mode: "wake".into(),
            field_wake_k: 1,
            field_wake_scale: 0.20,
            field_wake_max: 40.0,
            field_grad_blend: 0.15,
            field_wake_dist_tau: 50.0,
            force_ramp_tokens: 12,
            force_ramp_start: 0.20,
            targeted_splat_only: true,
            prefill_micro_dream: false,
            pain_recovery_ocean: false,
            prefill_bridge_scar: true,
            prefill_bridge_sigma: 90.0,
            prefill_bridge_alpha: 0.75,
            prefill_bridge_lambda: 0.005,
            prefill_bridge_offset_frac: 0.35,
            // Full stack stays ON — these only rate endocrine / console (see IMMUTABLE_RUN_CONTRACT).
            endocrine_cooldown_steps: 28,
            endocrine_entropy_min: 3.0,
            will_log_every: 40,
            will_log_neg_always: true,
        }
    }
}

impl Default for LogitPhysicsConfig {
    fn default() -> Self {
        Self {
            field_alpha: 0.15,
            splat_scale: 0.02,
            splat_top_m: 3,
            splat_top_k: 24,
            governor_enabled: true,
            governor_velocity_threshold: 0.95,
            governor_brake: 3.0,
            governor_window: 6,
            governor_viscosity_threshold: 0.92,
            governor_viscosity_gain: 6.0,
            governor_max_bias: 1.5,
            backslash_penalty: 3.0,
        }
    }
}

impl Default for HooksConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            site: "post_mlp".into(),
            start_frac: 0.5,
            end_frac: 1.0,
            norm_fraction: 0.0005,
            trace_out: String::new(),
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 500,
            temperature: 0.9,
            default_prompt: "Explain the Physics of Friendship in one paragraph.".to_string(),
            eos_token_ids: vec![128009, 128001],
            rep_penalty: 1.25,
            top_k: 0,
            top_p: 1.0,
            min_success_tokens: 15,
            pleasure_alpha: 1.2,
            pain_alpha: -0.6,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_splats: 500,
            consolidation_dist: 80.0,
            decay_rate: 0.98,
            prune_threshold: 0.01,
            online_decay_rate: 1.0, // off unless config sets < 1
            max_prefill_bridges: 24,
            max_pain_splats: 12,
            max_pain_mass: 5.0,
            pleasure_answer_after: 2,
            pleasure_answer_alpha: 0.55,
            pleasure_answer_sigma_scale: 1.2,
            memory_force_mode: "soft".into(),
            memory_pick_k: 8,
            memory_pick_selective: true,
            memory_pick_entropy_min: 2.5,
            memory_pick_margin_max: 0.15,
            memory_pick_residual_l2_min: 0.0,
            memory_pick_quality_weight: 1.0,
            memory_pick_fp_weight: 1.0,
        }
    }
}

impl Default for MicroDreamConfig {
    fn default() -> Self {
        Self {
            entropy_threshold: 3.0,
            fixed_interval: 25,
            adaptive_interval: 8,
            blend_normal: 0.10,
            blend_high_entropy: 0.15,
            topocot_threshold: 18.0,
        }
    }
}

impl Config {
    /// Load from a TOML file. Returns defaults if file doesn't exist.
    /// Validates all numeric invariants after deserialization.
    pub fn load(path: &Path) -> Result<Self, String> {
        let config: Self = if !path.exists() {
            Self::default()
        } else {
            match std::fs::read_to_string(path) {
                Ok(contents) => match toml::from_str(&contents) {
                    Ok(c) => {
                        println!("    Config loaded from: {}", path.display());
                        c
                    }
                    Err(e) => {
                        return Err(format!("Failed to parse config {}: {}", path.display(), e));
                    }
                },
                Err(e) => {
                    return Err(format!("Failed to read config {}: {}", path.display(), e));
                }
            }
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate all numeric invariants. Returns Err with the invalid field name.
    fn validate(&self) -> Result<(), String> {
        let p = &self.physics;
        if p.dt <= 0.0 {
            return Err("physics.dt must be > 0".into());
        }
        if p.viscosity_scale < 0.0 {
            return Err("physics.viscosity_scale must be >= 0".into());
        }
        if p.force_cap < 0.0 {
            return Err("physics.force_cap must be >= 0".into());
        }
        if p.splat_sigma <= 0.0 {
            return Err("physics.splat_sigma must be > 0".into());
        }
        if p.splat_alpha < 0.0 {
            return Err("physics.splat_alpha must be >= 0".into());
        }
        if p.min_splat_dist < 0.0 {
            return Err("physics.min_splat_dist must be >= 0".into());
        }
        if p.splat_delta_threshold < 0.0 {
            return Err("physics.splat_delta_threshold must be >= 0".into());
        }
        if p.bundle_min_dist <= 0.0 {
            return Err("physics.bundle_min_dist must be > 0".into());
        }
        if p.splat_lambda_default < 0.0 {
            return Err("physics.splat_lambda_default must be >= 0".into());
        }
        if p.pain_decay_factor <= 0.0 || p.pain_decay_factor > 1.0 {
            return Err("physics.pain_decay_factor must be in (0,1]".into());
        }
        if p.dream_correction_threshold < 0.0 {
            return Err("physics.dream_correction_threshold must be >= 0".into());
        }
        if p.goal_late_end < 0.0 || p.goal_late_end > 1.0 {
            return Err("physics.goal_late_end must be in [0, 1]".into());
        }
        if p.goal_late_start > 0 && p.goal_late_span == 0 {
            return Err("physics.goal_late_span must be > 0 when goal_late_start is set".into());
        }

        let lp = &self.logit_physics;
        if !lp.field_alpha.is_finite() || lp.field_alpha < 0.0 {
            return Err("logit_physics.field_alpha must be finite and >= 0".into());
        }
        if !lp.splat_scale.is_finite() || lp.splat_scale < 0.0 {
            return Err("logit_physics.splat_scale must be finite and >= 0".into());
        }
        if lp.splat_top_m == 0 || lp.splat_top_k == 0 {
            return Err("logit_physics splat_top_m and splat_top_k must be > 0".into());
        }
        if !(0.0..=1.0).contains(&lp.governor_velocity_threshold) {
            return Err("logit_physics.governor_velocity_threshold must be in [0, 1]".into());
        }
        if !(0.0..=1.0).contains(&lp.governor_viscosity_threshold) {
            return Err("logit_physics.governor_viscosity_threshold must be in [0, 1]".into());
        }
        if lp.governor_window == 0 {
            return Err("logit_physics.governor_window must be > 0".into());
        }
        if !lp.governor_brake.is_finite()
            || lp.governor_brake < 0.0
            || !lp.governor_viscosity_gain.is_finite()
            || lp.governor_viscosity_gain < 0.0
            || !lp.governor_max_bias.is_finite()
            || lp.governor_max_bias < 0.0
        {
            return Err("logit_physics governor gains must be finite and >= 0".into());
        }

        let h = &self.hooks;
        if !matches!(
            h.site.trim().to_ascii_lowercase().as_str(),
            "pre_layer"
                | "pre"
                | "post_attn"
                | "attn"
                | "post_mlp"
                | "mlp"
                | "layer"
                | "final_norm"
                | "final"
        ) {
            return Err("hooks.site must be pre_layer, post_attn, post_mlp, or final_norm".into());
        }
        if !h.start_frac.is_finite()
            || !h.end_frac.is_finite()
            || !(0.0..=1.0).contains(&h.start_frac)
            || !(0.0..=1.0).contains(&h.end_frac)
            || h.start_frac > h.end_frac
        {
            return Err("hooks start_frac/end_frac must satisfy 0 <= start <= end <= 1".into());
        }
        if !h.norm_fraction.is_finite() || !(0.0..=0.05).contains(&h.norm_fraction) {
            return Err("hooks.norm_fraction must be finite and in [0, 0.05]".into());
        }

        let g = &self.generation;
        if g.max_tokens == 0 {
            return Err("generation.max_tokens must be > 0".into());
        }
        // T≈0 is greedy argmax in main.rs (no divide-by-zero). Negative is invalid.
        if g.temperature < 0.0 {
            return Err("generation.temperature must be >= 0".into());
        }
        if g.rep_penalty < 1.0 {
            return Err("generation.rep_penalty must be >= 1.0".into());
        }
        if g.top_p <= 0.0 || g.top_p > 1.0 {
            return Err("generation.top_p must be in (0, 1]".into());
        }

        let m = &self.memory;
        if m.max_splats == 0 {
            return Err("memory.max_splats must be > 0".into());
        }
        if m.consolidation_dist < 0.0 {
            return Err("memory.consolidation_dist must be >= 0".into());
        }
        if m.decay_rate < 0.0 {
            return Err("memory.decay_rate must be >= 0".into());
        }
        if m.online_decay_rate <= 0.0 || m.online_decay_rate > 1.0 {
            return Err("memory.online_decay_rate must be in (0, 1]".into());
        }
        if m.prune_threshold < 0.0 {
            return Err("memory.prune_threshold must be >= 0".into());
        }
        let mode = m.memory_force_mode.trim().to_ascii_lowercase();
        if mode != "soft"
            && mode != "ranked"
            && mode != "pick"
            && mode != "topk"
            && mode != "top-k"
            && mode != "top_k"
        {
            return Err("memory.memory_force_mode must be \"soft\" or \"ranked\"".into());
        }
        if m.memory_pick_k == 0 {
            return Err("memory.memory_pick_k must be > 0".into());
        }
        if m.memory_pick_quality_weight < 0.0 {
            return Err("memory.memory_pick_quality_weight must be >= 0".into());
        }
        if m.memory_pick_fp_weight < 0.0 {
            return Err("memory.memory_pick_fp_weight must be >= 0".into());
        }
        if m.memory_pick_residual_l2_min < 0.0 {
            return Err("memory.memory_pick_residual_l2_min must be >= 0".into());
        }

        let d = &self.micro_dream;
        if d.entropy_threshold < 0.0 {
            return Err("micro_dream.entropy_threshold must be >= 0".into());
        }
        if d.fixed_interval == 0 {
            return Err("micro_dream.fixed_interval must be > 0".into());
        }
        if d.adaptive_interval == 0 {
            return Err("micro_dream.adaptive_interval must be > 0".into());
        }
        if d.blend_normal < 0.0 {
            return Err("micro_dream.blend_normal must be >= 0".into());
        }
        if d.blend_high_entropy < 0.0 {
            return Err("micro_dream.blend_high_entropy must be >= 0".into());
        }
        if d.topocot_threshold < 0.0 {
            return Err("micro_dream.topocot_threshold must be >= 0".into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let cfg = Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn toml_parsing_works() {
        let toml_str = r#"
[physics]
dt = 0.1
force_cap = 50.0

[generation]
temperature = 0.7
max_tokens = 200
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!((cfg.physics.dt - 0.1).abs() < 1e-6);
        assert!((cfg.physics.force_cap - 50.0).abs() < 1e-6);
        assert!((cfg.generation.temperature - 0.7).abs() < 1e-6);
        assert_eq!(cfg.generation.max_tokens, 200);
        // Non-specified fields get defaults
        assert!((cfg.physics.viscosity_scale - 0.25).abs() < 1e-6);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn unknown_sections_and_keys_are_rejected() {
        assert!(toml::from_str::<Config>("[model]\npath = \"ignored.gguf\"\n").is_err());
        assert!(toml::from_str::<Config>("[physics]\nfield_logit_alpha = 0.15\n").is_err());
    }

    #[test]
    fn force_off_disables_all_three_surfaces() {
        let cfg: Config = toml::from_str(
            r#"
[physics]
force_cap = 0.0
splat_force_scale = 0.0
goal_force_scale = 0.0
field_wake_scale = 0.0
force_ramp_tokens = 0
prefill_bridge_scar = false

[logit_physics]
field_alpha = 0.0
splat_scale = 0.0
governor_enabled = false

[hooks]
enabled = false
"#,
        )
        .unwrap();
        assert_eq!(cfg.physics.force_cap, 0.0);
        assert_eq!(cfg.logit_physics.field_alpha, 0.0);
        assert_eq!(cfg.logit_physics.splat_scale, 0.0);
        assert!(!cfg.logit_physics.governor_enabled);
        assert!(!cfg.hooks.enabled);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn supported_surface_configs_parse_and_validate() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let supported = [
            "config.example.toml",
            "config.force_off.toml",
            "configs/profiles/config.force_off.toml",
            "configs/profiles/config.ramp_off.toml",
            "configs/profiles/config.27b.toml",
            "configs/profiles/config.memory_ranked.toml",
            "configs/ablation/config_force_off.toml",
            "configs/ablation/config_sweep_decay.toml",
            "configs/ablation/config_sweep_lowdecay.toml",
            "configs/gemma4/config.gemma4_T09.toml",
            "configs/gemma4/config.gemma4_greedy.toml",
            "configs/gemma4/config.gemma4_near_vanilla.toml",
            "configs/gemma4/config.gemma4_stable.toml",
            "configs/gates/config.residual_only.toml",
            "configs/gates/config.logit_chain.toml",
            "configs/gates/config.hooks.toml",
            "configs/gates/config.three_surface.toml",
        ];
        for rel in supported {
            Config::load(&root.join(rel))
                .unwrap_or_else(|e| panic!("{rel} must remain a valid live-schema config: {e}"));
        }
    }

    #[test]
    fn validation_catches_negative_dt() {
        let mut cfg = Config::default();
        cfg.physics.dt = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validation_catches_zero_max_tokens() {
        let mut cfg = Config::default();
        cfg.generation.max_tokens = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn eos_token_ids_default() {
        let cfg = Config::default();
        assert!(cfg.generation.eos_token_ids.contains(&128009));
        assert!(cfg.generation.eos_token_ids.contains(&128001));
    }
}
