//! Quantized Gemma 4 GGUF loader — hydrodynamic-swarm.
//!
//! **Hard path (Jason, 2026-07-28):** Derive 3→4 from **our** `gemma.rs`, real
//! GGUF tensors/metadata, local HF assets under `data/google/gemma4_assets/`, and
//! garbage smokes until English. Do **not** re-open foreign C++ maps for this
//! port. Llama + Gemma 3 loaders were earned the same way (trial-and-error).
//!
//! Hydro contract (same as gemma.rs): `forward_with_hidden`, `project_to_logits`,
//! `token_embeddings` for residual/splat steering.
//!
//! Gemma 4 vs our Gemma 3 (what this file must get right — from GGUF + assets):
//! - Metadata namespace `gemma4.*` (not `gemma3.*`)
//! - Dual head dims: SWA `head_dim` vs full/global `global_head_dim`
//! - Sliding-window layers + full-attn layers (pattern + window size in GGUF)
//! - Full-attn RoPE: **proportional + partial** (`partial_rotary_factor` in HF
//!   config; GGUF may ship `rope_freqs.weight` and/or `rope.dimension_count`)
//! - SWA RoPE: default geometric base (`rope.freq_base_swa`)
//! - Optional missing `attn_v` → V ← K (`attention_k_eq_v`)
//! - FFN: GELU tanh form (not SiLU)
//! - Attn score scale often 1.0 (env override `GEMMA4_ATTN_SCALE=rsqrt`)
//! - Final logit softcap from GGUF / HF

use crate::hooks::{maybe_apply, HookSite, LayerHook};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;

pub const MAX_SEQ_LEN: usize = 8192;

// ── QMatMul ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qt: QTensor) -> Result<Self> {
        Ok(Self {
            inner: candle_core::quantized::QMatMul::from_qtensor(qt)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

// ── RoPE ──────────────────────────────────────────────────────────────────────
//
// Learned from garbage smokes + local HF `gemma4_assets/config.json` + GGUF keys:
// - SWA layers: standard RoPE over full head_dim, base ≈ 10_000.
// - Full layers: proportional RoPE — inv_freq exponents use **head_dim** as
//   denominator (not rotary_dim), then **partial** rotate: only the first
//   `n_rot` dims get non-trivial angles; the rest stay identity (cos=1, sin=0).
// - cos/sin last-dim always `head_dim/2` so candle rope applies to full head
//   with identity on the non-rotated pairs.
// - Style: default NeoX half-split (`rope`); Gemma3 path used interleaved
//   `rope_i`. Override with GEMMA4_ROPE_STYLE=i if a quant needs it.

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // Standard: all pairs rotated; exponent denom = head_dim.
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    precompute_freqs_from_inv_freq(&theta, device)
}

/// Proportional + optional partial RoPE (full-attn Gemma 4).
/// `n_rot` = dims that rotate (must be even, ≤ head_dim). Remaining pairs → 0 inv_freq.
fn precompute_proportional_partial(
    head_dim: usize,
    n_rot: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let n_rot = n_rot.min(head_dim) & !1; // even
    let rope_angles = n_rot / 2;
    let pairs = head_dim / 2;
    let mut inv = vec![0f32; pairs];
    for i in 0..rope_angles {
        // HF proportional: base ** (2i / head_dim), NOT / n_rot
        inv[i] = 1f32 / freq_base.powf((2 * i) as f32 / head_dim as f32);
    }
    precompute_freqs_from_inv_freq(&inv, device)
}

/// Pad or truncate inv_freq to `head_dim/2`. Extra zeros ⇒ identity rotation.
fn pad_inv_freq(inv_freq: &[f32], head_dim: usize) -> Vec<f32> {
    let pairs = head_dim / 2;
    let mut out = vec![0f32; pairs];
    let n = inv_freq.len().min(pairs);
    out[..n].copy_from_slice(&inv_freq[..n]);
    out
}

fn precompute_freqs_from_inv_freq(inv_freq: &[f32], device: &Device) -> Result<(Tensor, Tensor)> {
    let theta = Tensor::new(inv_freq, device)?;
    let idx = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx.cos()?, idx.sin()?))
}

/// Approximate GELU (tanh form) — used by Gemma 4 dense FFN in open references.
fn gelu_tanh(xs: &Tensor) -> Result<Tensor> {
    // 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))
    let x3 = xs.powf(3.0)?;
    let inner = (xs + (x3 * 0.044715)?)?;
    let t = (inner * (2.0f64 / std::f64::consts::PI).sqrt())?.tanh()?;
    (xs * 0.5)? * (1.0 + t)?
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, seq, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, seq, d))?
        .reshape((b, n_kv * n_rep, seq, d))
}

/// Causal (+ optional SWA) mask for prefill.
/// Shape is `[q_len, kv_len]` — must match attention scores after KV window trim.
/// When SWA keeps only the last `kv_len` keys, `kv_start` is the absolute index of
/// key column 0 in the original sequence (usually `q_len - kv_len` when trimmed).
fn make_mask(
    q_len: usize,
    kv_len: usize,
    kv_start: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![0u8; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            let abs_j = kv_start + j;
            let future = abs_j > i;
            let outside_window = match window {
                Some(w) => abs_j + w <= i,
                None => false,
            };
            if future || outside_window {
                mask[i * kv_len + j] = 1;
            }
        }
    }
    Tensor::from_slice(&mask, (q_len, kv_len), device)
}

// ── John A0 / SWA prefill integrity (static geometry) ─────────────────────────

/// Count unmasked keys per query row under the same rules as [`make_mask`].
///
/// Returns one count per query position `0..q_len`. A zero means that query's
/// softmax has empty support → NaN / pad-class degeneration (mlxcel #401 family).
pub fn valid_keys_per_query(
    q_len: usize,
    kv_len: usize,
    kv_start: usize,
    window: Option<usize>,
) -> Vec<usize> {
    let mut counts = vec![0usize; q_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            let abs_j = kv_start + j;
            let future = abs_j > i;
            let outside_window = match window {
                Some(w) => abs_j + w <= i,
                None => false,
            };
            if !future && !outside_window {
                counts[i] += 1;
            }
        }
    }
    counts
}

/// Prefill geometry **before** the A0 fix: trim K/V to last `window` then mask
/// full `q_len`. Empty rows begin at `q_len = window + 1`.
pub fn legacy_trim_prefill_valid_keys(prefill_len: usize, window: usize) -> Vec<usize> {
    let kv_len = prefill_len.min(window);
    let kv_start = prefill_len.saturating_sub(kv_len);
    valid_keys_per_query(prefill_len, kv_len, kv_start, Some(window))
}

/// Prefill geometry **after** the A0 fix: keep full K/V; SWA is mask-only.
/// Every query in a causal prefill has ≥ 1 valid key.
pub fn fixed_prefill_valid_keys(prefill_len: usize, window: usize) -> Vec<usize> {
    valid_keys_per_query(prefill_len, prefill_len, 0, Some(window))
}

/// How many query rows have zero valid keys (empty softmax support).
pub fn empty_valid_key_rows(counts: &[usize]) -> usize {
    counts.iter().filter(|&&n| n == 0).count()
}

// ── Layer ─────────────────────────────────────────────────────────────────────

struct Layer {
    attn_norm: RmsNorm,
    attn_q: QMatMul,
    attn_q_norm: RmsNorm,
    attn_k: QMatMul,
    attn_k_norm: RmsNorm,
    /// Missing when attention_k_eq_v / GGUF omits v_proj → V ← K.
    attn_v: Option<QMatMul>,
    attn_out: QMatMul,
    post_attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    post_ffn_norm: RmsNorm,
    /// Optional residual scale after the block (GGUF `layer_output_scale`).
    out_scale: Option<Tensor>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// Sliding-window size if this layer is SWA; None = full causal.
    swa_window: Option<usize>,
    /// Attention score multiplier (HF dense path often 1.0).
    attn_scale: f64,
    /// true = interleaved rope_i (Gemma3-style); false = NeoX half-split rope.
    rope_interleaved: bool,
    cos: Tensor,
    sin: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Layer {
    fn apply_rope(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, _h, seq, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq)?;
        let sin = self.sin.narrow(0, index_pos, seq)?;
        let x = x.contiguous()?;
        if self.rope_interleaved {
            candle_nn::rotary_emb::rope_i(&x, &cos, &sin)
        } else {
            candle_nn::rotary_emb::rope(&x, &cos, &sin)
        }
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        index_pos: usize,
        layer_idx: usize,
        hook: &mut Option<&mut dyn LayerHook>,
    ) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        let device = xs.device();

        let residual = xs;
        let h = self.attn_norm.forward(xs)?;

        let q = self.attn_q.forward(&h)?;
        let k = self.attn_k.forward(&h)?;
        let v = match &self.attn_v {
            Some(proj) => proj.forward(&h)?,
            None => k.clone(), // alternative attention: V ← K when v_proj absent
        };

        let q = q
            .reshape((b, seq, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // QK RMS norms before RoPE (same family as Gemma 3)
        let q = self
            .attn_q_norm
            .forward(&q.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?;
        let k = self
            .attn_k_norm
            .forward(&k.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?;
        // V: RMS without learned weight (HF/G4 path; no v_norm tensor in dense GGUF)
        let v = {
            let eps = 1e-6f64;
            let v2 = v.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            let inv = (v2 + eps)?.sqrt()?.recip()?;
            v.broadcast_mul(&inv)?
        };

        let q = self.apply_rope(&q, index_pos)?;
        let k = self.apply_rope(&k, index_pos)?;

        // KV cache + SWA trim
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((kc, vc)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[kc, &k], 2)?;
                    let v = Tensor::cat(&[vc, &v], 2)?;
                    (k, v)
                }
            }
        };
        // SWA K/V policy (John A0 / 2026-07-30):
        // - **Prefill** (`seq > 1`): keep full K/V. Sliding window is enforced by the
        //   attention mask only. Trimming to the last `w` keys *before* masking left
        //   early query rows with zero valid keys once prefill_len > w (empty softmax
        //   support → NaN / pad-class collapse; see mlxcel #401, HF John6666 note).
        // - **Decode** (`seq == 1`): trim the rotating cache to `w` for memory; the
        //   single query sits at the end of the window so every retained key is valid.
        let (k, v) = if let Some(w) = self.swa_window {
            let t = k.dim(2)?;
            if seq == 1 && t > w {
                let start = t - w;
                (k.narrow(2, start, w)?, v.narrow(2, start, w)?)
            } else {
                (k, v)
            }
        } else {
            (k, v)
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let n_rep = self.n_head / self.n_kv_head.max(1);
        let kv_len = k.dim(2)?;
        // Absolute position of first key after any decode-only SWA trim.
        let kv_start = index_pos + seq.saturating_sub(kv_len);
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        // Prefill mask: [q_len, kv_len] causal (+ SWA). Full K/V on prefill ⇒ no empty rows.
        // Decode seq==1: single query vs windowed cache — all past keys are valid.
        let mask = if seq > 1 {
            Some(make_mask(seq, kv_len, kv_start, self.swa_window, device)?)
        } else {
            None
        };

        let att = (q.matmul(&k.t()?)? * self.attn_scale)?;
        let att = match mask {
            None => att,
            Some(m) => {
                let neg_inf = Tensor::new(f32::NEG_INFINITY, att.device())?;
                let m = m.broadcast_as(att.shape())?;
                m.where_cond(&neg_inf.broadcast_as(att.shape())?, &att)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, self.n_head * self.head_dim))?;
        let y = self.attn_out.forward(&y)?;
        let y = self.post_attn_norm.forward(&y)?;
        let h = (y + residual)?;
        let h = maybe_apply(hook, HookSite::PostAttn, layer_idx, h)?;

        // FFN — GELU gate (Gemma 4 dense path; candle-nn 0.9 has no ops::gelu)
        let residual_ffn = &h;
        let h_normed = self.ffn_norm.forward(&h)?;
        let gate = gelu_tanh(&self.ffn_gate.forward(&h_normed)?)?;
        let up = self.ffn_up.forward(&h_normed)?;
        let ffn_out = self.ffn_down.forward(&(gate * up)?)?;
        let ffn_out = self.post_ffn_norm.forward(&ffn_out)?;
        let mut h = (ffn_out + residual_ffn)?;

        if let Some(scale) = &self.out_scale {
            h = h.broadcast_mul(scale)?;
        }
        Ok(h)
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: Option<QMatMul>,
    /// From GGUF / HF config `final_logit_softcapping` (0 = off).
    final_logit_softcap: f64,
    #[allow(dead_code)]
    device: Device,
    pub hidden_dim: usize,
}

fn meta_u32(ct: &gguf_file::Content, key: &str) -> Result<u32> {
    ct.metadata
        .get(key)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF key {key}")))?
        .to_u32()
}

fn meta_f32(ct: &gguf_file::Content, key: &str, default: f32) -> f32 {
    ct.metadata
        .get(key)
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(default)
}

fn meta_u32_array(ct: &gguf_file::Content, key: &str, n: usize, fill: u32) -> Vec<u32> {
    let mut out = vec![fill; n];
    let Some(v) = ct.metadata.get(key) else {
        return out;
    };
    let Ok(arr) = v.to_vec() else {
        // scalar fallback
        if let Ok(x) = v.to_u32() {
            return vec![x; n];
        }
        return out;
    };
    for (i, item) in arr.iter().enumerate().take(n) {
        if let Ok(x) = item.to_u32() {
            out[i] = x;
        }
    }
    out
}

fn meta_bool_array(ct: &gguf_file::Content, key: &str, n: usize, fill: bool) -> Vec<bool> {
    let mut out = vec![fill; n];
    let Some(v) = ct.metadata.get(key) else {
        return out;
    };
    let Ok(arr) = v.to_vec() else {
        if let Ok(x) = v.to_bool() {
            return vec![x; n];
        }
        return out;
    };
    for (i, item) in arr.iter().enumerate().take(n) {
        if let Ok(x) = item.to_bool() {
            out[i] = x;
        } else if let Ok(x) = item.to_u32() {
            out[i] = x != 0;
        }
    }
    out
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let block_count = meta_u32(&ct, "gemma4.block_count")? as usize;
        let hidden_dim = meta_u32(&ct, "gemma4.embedding_length")? as usize;
        let n_head = meta_u32(&ct, "gemma4.attention.head_count")? as usize;
        let head_dim_full = meta_u32(&ct, "gemma4.attention.key_length")? as usize;
        let head_dim_swa = ct
            .metadata
            .get("gemma4.attention.key_length_swa")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(head_dim_full / 2);
        let rms_eps = meta_f32(&ct, "gemma4.attention.layer_norm_rms_epsilon", 1e-6) as f64;
        let rope_base_full = meta_f32(&ct, "gemma4.rope.freq_base", 1_000_000.0);
        let rope_base_swa = meta_f32(&ct, "gemma4.rope.freq_base_swa", 10_000.0);
        let swa_window = meta_u32(&ct, "gemma4.attention.sliding_window").unwrap_or(1024) as usize;
        // Pattern: true = SWA layer (local), false = full attention
        let is_swa = meta_bool_array(
            &ct,
            "gemma4.attention.sliding_window_pattern",
            block_count,
            true,
        );
        let _kv_heads_meta = meta_u32_array(
            &ct,
            "gemma4.attention.head_count_kv",
            block_count,
            (n_head / 2).max(1) as u32,
        );
        // Partial rotary on full-attn (HF assets: partial_rotary_factor=0.25).
        // GGUF often stores dimension_count == head_dim (storage size), which is *not*
        // proof of full rotation — so we only trust it when it is *smaller* than head_dim.
        // Override: GEMMA4_PARTIAL_ROTARY=1.0 for full rotate, or e.g. 0.25.
        let partial_factor = std::env::var("GEMMA4_PARTIAL_ROTARY")
            .ok()
            .and_then(|s| s.parse::<f32>().ok());
        let n_rot_meta = ct
            .metadata
            .get("gemma4.rope.dimension_count")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let n_rot_full = {
            let from_factor = |f: f32| (((head_dim_full as f32) * f).round() as usize).max(2) & !1;
            if let Some(f) = partial_factor {
                from_factor(f.clamp(0.0, 1.0))
            } else if let Some(n) = n_rot_meta.filter(|&n| n > 0 && n < head_dim_full) {
                n & !1
            } else {
                from_factor(0.25) // HF default for full_attention
            }
        };
        let n_rot_swa = ct
            .metadata
            .get("gemma4.rope.dimension_count_swa")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .filter(|&n| n > 0 && n <= head_dim_swa)
            .unwrap_or(head_dim_swa)
            & !1;
        // Attn scale: default 1.0 (not 1/√d). Override: GEMMA4_ATTN_SCALE=rsqrt
        let attn_scale_mode = std::env::var("GEMMA4_ATTN_SCALE").unwrap_or_else(|_| "one".into());
        // RoPE layout: default NeoX half (`rope`). GEMMA4_ROPE_STYLE=i → interleaved.
        let rope_interleaved = matches!(
            std::env::var("GEMMA4_ROPE_STYLE")
                .unwrap_or_else(|_| "half".into())
                .to_ascii_lowercase()
                .as_str(),
            "i" | "interleaved" | "rope_i"
        );
        let final_logit_softcap = meta_f32(&ct, "gemma4.final_logit_softcapping", 30.0) as f64;

        println!(
            "    [Gemma4] blocks={} hidden={} heads={} head_dim_full={} head_dim_swa={} n_rot_full={} n_rot_swa={} swa_win={} rope_full={:.0} rope_swa={:.0} logit_softcap={:.1} attn_scale={} rope_style={}",
            block_count, hidden_dim, n_head, head_dim_full, head_dim_swa, n_rot_full, n_rot_swa, swa_window, rope_base_full, rope_base_swa, final_logit_softcap, attn_scale_mode,
            if rope_interleaved { "interleaved" } else { "half" }
        );

        // Full layers: GGUF rope_freqs if present, pad to head_dim/2, then enforce partial
        // (zero inv pairs beyond n_rot/2 so non-rotated dims stay identity).
        let (cos_full, sin_full) = match ct.tensor(reader, "rope_freqs.weight", device) {
            Ok(t) => {
                let deq = t.dequantize(device)?;
                let inv_raw: Vec<f32> = deq.to_vec1()?;
                let mut inv = pad_inv_freq(&inv_raw, head_dim_full);
                let keep = (n_rot_full / 2).min(inv.len());
                for x in inv.iter_mut().skip(keep) {
                    *x = 0.0;
                }
                let nonzero = inv.iter().filter(|&&x| x != 0.0).count();
                println!(
                    "    [Gemma4] full-attn RoPE rope_freqs.weight (raw_len={} pad={} keep_pairs={} nonzero={})",
                    inv_raw.len(),
                    inv.len(),
                    keep,
                    nonzero
                );
                precompute_freqs_from_inv_freq(&inv, device)?
            }
            Err(_) => {
                println!(
                    "    [Gemma4] full-attn RoPE proportional+partial base={rope_base_full} n_rot={n_rot_full}"
                );
                precompute_proportional_partial(head_dim_full, n_rot_full, rope_base_full, device)?
            }
        };
        let (cos_swa, sin_swa) = if n_rot_swa < head_dim_swa {
            println!(
                "    [Gemma4] SWA RoPE partial n_rot={n_rot_swa} head_dim={head_dim_swa} base={rope_base_swa}"
            );
            precompute_proportional_partial(head_dim_swa, n_rot_swa, rope_base_swa, device)?
        } else {
            precompute_freqs_cis(head_dim_swa, rope_base_swa, device)?
        };

        let tok_embd_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embd = tok_embd_q.dequantize(device)?;
        let norm =
            RmsNorm::from_qtensor(ct.tensor(reader, "output_norm.weight", device)?, rms_eps)?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => Some(QMatMul::from_qtensor(t)?),
            Err(_) => None,
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let p = format!("blk.{i}");
            if i % 10 == 0 {
                println!("    Loading Gemma4 layer {i}/{block_count}...");
            }
            // Prefer tensor shapes over metadata arrays (arrays can parse poorly).
            let q_norm_t = ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?;
            let head_dim = q_norm_t.shape().dims()[0];
            let swa = head_dim == head_dim_swa
                || (head_dim != head_dim_full && is_swa.get(i).copied().unwrap_or(true));
            let q_w = ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?;
            let k_w = ct.tensor(reader, &format!("{p}.attn_k.weight"), device)?;
            let q_dims = q_w.shape().dims().to_vec();
            let k_dims = k_w.shape().dims().to_vec();
            // Candle GGUF: weight often [out_features, in_features] or reverse — pick dim ≠ hidden.
            let pick_out = |dims: &[usize]| {
                dims.iter()
                    .copied()
                    .find(|&d| d != hidden_dim && d % head_dim == 0)
                    .unwrap_or(head_dim)
            };
            let q_out = pick_out(&q_dims);
            let k_out = pick_out(&k_dims);
            let n_head_layer = (q_out / head_dim).max(1);
            let n_kv = (k_out / head_dim).max(1);
            let attn_scale = if attn_scale_mode == "one" {
                1.0
            } else {
                1.0 / (head_dim as f64).sqrt()
            };
            let (cos, sin) = if swa {
                (cos_swa.clone(), sin_swa.clone())
            } else {
                (cos_full.clone(), sin_full.clone())
            };

            let out_scale = ct
                .tensor(reader, &format!("{p}.layer_output_scale.weight"), device)
                .ok()
                .and_then(|t| t.dequantize(device).ok());

            if i < 8 || i % 10 == 0 {
                println!(
                    "      layer {i}: head_dim={head_dim} n_head={n_head_layer} n_kv={n_kv} swa={swa} q={q_dims:?} k={k_dims:?}"
                );
            }

            layers.push(Layer {
                attn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                    rms_eps,
                )?,
                attn_q: QMatMul::from_qtensor(q_w)?,
                attn_q_norm: RmsNorm::from_qtensor(q_norm_t, rms_eps)?,
                attn_k: QMatMul::from_qtensor(k_w)?,
                attn_k_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device)?,
                    rms_eps,
                )?,
                attn_v: ct
                    .tensor(reader, &format!("{p}.attn_v.weight"), device)
                    .ok()
                    .and_then(|t| QMatMul::from_qtensor(t).ok()),
                attn_out: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.attn_output.weight"),
                    device,
                )?)?,
                post_attn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.post_attention_norm.weight"), device)?,
                    rms_eps,
                )?,
                ffn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?,
                    rms_eps,
                )?,
                ffn_gate: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_gate.weight"),
                    device,
                )?)?,
                ffn_up: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_up.weight"),
                    device,
                )?)?,
                ffn_down: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_down.weight"),
                    device,
                )?)?,
                post_ffn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.post_ffw_norm.weight"), device)?,
                    rms_eps,
                )?,
                out_scale,
                n_head: n_head_layer,
                n_kv_head: n_kv,
                head_dim,
                swa_window: if swa { Some(swa_window) } else { None },
                attn_scale,
                rope_interleaved,
                cos,
                sin,
                kv_cache: None,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embd, hidden_dim),
            layers,
            norm,
            output,
            final_logit_softcap,
            device: device.clone(),
            hidden_dim,
        })
    }

    fn run_layers(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.run_layers_hooked(tokens, index_pos, None)
    }

    /// `run_layers` with an optional forward-pass physics hook.
    ///
    /// The sites reached from *this* function are the block boundaries: `PreLayer`, `PostMlp`
    /// and `FinalNorm`. `PostAttn` is live too — `Layer::forward` takes the hook and fires it
    /// itself (see the `maybe_apply` inside), so all four `HookSite` variants work on Gemma 4.
    /// As in gemma.rs, the `√hidden_dim` scale means these activations are not in
    /// pre-`lm_head` residual space — hook forces are scale-free for that reason.
    fn run_layers_hooked(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
        mut hook: Option<&mut dyn LayerHook>,
    ) -> Result<Tensor> {
        let (_b, seq) = tokens.dims2()?;
        if let Some(hk) = hook.as_deref_mut() {
            hk.begin_token(self.layers.len());
        }
        let mut h = self.tok_embeddings.forward(tokens)?;
        let scale = (self.hidden_dim as f64).sqrt();
        h = (h * scale)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            h = maybe_apply(&mut hook, HookSite::PreLayer, i, h)?;
            h = layer.forward(&h, index_pos, i, &mut hook)?;
            h = maybe_apply(&mut hook, HookSite::PostMlp, i, h)?;
        }
        let h = self.norm.forward(&h)?;
        let h = maybe_apply(&mut hook, HookSite::FinalNorm, self.layers.len(), h)?;
        h.narrow(1, seq - 1, 1)?.squeeze(1)
    }

    /// Number of transformer blocks — lets a hook resolve a depth-fraction band.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn forward(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        let hidden = self.run_layers(tokens, index_pos)?;
        self.project_hidden_to_logits(&hidden)
    }

    pub fn forward_with_hidden(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_with_hidden_hooked(tokens, index_pos, None)
    }

    /// `forward_with_hidden` with an optional forward-pass physics hook.
    pub fn forward_with_hidden_hooked(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
        hook: Option<&mut dyn LayerHook>,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.run_layers_hooked(tokens, index_pos, hook)?;
        let logits = self.project_hidden_to_logits(&hidden)?;
        Ok((logits, hidden))
    }

    pub fn project_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.project_hidden_to_logits(hidden)
    }

    /// Final norm + lm_head — the Jacobian lens's `unembed`.
    ///
    /// `project_to_logits` skips the final norm because `run_layers` already applied it.
    /// The lens transports a *block output* (pre-norm), so the norm belongs here; folding
    /// RMSNorm into the fitted `J` would stop matching `jlens.protocol.LensModel.unembed`.
    ///
    /// Flattens to 2-D first: the tied-weight branch of `project_hidden_to_logits` uses
    /// `matmul`, which needs equal ranks, so a `[b, seq, d]` residual would fail there.
    pub fn unembed(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(hidden)?.contiguous()?;
        let mut shape = normed.dims().to_vec();
        let d = *shape.last().expect("residual has a last dim");
        let logits = self.project_hidden_to_logits(&normed.reshape(((), d))?)?;
        let vocab = logits.dim(logits.rank() - 1)?;
        *shape.last_mut().expect("residual has a last dim") = vocab;
        logits.reshape(shape)
    }

    fn project_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let logits = match &self.output {
            Some(out) => out.forward(hidden)?,
            None => {
                let emb = self.tok_embeddings.embeddings();
                // `run_layers` hands us `narrow(..).squeeze(..)`, which is non-contiguous
                // as soon as batch > 1, and candle's matmul rejects that. Transposed-
                // contiguous rhs is fine, so only the lhs needs the copy — never
                // `emb.t().contiguous()`, which would materialise the whole vocab matrix.
                hidden.contiguous()?.matmul(&emb.t()?)?
            }
        };
        // HF config: final_logit_softcapping (softcap * tanh(logits/softcap))
        if self.final_logit_softcap > 0.0 {
            let s = self.final_logit_softcap;
            let scaled = (logits / s)?.tanh()?;
            Ok((scaled * s)?)
        } else {
            Ok(logits)
        }
    }

    pub fn token_embeddings(&self) -> &Tensor {
        self.tok_embeddings.embeddings()
    }

    /// Clear per-layer attention cache before prefilling a fresh transcript.
    /// Required for multi-turn coherence when each turn re-prefills history.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
    }
}

// ── A0 unit tests (no GPU / no weights) ───────────────────────────────────────

#[cfg(test)]
mod a0_swa_tests {
    use super::*;

    const W: usize = 1024;

    #[test]
    fn john_table_legacy_trim_empty_rows() {
        // John's arithmetic: empty rows appear immediately above the window.
        assert_eq!(empty_valid_key_rows(&legacy_trim_prefill_valid_keys(1023, W)), 0);
        assert_eq!(empty_valid_key_rows(&legacy_trim_prefill_valid_keys(1024, W)), 0);
        assert_eq!(empty_valid_key_rows(&legacy_trim_prefill_valid_keys(1025, W)), 1);
        assert_eq!(empty_valid_key_rows(&legacy_trim_prefill_valid_keys(1039, W)), 15);
        assert_eq!(empty_valid_key_rows(&legacy_trim_prefill_valid_keys(2048, W)), 1024);
    }

    #[test]
    fn a0_fixed_prefill_no_empty_rows_at_boundary() {
        for len in [1usize, 512, 1023, 1024, 1025, 1039, 2048, 4096] {
            let counts = fixed_prefill_valid_keys(len, W);
            let empty = empty_valid_key_rows(&counts);
            assert_eq!(
                empty, 0,
                "prefill_len={len}: expected 0 empty rows, got {empty}"
            );
            // Every query has at least itself (causal).
            assert!(counts.iter().all(|&n| n >= 1));
            // First query sees only key 0; last query sees min(len, W) keys.
            assert_eq!(counts[0], 1);
            assert_eq!(*counts.last().unwrap(), len.min(W));
        }
    }

    #[test]
    fn a0_boundary_lengths_match_john_checklist() {
        for &len in &[1023usize, 1024, 1025, 1039] {
            assert_eq!(
                empty_valid_key_rows(&fixed_prefill_valid_keys(len, W)),
                0,
                "A0 gate fail at prefill_len={len}"
            );
        }
    }

    #[test]
    fn valid_keys_self_consistent_with_window() {
        let len = 1500;
        let counts = fixed_prefill_valid_keys(len, W);
        for (i, &n) in counts.iter().enumerate() {
            // Causal + SWA: keys in [max(0, i+1-W), i] inclusive → count = min(i+1, W)
            let expected = (i + 1).min(W);
            assert_eq!(n, expected, "query {i}");
        }
    }
}
