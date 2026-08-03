//! Vendored quantized Gemma 3 model — GGUF native reader.
//!
//! MODIFIED — adapted from in-tree `quantized_gemma.rs` / `llama.rs` vendoring
//! pattern with physics steering extensions (`forward_with_hidden`,
//! `project_to_logits`, `token_embeddings`) for main.rs and the Niodoo engine.
//!
//! Gemma 3 weights: Google Gemma Terms of Use (NOT Apache — Gemma 4 is Apache).
//! Candle loader code in this file: Apache-2.0 OR MIT. See NOTICE.
//!
//! Currently unused by `main()` — Llama is the only backend wired in. Kept as
//! scaffolding for the planned Gemma 3 swap, so the whole module is marked
//! `allow(dead_code)`.

#![allow(dead_code)]
//!
//! Gemma 3 architecture specifics:
//!   - Per-head QK RMS norms (attn_q_norm, attn_k_norm)
//!   - Post-attention and post-FFW scaling norms (Gemma 3 pre-residual)
//!   - GeGLU activation in the MLP (gate × up, then down)
//!   - Grouped query attention (n_head != n_kv_head)
//!   - Token embeddings shared with output projection (tied weights)
//!   - Embedding scaling by sqrt(hidden_dim)

use crate::hooks::{maybe_apply, HookSite, LayerHook};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use std::collections::HashMap;

pub const MAX_SEQ_LEN: usize = 8192;

// ── QMatMul ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qt: QTensor) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self {
            inner: candle_core::quantized::QMatMul::from_qtensor(qt)?,
            span,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// ── RoPE precomputation ───────────────────────────────────────────────────────

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx.cos()?, idx.sin()?))
}

// ── GQA repeat_kv ─────────────────────────────────────────────────────────────

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, seq, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, seq, d))?
        .reshape((b, n_kv * n_rep, seq, d))
}

// ── Causal mask ───────────────────────────────────────────────────────────────

fn make_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}

// ── Layer ─────────────────────────────────────────────────────────────────────

struct Layer {
    // Pre-attention norm
    attn_norm: RmsNorm,
    // Attention projections
    attn_q: QMatMul,
    attn_q_norm: RmsNorm,
    attn_k: QMatMul,
    attn_k_norm: RmsNorm,
    attn_v: QMatMul,
    attn_out: QMatMul,
    // Post-attention scaling (Gemma 3 pre-residual norm)
    post_attn_norm: RmsNorm,
    // FFN
    ffn_norm: RmsNorm,
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    post_ffn_norm: RmsNorm,
    // Config
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    // KV cache
    kv_cache: Option<(Tensor, Tensor)>,
    // Tracing spans
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

impl Layer {
    fn apply_rope(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b, _h, seq, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq)?;
        let sin = self.sin.narrow(0, index_pos, seq)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;

        // ── Pre-attention norm ───────────────────────────────────────────────
        let residual = xs;
        let h = self.attn_norm.forward(xs)?;

        // ── Attention projections ────────────────────────────────────────────
        let _enter_attn = self.span_attn.enter();
        let q = self.attn_q.forward(&h)?;
        let k = self.attn_k.forward(&h)?;
        let v = self.attn_v.forward(&h)?;

        // Reshape to multi-head layout
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

        // Gemma 3: per-head QK RMS norms (applied before RoPE)
        let q = self
            .attn_q_norm
            .forward(&q.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?;
        let k = self
            .attn_k_norm
            .forward(&k.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?;

        // RoPE
        let q = self.apply_rope(&q, index_pos)?;
        let k = self.apply_rope(&k, index_pos)?;

        // KV cache
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
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA expand
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Scaled dot-product
        let scale = (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? / scale)?;
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

        // Post-attention pre-residual norm (Gemma 3)
        let y = self.post_attn_norm.forward(&y)?;
        let h = (y + residual)?;

        // ── FFN (GeGLU) ──────────────────────────────────────────────────────
        let _enter_mlp = self.span_mlp.enter();
        let residual_ffn = &h;
        let h_normed = self.ffn_norm.forward(&h)?;
        let gate = candle_nn::ops::silu(&self.ffn_gate.forward(&h_normed)?)?;
        let up = self.ffn_up.forward(&h_normed)?;
        let ffn_out = self.ffn_down.forward(&(gate * up)?)?;
        // Post-FFN pre-residual norm (Gemma 3)
        let ffn_out = self.post_ffn_norm.forward(&ffn_out)?;
        let h = (ffn_out + residual_ffn)?;

        Ok(h)
    }
}

// ── ModelWeights ──────────────────────────────────────────────────────────────

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: Option<QMatMul>,
    masks: HashMap<usize, Tensor>,
    #[allow(dead_code)]
    device: Device,
    pub hidden_dim: usize,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    /// Load a quantized Gemma 3 model directly from a GGUF file.
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Try both "gemma3.*" and "llama.*" namespaces — Unsloth uses llama
        let get_u32 = |k1: &str, k2: &str| -> Result<u32> {
            ct.metadata
                .get(k1)
                .or_else(|| ct.metadata.get(k2))
                .ok_or_else(|| candle_core::Error::Msg(format!("missing key {k1} or {k2}")))?
                .to_u32()
        };

        let get_f32 = |k1: &str, k2: &str, default: f32| -> f32 {
            ct.metadata
                .get(k1)
                .or_else(|| ct.metadata.get(k2))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(default)
        };

        let n_head = get_u32("gemma3.attention.head_count", "llama.attention.head_count")? as usize;
        let n_kv_head = get_u32(
            "gemma3.attention.head_count_kv",
            "llama.attention.head_count_kv",
        )? as usize;
        let block_count = get_u32("gemma3.block_count", "llama.block_count")? as usize;
        let hidden_dim = get_u32("gemma3.embedding_length", "llama.embedding_length")? as usize;
        let rms_eps = get_f32(
            "gemma3.attention.layer_norm_rms_epsilon",
            "llama.attention.layer_norm_rms_epsilon",
            1e-6,
        ) as f64;
        let rope_base = get_f32("gemma3.rope.freq_base", "llama.rope.freq_base", 10000.0);

        // Gemma 3: head_dim is NOT hidden_dim/n_head (5376/32=168 is WRONG).
        // Read from GGUF metadata (attention.key_length) or infer from Q norm tensor.
        // Gemma 3 27B uses head_dim=128, Q output = n_head * 128 = 4096.
        let head_dim = ct
            .metadata
            .get("gemma3.attention.key_length")
            .or_else(|| ct.metadata.get("llama.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                // Fallback: infer from blk.0.attn_q_norm.weight shape
                // That tensor is [head_dim], so its shape tells us directly
                ct.tensor(reader, "blk.0.attn_q_norm.weight", device)
                    .map(|t| t.shape().dims()[0])
                    .unwrap_or(hidden_dim / n_head)
            });

        println!(
            "    [Gemma3] heads={}, kv_heads={}, blocks={}, hidden={}, head_dim={}, rope_base={:.1}",
            n_head, n_kv_head, block_count, hidden_dim, head_dim, rope_base
        );

        let (cos, sin) = precompute_freqs_cis(head_dim, rope_base, device)?;

        // ── Token embeddings ──────────────────────────────────────────────────
        let tok_embd_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embd = tok_embd_q.dequantize(device)?;

        // ── Output norm ───────────────────────────────────────────────────────
        let norm =
            RmsNorm::from_qtensor(ct.tensor(reader, "output_norm.weight", device)?, rms_eps)?;

        // ── Output projection (may be tied to token embeddings) ───────────────
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => Some(QMatMul::from_qtensor(t)?),
            Err(_) => None,
        };

        // ── Transformer layers ────────────────────────────────────────────────
        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let p = format!("blk.{i}");
            if i % 10 == 0 {
                println!("    Loading layer {}/{}...", i, block_count);
            }
            let span_attn = tracing::span!(tracing::Level::TRACE, "gemma-attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "gemma-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "gemma-mlp");

            layers.push(Layer {
                attn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                    rms_eps,
                )?,
                attn_q: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.attn_q.weight"),
                    device,
                )?)?,
                attn_q_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?,
                    rms_eps,
                )?,
                attn_k: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.attn_k.weight"),
                    device,
                )?)?,
                attn_k_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device)?,
                    rms_eps,
                )?,
                attn_v: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.attn_v.weight"),
                    device,
                )?)?,
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
                n_head,
                n_kv_head,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "gemma-model");
        let span_output = tracing::span!(tracing::Level::TRACE, "gemma-output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embd, hidden_dim),
            layers,
            norm,
            output,
            masks: HashMap::new(),
            device: device.clone(),
            hidden_dim,
            span,
            span_output,
        })
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn cached_mask(&mut self, seq_len: usize, device: &Device) -> Result<Tensor> {
        if let Some(m) = self.masks.get(&seq_len) {
            return Ok(m.clone());
        }
        let m = make_causal_mask(seq_len, device)?;
        self.masks.insert(seq_len, m.clone());
        Ok(m)
    }

    /// Run all transformer layers; returns post-norm hidden state (b, hidden_dim).
    fn run_layers(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.run_layers_hooked(tokens, index_pos, None)
    }

    /// `run_layers` with an optional forward-pass physics hook.
    ///
    /// `Layer::forward` is opaque here (unlike llama.rs), so only the block boundaries
    /// are hookable: `PreLayer`, `PostMlp` (== block output) and `FinalNorm`.
    /// Note the `√hidden_dim` embedding scale below — activations inside this stack are
    /// NOT in the same space as the pre-`lm_head` residual, which is why hook forces are
    /// scale-free (see hooks.rs module docs).
    fn run_layers_hooked(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
        mut hook: Option<&mut dyn LayerHook>,
    ) -> Result<Tensor> {
        let (_b, seq) = tokens.dims2()?;
        // Compute mask before taking span borrow to avoid borrow conflict
        let mask = if seq == 1 {
            None
        } else {
            Some(self.cached_mask(seq, tokens.device())?)
        };

        let _enter = self.span.enter();
        if let Some(hk) = hook.as_deref_mut() {
            hk.begin_token(self.layers.len());
        }
        let mut h = self.tok_embeddings.forward(tokens)?;
        // Gemma embedding scaling: multiply by sqrt(hidden_dim)
        let scale = (self.hidden_dim as f64).sqrt();
        h = (h * scale)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            h = maybe_apply(&mut hook, HookSite::PreLayer, i, h)?;
            h = layer.forward(&h, mask.as_ref(), index_pos)?;
            h = maybe_apply(&mut hook, HookSite::PostMlp, i, h)?;
        }
        // Final output norm
        let h = self.norm.forward(&h)?;
        let h = maybe_apply(&mut hook, HookSite::FinalNorm, self.layers.len(), h)?;
        // Return last position: [b, hidden_dim] — squeeze seq dim
        h.narrow(1, seq - 1, 1)?.squeeze(1)
    }

    /// Number of transformer blocks — lets a hook resolve a depth-fraction band.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    // ── Public API (matching llama.rs interface) ──────────────────────────────

    /// Standard forward pass: returns logits [b, vocab_size].
    pub fn forward(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        let hidden = self.run_layers(tokens, index_pos)?;
        let _enter = self.span_output.enter();
        self.project_hidden_to_logits(&hidden)
    }

    /// Return only the hidden state (D-dimensional, pre-lm_head).
    #[allow(dead_code)]
    pub fn forward_hidden(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.run_layers(tokens, index_pos)
    }

    /// Return both logits AND hidden state in one pass (no wasted compute).
    /// Returns (logits, hidden_state) — matches llama.rs interface for Niodoo.
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
        let _enter = self.span_output.enter();
        let logits = self.project_hidden_to_logits(&hidden)?;
        Ok((logits, hidden))
    }

    /// Project a hidden state through the lm_head to get logits.
    /// Used for projecting steered hidden states back to vocab space.
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

    /// Internal: project hidden → logits, handling tied vs untied weights.
    fn project_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        match &self.output {
            Some(out) => out.forward(hidden),
            None => {
                // Tied weights: project through tok_embeddings matrix
                // hidden: [b, hidden_dim], emb: [vocab, hidden_dim]
                // logits = hidden @ emb.T
                //
                // `contiguous()` on the lhs only: `run_layers` hands us a narrow+squeeze,
                // which is non-contiguous as soon as batch > 1, and candle's matmul
                // rejects that. A transposed-contiguous rhs is accepted, so `emb.t()`
                // stays as it is — copying it would materialise the whole vocab matrix.
                let emb = self.tok_embeddings.embeddings();
                hidden.contiguous()?.matmul(&emb.t()?)
            }
        }
    }

    /// Access the raw token embedding matrix (vocab_size, hidden_dim).
    /// Used to build the live Diderot field from model weights.
    pub fn token_embeddings(&self) -> &Tensor {
        self.tok_embeddings.embeddings()
    }

    /// Clear per-layer attention cache before prefilling a fresh transcript.
    /// Multi-turn chat re-encodes full history at `index_pos=0`; stale cache
    /// (or SWA edge cases) must not ride into the next prefill.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
    }
}
