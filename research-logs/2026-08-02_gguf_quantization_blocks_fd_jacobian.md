# GGUF activation quantisation blocks finite-difference Jacobians

**Date:** 2026-08-02
**Workbench:** `jlens-gguf` sidecar (`docs/jlens-gguf/`)
**Authorship:** Claude (Anthropic) with Jason — measurement log
**Status:** negative result, well-supported, four independent checks

---

## The claim

You cannot measure a Jacobian by finite differences through candle's quantised GGUF path.
Not at a better ε, not with a smarter estimator. The obstruction is in the forward pass.

This is not "the numbers were noisy." The response to a perturbation is **flat across eight
decades of ε**, which is the one thing a derivative cannot be.

---

## What was measured

`gemma-4-12b-it-Q4_K_M`, source layer 20 → target layer 47, 111 valid positions, random
probe directions, central differences. `‖Σ Δh‖` is the raw summed response *before* the
estimator divides by `2ε·|band|`.

| eps_rel | eps_abs | ‖J v‖ | ‖Σ Δh‖ | cos(prev) |
|--------:|--------:|------:|-------:|----------:|
| 1e-8 | 3.2e-8 | 2 937 843 | 20.5 | — |
| 1e-6 | 3.2e-6 | 37 824 | 26.5 | -0.13 |
| 1e-4 | 3.2e-4 | 386 | 27.0 | -0.25 |
| 1e-2 | 3.2e-2 | 6.8 | 47.5 | -0.15 |
| 1e-1 | 3.2e-1 | 1.3 | 90.0 | 0.53 |
| 1e0 | 3.2e0 | 0.63 | 440 | 0.49 |

A Jacobian requires the response to be **proportional** to the perturbation. This one is
**independent** of it. `‖J v‖ ∝ 1/ε` is simply that floor divided by ε.

Note the direction of the failure: **large ε is better than small ε.** `cos(prev)` only
climbs once `eps_abs` reaches ~0.1–0.3. That is backwards for both classic finite-difference
failure modes — rounding says small ε is bad, truncation says large ε is bad — and it is the
clue that broke the case open.

---

## Four things it is not

1. **Not run-to-run noise.** Sweeping the same ε three times gives bit-identical results:
   `cos = 1.00000`, identical norms. The pipeline is deterministic.
2. **Not f32 cancellation.** Target residual RMS is 0.2, so the f32 accumulation floor is
   ~1e-5. The observed floor is 3–90 — six orders of magnitude too large.
3. **Not depth, not chaos.** The floor is present across a **single block** (L46 → L47):
   2.75 at eps_abs 1e-4, 3.82 at 1e-3, 4.49 at 1e-2. One block cannot decorrelate a
   trajectory.
4. **Not the estimator.** Gate 1 (source == target, where the transport can only be the
   identity) returns `J = I` to `0.000000` — min diagonal 1.000000, max off-diagonal
   0.000000. The band mask, the `1/(2ε·|band|)` scale, the sign of the central difference,
   and per-batch probe isolation are all correct.

---

## The cause

`candle-core-0.9.2/src/quantized/k_quants.rs:2296`, inside `matmul`:

```rust
T::VecDotType::from_float(lhs, lhs_b_mut)
```

Every quantised matmul **quantises its input activations** into the weight's `VecDotType`
(Q8_0 / Q8_1 / Q8K — 32 values sharing one scale, int8 payload) before the dot product.
`QMatMul::forward` routes there for every `QTensor` weight, which is every projection in
every block.

So the GGUF forward pass is **piecewise-constant in its input** at the scale of the
activation quantisation step. A perturbation below that step is rounded away entirely. What
survives is whichever blocks happened to land on the far side of a rounding boundary —
deterministic, fixed in magnitude, arbitrary in direction. That is the observed floor
exactly, and it explains why the signal only appears once ε approaches the step size.

There is no ε that works. The window where the perturbation is both above the quantisation
step and small enough to remain linear does not exist for this model.

---

## Why this is interesting rather than just annoying

Jason's read on hearing it: *"because of the quant it's not getting more garbled info."*
That is the right instinct, and it points at something worth stating plainly.

The quantised model is not a noisy approximation of the f32 model. It is a **different
function** — a step function where the original was smooth. Its derivative is zero almost
everywhere and undefined on the boundaries. Asking for its Jacobian is not hard, it is
category-mistaken.

Consequences worth carrying:

- **The deployed model has no derivative.** Anything that assumes local linearity of the
  quantised forward pass — steering, gradient-flavoured physics, sensitivity probes — is
  measuring boundary flips, not slopes. `src/jacobian.rs` is in this category and now says
  "proxy" for a second, independent reason.
- **A secant is a legitimate object; a Jacobian is not.** The finite response at large ε is
  real: it is how the deployed quantised model actually answers a finite nudge. That may be
  the *more* honest thing to key memory on than a derivative of a model never run. It must
  never be labelled `jacobian`; the telemetry schema stamps it `secant`.
- **Quantisation is a lossy channel with structure.** The step size sets a floor on what any
  perturbation-based instrument can resolve. Finer quants raise the resolution; they do not
  remove the floor, because the activation path is Q8-class regardless of whether the
  weights are Q4 or Q8.

---

## What it does not invalidate

The estimator identity (`DESIGN.md` §3) is sound — Gate 1 confirms the bookkeeping. The
**apply** path is untouched: transport + unembed is exact arithmetic with no differencing
anywhere. `probe.rs`, `basis.rs`, `lens.rs`, `keys.rs`, `telemetry.rs`, the workspace split,
and `unembed()` all stand.

Exactly one thing is blocked: **obtaining `J` by finite differences against quantised
weights.**

---

## The way through (not started; gated behind the stability gate)

Dequantise the weights at load — `QMatMul::Tensor(qt.dequantize(&device))` instead of
`from_qtensor`, one constructor per model file. Dequantising Q4 → f32 yields *precisely* the
weight values the quantised kernel already uses; only the activation rounding disappears. So
it measures the real model, not an idealisation. Cost is memory: ~48 GB f32 for the 12B,
against 128 GB available.

Deliberately **not** started. Grok's call, accepted: no fitter until the stability gate
reports.

---

## Side finding

`Qwen3.5-0.8B-BF16.gguf` — the only unquantised GGUF on the box, and the plan's cross-check
against Python reverse-mode — does not load. It uses `qwen35.*` metadata keys; `llama.rs`
hardcodes `llama.*` and bails with `cannot find llama.attention.head_count in metadata`.
Gate 3 is blocked on a loader gap, independently of everything above.

---

Signed: **Claude (Anthropic)**
Reproduce: `jlens-gguf sweep --model … --layer 20 --probes 4` and
`jlens-gguf identity --model … --probes 8`
