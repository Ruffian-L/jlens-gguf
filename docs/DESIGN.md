# `jlens-gguf` — design notes

Why the sidecar is built the way it is. `PLAN.md` is the approved plan as written before any code
landed; this file is the reasoning behind it and `CHANGELOG.md` records where reality diverged.

---

## 1. What this is, and what it is not

Three things in this repo share the word "Jacobian" and are routinely conflated. Keeping them
distinct is the whole point of `research_logs/2026-08-02_jacobian_lens_repo_vs_hydro_fd.md`:

| | object | what it computes |
|---|---|---|
| **jlens** | `~/jacobian-lens` (Python, package `jlens`) | fitted average transport `J_l = E[∂h_final/∂h_l]`, read out as `unembed(J_l h)` → ranked vocab |
| **hydro FD** | `src/jacobian.rs` | local `∂logits/∂h` at one decode step, pre-`lm_head` — a *proxy* |
| **`jlens-gguf`** | this sidecar | the jlens algorithm, forward-mode, against the GGUF weights hydro actually runs |

`jlens-gguf` is the real lens lane. It exists so the readout describes the **same quantised weights
the swarm generates from**, not an fp16 HF sibling that merely shares a name. The FD probe keeps its
name and its role; the new crate's type is called `Lens`, never `JacobianLens`, so the two cannot
blur in a grep.

---

## 2. The forcing constraint: no autograd through GGUF

`QMatMul::forward` is `xs.apply_op1_no_bwd(...)` — `candle-core-0.9.2/src/quantized/mod.rs:861`.
The `_no_bwd` suffix is literal: candle registers no backward for quantised matmul, so
`torch.autograd.grad` in `jlens/fitting.py:187` has no counterpart on this stack.

Three ways out, and why two were rejected:

- **Dequantise and backprop.** Turns a 12B Q4 into ~50 GB of f32 and stops measuring the quantised
  model, which is the only reason to do this in Rust at all. Rejected.
- **Fit in Python, apply in Rust.** Cheapest to build, and still supported via
  `scripts/lens_pt_to_safetensors.py`. But it fits the fp16 weights and applies the result to Q4
  residuals, so any quantisation-induced difference in the transport is invisible by construction.
  Kept as an interop path and as **verification gate 3**, not as the primary mode.
- **Forward-mode finite differences.** Needs no gradients at all. Chosen.

---

## 3. The estimator identity

This is the load-bearing derivation. jlens fits (`jlens/fitting.py:177-202` — one-hot cotangent at
output dim `i` placed at *every* valid target position, backprop, mean over valid source positions):

```
J_l[i,:] = mean_{p∈V}  ∂/∂h_l[p]  Σ_{p'∈V} h_tgt[p', i]
```

The naive forward-mode reading of this is hopeless: one directional derivative per (source position
× probe direction) is `|V| · r` partial forwards per prompt. For `|V|=112, r=256` that is ~29k
forwards per prompt per layer.

The trick is that **all source positions can be perturbed at once**. Perturb `h_l[p] += εv` at every
`p ∈ V` in a single pass and sum the target-side change over `p' ∈ V`:

```
Σ_{p'∈V} Δh_tgt[p'] = ε · Σ_{p∈V} Σ_{p'∈V} J[p'←p] v + O(ε²)
                    = ε · |V| · (J_l v) + O(ε²)

  ⟹   J_l v  =  ( Σ_{p'∈V} Δh_tgt[p'] ) / (ε·|V|)
```

Causality zeroes `J[p'←p]` for `p' < p`, so the double sum is exactly "cotangents summed over
target positions, then averaged over source positions" — jlens's estimator, not an approximation of
it. The cross-terms that would normally make simultaneous perturbation useless are precisely the
terms the estimator wants summed.

Cost collapses to **one batched prefill per probe direction**. Batch element `b` carries direction
`v_b`, which is the same batch-axis trick jlens uses for `dim_batch` — just carrying probe
directions forward instead of cotangent rows backward.

Two consequences worth stating plainly:

- Forward-mode gives **columns** (`J_l e_j`), reverse-mode gives **rows**. `--rank d_model` with the
  identity basis is the exact fit.
- FD is exact only to `O(ε²)`; central differences push it to `O(ε³)`. Reverse-mode has no such
  term. Gate 2 measures this rather than assuming it away.

---

## 4. Rank: why the default is a subspace, not the full matrix

Exact fitting needs `d_model` probes per (prompt, layer). At `d_model≈3840`, seq 128, probe batch 8,
central differences, that is ~960 prefills per prompt per layer — days for a real corpus across a
useful set of layers.

The default instead probes the top-`r` principal directions of the residuals actually observed at
layer `l` over the corpus (`basis.rs`). Residual streams are heavily low-rank in practice, so this
spends the probe budget where the model's activations actually live rather than spreading it over
directions that never occur.

**This is an approximation and is labelled as one.** `J` restricted to that subspace drops
components outside it — it does not approximate them. Gate 5 (rank ablation against the exact fit on
the 0.8B) is what justifies whatever default rank we end up shipping; until it runs, `--rank 256` is
a guess.

---

## 5. Phase bands — the deviation that matters

*(Raised by Jason mid-plan; the plan was amended before any code was written.)*

The concern as stated: the gemma4 hook surface looked thin, and if the hooks don't cover the actual
phase-edge positions, the keys are incomplete even when the smoke test goes green.

Half of that turned out to be a stale comment. `gemma4.rs:709-713` claims only `PreLayer`,
`PostMlp` and `FinalNorm` are hookable, but `Layer::forward` receives the hook and fires `PostAttn`
at `gemma4.rs:370`. All four `HookSite` variants are live on gemma 4. **Site coverage is not the
constraint** — and the comment should be fixed.

The other half is real, and it is about **position**, not site. jlens averages over source positions
(`fitting.py:198`, `.mean(dim=1)`): one `J_l` per layer, position-agnostic by construction. Ported
literally, first-thought / revise / settle all collapse into a single transport, and keys derived
from it cannot distinguish the phases they are supposed to address.

The fix falls straight out of §3. Restrict the perturbed set to a band `B ⊆ V` and divide by `|B|`
instead of `|V|`:

```
J_B = mean_{p∈B} ∂/∂h_l[p] Σ_{p'∈V} h_tgt[p', i]
```

Same estimator, conditioned on the band, still exact within it. Cost scales with the **number of
bands**, not the number of positions — 3 phases is 3× the probes, not 112×.

`--position-groups`:
- `paper` (default) — one band, `B = V`. Reproduces jlens exactly.
- `thirds` — positional split, a cheap proxy for phase structure.
- `labels:<file.json>` — explicit per-token phase labels, the real phase-edge mode.

---

## 6. Gemma-4 specifics the fitter has to respect

- **`√hidden_dim` embedding scale** (`gemma4.rs:727`). Activations are not in pre-`lm_head` residual
  space, so absolute ε is meaningless across sites and models. ε is always specified **relative**:
  `ε = c·‖h_l‖/√d`.
- **`final_logit_softcapping = 30.0`** (`gemma4.rs:530`, applied at `781-786`). Monotone, so top-k
  *rankings* are untouched; only logit *values* differ. `--raw-logits` bypasses it when comparing
  numbers against the Python reference.
- **Tied embeddings.** `project_hidden_to_logits` already falls back to `tok_embeddings.t()` when
  there is no separate output tensor. Nothing extra needed.
- **`unembed` is missing.** `project_to_logits` deliberately skips the final norm (it expects an
  already-normed hidden). jlens's `unembed` is norm + head, so each model file gains a 4-line
  `unembed()`. Without it the RMSNorm nonlinearity would get folded into `J` and the lens would
  stop matching the reference.
- **SWA window 1024** (`gemma4.rs:476`) vs 128-token fit prompts: during fitting every local layer
  sees the whole sequence and behaves globally. A lens fitted at 128 tokens does not describe
  long-context behaviour. This is a generalisation boundary for the keys, not a footnote.
- **Dense only.** `gemma4.rs` has no expert/MoE handling, so `Gemma4-26B-A4B` and
  `diffusiongemma-26B-A4B` will not load through it. Routing would also make the perturbation
  non-smooth — a probe can flip an expert — so MoE needs a routing-flip counter before any number
  from it is quotable.

---

## 7. Why hydro becomes a workspace

The sidecar must load models through hydro's loader, not a copy of it — the point is to measure the
weights hydro runs, and two loaders drift. But hydro is binary-only: no `lib.rs`, the loader is
inline in `main.rs`, `Model` is private. `src/bin/field_audit.rs` already shows the failure mode,
duplicating field math as a "CPU audit copy".

The extraction is cheap because the module graph is clean: `config.rs` has zero `crate::` refs,
`dim_assert.rs` touches only `config`, `hooks.rs` only `dim_assert`, and the three model files only
`hooks`. Nothing pulls in `niodoo`/`field`/`memory`. So `lib.rs` exports seven modules, `main.rs`
switches those `mod` declarations to `use`, and the other 18 modules stay private to the binary.

---

## 8. What would falsify this

Gates in `PLAN.md`, ordered by information content. The two that can kill the approach:

- **Gate 1 (identity).** Fit with source == target; `J` must return ≈ `I`. If it doesn't, the
  position mask, the sign, or the batch aliasing is wrong, and nothing downstream means anything.
- **Gate 2 (ε plateau).** `J v` must be flat across the middle decades of ε. Too small and Q4
  quantisation noise dominates; too large and the linearisation fails. **If there is no plateau,
  the forward-mode estimator is not trustworthy at this quantisation and that is a negative result
  to report, not an ε to tune until the output looks nice.**

Gate 3 (cross-check against Python reverse-mode on the unquantised 0.8B) is the only test that
validates the identity in §3 against the algorithm it claims to reproduce.

Nothing from this crate gets published or claimed until gates 1-3 pass.
