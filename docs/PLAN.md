# `jlens-gguf` — a Jacobian-lens sidecar for GGUF models

## North star (why this exists)

See **`research_logs/2026-08-02_first_thought_multi_address_memory.md`**.

One cold log, **multi-address** memory: classic `(source, source_key)` + semantic embedding +
**J-key / first-thought disposition** (+ later hidden-state handle). Cluster/dream on **how the
thought opened**, not only the closing speech. Team AIs are **source filters**, not silos.
Private CLI / durable keys first. This sidecar is the GGUF instrument that makes J-keys real on the
**same quantised weights** hydro runs.

## Context

`research_logs/2026-08-02_jacobian_lens_repo_vs_hydro_fd.md` names two different objects that keep
getting conflated: the real Jacobian lens (`~/jacobian-lens`, package `jlens`, Python/PyTorch/HF —
fitted average transport `J_l = E[∂h_final/∂h_l]`, read out as `unembed(J_l h)`), and hydro's
finite-difference probe (`src/jacobian.rs` — local `∂logits/∂h` at one decode step). The log's
"next experiment (corrected)" is option (B): get a real jlens readout whose top tokens can hash into
the multi-key / text-bridge schema. This plan builds that as a Rust sidecar that runs against the
GGUF weights hydro already loads, so the lens reads the *same* model the swarm runs — not an fp16
HF sibling.

Two facts from the current tree shape everything below.

**1. The existing FD lane returns zeros.** `src/jacobian.rs:245-258` builds the perturbation as
`Tensor::zeros(...).add(&scalar)`, which broadcasts ε to every dimension instead of writing it at
`dim`; `pos_val`/`neg_val` (lines 246, 252) are computed and discarded — the intended indexed write
was dropped. Then `neg_hidden = hidden.sub(&neg_perturbation)` with `neg_perturbation` all `-ε`
gives `hidden + ε`, identical to `pos_hidden`. Every entry of `sensitivity` is exactly 0. It is
live: `main.rs:1154` → `measure_jacobian_step` (`main.rs:3880`) → `lens.measure`. Any
`DimSignature` / `JacobianKey` produced so far is degenerate. The sidecar does not build on it.

**2. There is no autograd through GGUF.** `QMatMul::forward` is `xs.apply_op1_no_bwd(...)`
(`candle-core-0.9.2/src/quantized/mod.rs:861`), so jlens's reverse-mode estimator cannot be ported
literally. Dequantising to f32 to get gradients defeats the point (a 12B Q4 becomes ~50 GB).

## The estimator

jlens fits (`jlens/fitting.py:177-202`): one-hot cotangent at output dim `i` placed at *every* valid
target position, backprop, mean of the gradient over valid source positions:

```
J_l[i,:] = mean_{p∈V}  ∂/∂h_l[p]  Σ_{p'∈V} h_tgt[p', i]
```

Forward mode gives the same object with no gradients. Perturb `h_l[p] += εv` at **every** `p ∈ V`
in one pass and sum the target-side change over `p' ∈ V`:

```
Σ_{p'∈V} Δh_tgt[p'] = ε · Σ_{p∈V} Σ_{p'∈V} J[p'←p] v + O(ε²)  =  ε·|V|·(J_l v) + O(ε²)

  ⟹   J_l v  =  ( Σ_{p'∈V} Δh_tgt[p'] ) / (ε·|V|)
```

Causality zeroes `J[p'←p]` for `p' < p`, so this is exactly jlens's "cotangents summed over target
positions, then averaged over source positions". Central differences (±ε) cancel the `O(ε²)` term.
Cost: **one batched prefill per probe direction** — batch element `b` carries direction `v_b`,
mirroring jlens's `dim_batch`. `v = e_j` recovers column `j` exactly; `--rank d_model` is the exact
fit.

**Per-phase transports.** Restricting the perturbed set to a band `B ⊆ V` and dividing by `|B|`
gives `J_B` — the same estimator conditioned on that band. This is what keeps the position
information that the paper's single `J_l` averages away, and it is the difference between keys that
distinguish first-thought / revise / settle and keys that don't. Cost scales with the *number of
bands*, not the number of positions. `--position-groups paper` (one band = `V`, the paper
estimator) is the default; `thirds` and `labels:<file.json>` (explicit per-token phase labels) are
the phase-edge modes.

`V` = `valid_position_mask` ported verbatim from `jlens/fitting.py:45-72`: skip the first 16
(attention sinks) and the last (no next-token target).

## Layout

Hydro is binary-only — no `lib.rs`, the loader is inline at `main.rs:2065-2149`, `Model` is private
at `main.rs:70`. `src/bin/field_audit.rs` works around this by duplicating math ("CPU audit copy").
Make it a workspace instead. The dependency graph is clean: `config.rs` has zero `crate::`
references, `dim_assert.rs` touches only `crate::config`, `hooks.rs` only `crate::dim_assert`, and
`llama.rs`/`gemma.rs`/`gemma4.rs` only `crate::hooks`. Nothing drags in `niodoo`/`field`/`memory`.

**In `hydrodynamic-swarm-3surface/`:**

- `Cargo.toml` — add `[workspace] members = ["jlens-gguf"]`.
- `src/lib.rs` (new) — `pub mod config; pub mod dim_assert; pub mod hooks; pub mod llama;
  pub mod gemma; pub mod gemma4; pub mod loader;`
- `src/loader.rs` (new) — move `Model` (`main.rs:70-116`) here and make it `pub`; extract
  `main.rs:2065-2149` into `pub fn load_gguf(model_path, tokenizer_override, device) ->
  Result<(Model, Tokenizer, String)>`, keeping the arch sniffing (`gguf_architecture`), the gemma3n
  bail, and the Gemma-4-must-not-use-Gemma-3-tokenizer rule at `main.rs:2120-2126` intact.
- `src/main.rs` — swap the `mod` declarations at lines 19/21/25/26/28/31 for
  `use hydrodynamic_swarm::{config, dim_assert, gemma, gemma4, hooks, llama, loader};` and call
  `loader::load_gguf`. Mechanical; the other 18 modules stay `mod` in the binary.
- `src/llama.rs`, `src/gemma.rs`, `src/gemma4.rs` — add `pub fn unembed(&self, h: &Tensor) ->
  Result<Tensor>` = final norm then the existing logits projection (~4 lines each). `norm` is
  private and `project_to_logits` deliberately skips it (it takes an already-normed hidden), so the
  lens needs this to match jlens's `unembed` = norm + head. gemma4's
  `project_hidden_to_logits` (`gemma4.rs:772`) already handles tied embeddings and the softcap.

**New crate `jlens-gguf/`** (deps: `hydrodynamic-swarm`, candle, `safetensors`, `tokenizers`,
`rayon`, `clap`, `serde_json` — all already in hydro's lockfile except `clap`):

- `src/probe.rs` — `ProbeHook: LayerHook`. `wants()` fires at the source site for layer `l` and at
  the target site; `apply()` adds `±ε·v_b` to batch element `b` at the source-band positions, and
  clones the target tensor out. This is the whole injection/capture mechanism — no changes to the
  forward passes beyond `unembed`.
- `src/basis.rs` — collect residuals at layer `l` over the corpus, PCA, top-`r` directions. Default
  probe basis for `--rank r`.
- `src/fit.rs` — port of `jlens/fitting.py`: `valid_position_mask`, `_check_layer_indices`, running
  mean over prompts, atomic resumable checkpoints, and the per-prompt diagnostics (`‖J‖/√d`,
  relative mean shift) that flag heavy-tailed prompts.
- `src/lens.rs` — port of `jlens/lens.py`: `Lens { j: HashMap<(layer, BandId), Tensor>, n_prompts,
  d_model, band_spec }`, plus `transport`, `apply`, `merge` (`n_prompts`-weighted, as
  `lens.py:106-133`). Storage is **safetensors**, not `.pt`.
- `src/keys.rs` — readout → `jacobian::DimSignature` / `JacobianKey` / `MultiKeyAddress`, reusing
  `text_bridge_hash`, `weighted_jaccard` and `cluster_signatures` from `src/jacobian.rs:612-737`.
  Honours the bridge rule from the research log: the pick carries text/token ids, never a raw
  `d_model` vector.
- `src/main.rs` — `fit` / `apply` / `basis` / `sweep` subcommands. HTTP `serve` lands after the
  numbers are trusted.
- `scripts/lens_pt_to_safetensors.py` — converts `~/jacobian-lens` checkpoints both directions, so
  a Python fit and a Rust fit are directly comparable.

Name the new type `Lens`, not `JacobianLens` — `jacobian::JacobianLens` (the FD proxy) keeps its
name and the two must not blur.

## First model

`models/gemma-4-12b-it-Q4_K_M.gguf` — dense. `gemma4.rs` has no `expert`/`moe` handling at all, so
`Gemma4-26B-A4B` and `diffusiongemma-26B-A4B` will not load through it; MoE routing would also make
the perturbation non-smooth (a probe can flip an expert), so dense is the right first target
regardless. Tokenizer via `data/google/gemma4_assets/tokenizer.json`.

Gemma-4 specifics the fitter must respect: the `√hidden_dim` embedding scale (`gemma4.rs:727`)
means activations aren't in pre-`lm_head` residual space — so ε is specified **relative**,
`ε = c·‖h_l‖/√d`, never absolute. The `final_logit_softcapping = 30.0`
(`gemma4.rs:530, 781-786`) is monotone, so top-k rankings are unaffected; `--raw-logits` bypasses it
when comparing logit *values* against the Python reference.

## Steps

0. (Small, separate) Fix `src/jacobian.rs:243-275` to write ε at `dims_to_measure[dim_idx]` instead
   of broadcasting. Independent of the sidecar, but the FD keys lane is dead until it lands.
1. Workspace + `lib.rs` + `loader.rs` extraction; `cargo build --release` still produces a working
   `hydrodynamic-swarm` binary.
2. `unembed` on the three model files.
3. `probe.rs` + a `--rank 1` smoke that proves injection and capture work batched.
4. `basis.rs`, then `fit.rs`, then `lens.rs`.
5. `keys.rs` bridge into `MultiKeyAddress`.
6. Phase bands (`--position-groups`).

## Verification

Gates, in order of how much they'd tell us:

1. **Identity check.** Fit with source == target layer. `J` must come back ≈ `I` (relative Frobenius
   error < 1e-2). Cheap, and it catches sign errors, off-by-one in the position mask, and batch
   aliasing in one shot.
2. **ε plateau.** `jlens-gguf sweep --layer 20 --eps 1e-4,1e-3,1e-2,1e-1` on a fixed prompt/direction.
   `J v` must be flat across the middle decades — too small and Q4 quantisation noise dominates, too
   large and the linearisation fails. **If there is no plateau, the estimator is not trustworthy on
   this quantisation and the result is negative — say so rather than picking the prettiest ε.**
3. **Cross-check against the reference.** `Qwen3.5-0.8B-BF16.gguf` is unquantised, so the same
   weights can be fitted in Python via `~/jacobian-lens` reverse-mode and in Rust forward-mode.
   Compare relative Frobenius error per layer and top-5 token agreement. This is the only test that
   validates the forward-mode identity against the algorithm it claims to reproduce.
4. **Readout smoke.** The paper's own prompt, `"Fact: The currency used in the country shaped like a
   boot is"`, at `positions=[-2]`: mid-layer top tokens should surface euro/lira *before* the final
   layer. Matches the README example so the output is directly comparable.
5. **Rank ablation.** `r ∈ {64, 128, 256, 512}` vs the exact fit on the 0.8B: report top-1 agreement
   per layer. This is what justifies the default rank rather than assuming 256 is enough.
6. **Key stability.** Same prompt twice → identical `DimSignature`. Paraphrase → high
   `weighted_jaccard`. Unrelated prompt → low. Without this the keys are noise.
7. **No regression.** `cargo test` plus one normal `run_swarm.sh` generation, confirming the
   workspace split didn't change decode behaviour.

Rough cost at seq_len=128, d_model≈3840, probe batch 8, central differences: `--rank 256` is
~64 prefills per (prompt, layer) — order tens of minutes for 8 layers × 100 prompts. The exact fit
is ~15× that per layer. Fit a strided subset of layers first.

## Known limits to state up front

- **Fit length vs SWA.** The window is 1024 (`gemma4.rs:476`) and fit prompts are 128 tokens, so
  during fitting every local layer sees the whole sequence and behaves like a global one. A lens
  fitted at 128 tokens does not describe long-context behaviour where the local layers are actually
  windowed. Fit length is a real generalisation boundary for the keys, not a detail.
- **Truncation error.** Forward-mode FD is exact only to `O(ε²)`; reverse-mode is exact. Gate 2
  measures this rather than assuming it away.
- **Sketch ≠ exact.** `--rank r < d_model` gives the transport restricted to the residual subspace
  the corpus actually occupies. Components outside it are dropped, not approximated.
- **MoE is out of scope** until `gemma4.rs` grows expert support, and would need a routing-flip
  counter before any number from it is quotable.
- Nothing here is published or claimed until gates 1-3 pass.
