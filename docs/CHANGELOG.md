# `jlens-gguf` — changelog

What actually changed, and where it diverged from `PLAN.md`. Newest first.
Design reasoning lives in `DESIGN.md`; `PLAN.md` is frozen as approved.

Conventions: **DEVIATION** marks a departure from the approved plan.
**FINDING** marks something discovered in the existing tree that changed the work.

---

## 2026-08-02 — Subject-gate FAIL accepted; stance gate is next (protocol only)

**Decision (Jason + multi-view synthesis):** decode-time **subject** clustering FAIL
(AUC ≈ 0.5) is **good news** under the architecture — not a broken instrument.

- Architecture: same subject, different first thoughts → **different basins**.  
- Gate had asked: do first-thought keys stick to **subject**? Answer: no.  
- Openings are stance/phrasing-shaped (“The most common…” / “If you are…” / “To provide the…”).

**Locked:** no fitter, no dequant, no commit redesign to chase subject AUC; header prefill +
content-token N + position-matched baseline stay permanent.

**Next brick:** `STANCE_GATE.md` (thresholds + corpus + null). **No implement until PI says go.**

HF weights available for later interop (not default gate body):  
`/home/ruffianl/models/gemma-4-12B-it`, `/home/ruffianl/models/gemma-4-E2B-it`.

---

## 2026-08-02 — GATE 2 PASSES. The blocker was an environment variable.

`candle-core-0.9.2/src/quantized/mod.rs:700-737`. `QMatMul::from_arc` consults a
thread-local read from `CANDLE_DEQUANTIZE_ALL`; when set, every tensor is dequantised into
`Self::Tensor` — a plain f32 matmul with **no activation quantisation**. The step function
that made finite differences impossible is gone.

**Zero code changes.** One env var on the existing Q4 file.

### Layer 46 → 47 (one block)

| eps_rel | ‖J v‖ | ‖Σ Δh‖ | cos(prev) |
|--------:|------:|-------:|----------:|
| 1e-4 | 1.1751 | 0.029 | — |
| 1e-3 | 1.1734 | 0.285 | 0.99989 |
| 1e-2 | 1.1733 | 2.848 | 1.00000 |
| 1e-1 | 1.1733 | 28.476 | 1.00000 |

`‖Σ Δh‖` is now **exactly linear in ε** — 10× per decade — which is the definition of the
thing we were trying to measure and the exact opposite of the flat floor recorded below.

### Layer 20 → 47 (27 blocks)

| eps_rel | ‖J v‖ | ‖Σ Δh‖ | cos(prev) |
|--------:|------:|-------:|----------:|
| 1e-4 | 1.1847 | 0.081 | — |
| 1e-3 | 1.1844 | 0.812 | 0.99980 |
| 1e-2 | 1.1765 | 8.067 | 0.99981 |
| 1e-1 | 1.2597 | 86.372 | 0.86173 |

Plateau 1e-4 … 1e-2, then cos falls to 0.86 at 1e-1 — the linearisation breaking down at
large ε, precisely where finite-difference theory says it should. Rounding floor on the
left, truncation on the right, plateau between: the textbook U-curve. **Fit in 1e-3 … 1e-2.**

### Scope — what this Jacobian is of

The dequantised model has the **same weight values** the Q4 kernel uses; dequantisation is
exact in that direction. What disappears is the rounding of *activations* at each matmul.
So this is the Jacobian of "the model whose weights are the Q4 grid, run in f32" — a real
and useful object, and the right one for a lens.

It is **not** the Jacobian of the deployed quantised forward pass. That function is still
piecewise-constant and still has no derivative (see the entry below). Anything measuring the
*deployed* path — steering, `src/jacobian.rs` — is unaffected by this and its "proxy"
labelling stands.

### Second route, not needed but recorded

`from_arc:725` dequantises `F32 | F16 | BF16` GGUF tensors **automatically**, no env var. So
a BF16 GGUF converted from the safetensors checkpoints (E2B 9.6G, 12B 23G, downloaded
2026-08-02) would take the same path while keeping every bit of hydro's existing GGUF
steering. Swinging to safetensors is not required.

### Cost

Dequantised 12B in f32 is ~48 GB against 128 GB available. Loads and runs.

---

## 2026-08-02 — Decode-time gate also FAILS — and the gate may have tested the wrong hypothesis

Moved the commit inside the generated thought stream as agreed, swept N ∈ {2,4,8,16,32}
content tokens deep × 3 layers, same pre-registered thresholds, same null. **FAIL at every
depth and every layer**, AUC 0.47–0.53, dense cosine 0.48–0.53.

Determinism: greedy decode twice → identical. Anchor: mean step 0.0 over 58/60 prompts.

### The generation is coherent this time — that took two fixes

First decode attempt produced `<|channel>.\n.\n.\n.\n.\n` for every prompt: repetition
thrash, no thought at all, and the gate numbers from it were meaningless. Asked to open its
own thought block under greedy decode, Gemma 4 collapses. This is a **known** property of
this checkpoint in this repo — `main.rs:287` pre-fills `<|channel>thought\n<channel|>`
rather than letting the model emit it, and `gemma4_hyphen_thrash` exists to detect the same
repetition family during normal generation.

With the header pre-filled the thought is real:

```
antibiotic resistance[0] -> "The most common"
antibiotic resistance[1] -> "If you are"
antibiotic resistance[2] -> "To provide the"
```

Also fixed: `N` counts **content tokens** after the channel header closes, not raw decode
steps, so every sample sits at the same functional depth (Gemini's note — header length
varies with how the prompt ends).

### The finding hiding in that sample

Those three openings are three **paraphrases of the same subject**, and the model opens each
one differently. The thought's opening tracks the *phrasing* of the question — what kind of
question is this, how should I structure a reply — not the topic.

Which means the gate tested a hypothesis the architecture never actually claimed. The gate
asked: *do first-thought keys cluster by subject?* Answer, robustly: no. But the design note
this work came from says, in its own words:

> Same subject, different first thoughts → different basins.

So different first thoughts for the same subject is the **expected** behaviour, not a
failure of the key. What failed is the assumption underneath the recall query "show me first
thoughts on subject S" — that first-thought keys would be subject-shaped. They are
disposition-shaped, and disposition co-varies with phrasing.

**Before redesigning the commit rule, the hypothesis needs redesigning.** A gate for
opening-*stance* clustering needs stance labels, not subject labels: does "The most
common…" key closer to another enumerating opening than to "If you are…"? That is a
different corpus and a different null, and it is testable with the machinery now built.

### Robustness of the negative

Subject-clustering has now failed under: prefill hinge (5 layers × 6 offsets × 2 baseline
modes), decode-time (3 layers × 5 depths), sparse `dim_signature` and dense cosine, with
determinism confirmed at every stage and generation verified coherent.

### Not done, as agreed

No fitter. No dequantisation. Reported and stopped.

---

## 2026-08-02 — STABILITY GATE FAILS, and the reason is the model's format

Pre-registered protocol and thresholds: `STABILITY_GATE.md`, written before the run.
**Verdict: FAIL.** AUC ≈ 0.50 everywhere. Determinism passed exactly (1.000000).

### The result

12 subjects × 5 paraphrases, 120 positive pairs against 1650 null pairs, one-shot arm:

| layer | med(pos) | med(null) | AUC |
|-------|---------:|----------:|----:|
| L24 | 0.2580 | 0.2582 | 0.4957 |
| L28 | 0.1787 | 0.1807 | 0.5277 |
| L32 | 0.1343 | 0.1267 | 0.5237 |
| L36 | 0.1377 | 0.1394 | 0.5010 |
| L40 | 0.1796 | 0.1817 | 0.4890 |

Failure mode #1 from the pre-registered list: the key carries no subject information at all.
Held across **5 layers × 6 commit offsets × 2 baseline modes**, and for both the sparse
`dim_signature` and a dense full-vector cosine. Not a compression problem — the signal is
not in the residual to begin with.

### The 3-prompt preview was noise

The preview reported margins of +0.177 to +0.291 and singled out L36. Against a proper null
distribution L36 scores **0.5010**. Three prompts produced a confident, entirely spurious
signal. This is the whole reason the gate requires a null population, and it would have sent
us into the fitter chasing nothing.

### Two real bugs found while diagnosing

**The baseline must be position-matched.** It averaged over all valid positions while the
readout is at the last position, whose distribution is completely different. Subtracting the
wrong mean left a large common-mode: cosine between two *unrelated* centred commit residuals
was 0.74. With `--commit-only` it drops to 0.35. Fixed; AUC unchanged, so this was a real
bug that was not the cause.

**Centring before the unembed destroys the readout.** On forced-completion prompts whose
paraphrases must predict the same word, centred top-8 token sets shared **nothing** — median
Jaccard 0.0000. `μ` carries the component driving the shared prediction, so `h - μ` removes
the answer. Now: `text_bridge` unembeds the **raw** residual (and reproduces
`model.forward()` exactly), `dim_signature` ranks the z-score. The earlier claim in
`baseline.rs` that centring was "safe to unembed" was wrong and is corrected.

### Root cause: Gemma 4 is a thinking model, and we were reading the wrong moment

Positive control, templated, one-word-answer prompts, reading the model's own final-layer
prediction at the hinge:

```
"What is the capital of Japan?"          -> '<|channel>' '_' '.' ':'
"What is the chemical symbol for gold?"  -> '<|channel>' '_' ':'
"What colour is fresh snow?"             -> '<|channel>' '---' '}' '<' ':'
```

The model is not confused. It is doing exactly the right thing: **it wants to open a thought
block.** At the prefill hinge, a Gemma 4 IT model has committed to nothing except "I should
start thinking". Its disposition there is about *format*, not *subject* — which is why the
text bridge read `hello / Welcome / 👋` and why no layer and no position separates subjects.

Raw completion prompts, tried as an alternative, are off-distribution for a thinking model
and collapse to `'1' '.' '0' '-'` for every input, including "The capital city of Japan is".
That is not a broken loader — tokenizers verified identical
(`md5 72b1044584e75adc53dd4372e903925c`), and `--verify` confirms the readout reproduces
`model.forward()` bit-for-bit.

### What this means for first-thought memory

**The commit rule needs revising.** "Residual at the last prompt token, pre-first-generated"
is the wrong hinge for a thinking model — at that point the thought has not opened yet, it
is only about to. The subject-bearing disposition lives **inside the generated thought
stream**, a few tokens in.

Prefill-only telemetry cannot reach it. Capturing first-thought on Gemma 4 requires
**decode-time capture**: generate N tokens into the thought block and read the residual
there. That is the next brick, and it is a change to the sidecar (capture during decode, not
just prefill), not a change to the theory.

The inversion still stands — the opening disposition may well be the real memory of a
conversation. We have now established, with a null distribution, that it is not visible at
the prefill hinge on this model.

### Not done, deliberately

No fitter. No dequantisation. Grok's ordering holds: diagnose before touching the fitter.

---

## 2026-08-02 — Step 1 shipped: no-fit logit-lens telemetry, three doors, verified

`jlens-gguf readout` emits JSONL disposition snapshots with no fit and no differencing, so
it is unaffected by the quantisation obstruction below. `jlens-gguf baseline` collects the
statistics it needs. 18 unit tests pass.

### The pipeline is provably correct

`--verify` runs the model's own `forward()` alongside the readout. Unembedding the last
block's output reproduces the real logits **exactly** — `"1"(19.673693)` from the readout
against `"1"(19.67)` from `forward()`, same five tokens in the same order. Any future change
that breaks capture or `unembed` will show up here immediately.

### FINDING — raw-residual keys do not discriminate at all

First batch keyed on the top-magnitude dimensions of the raw residual. Across two
paraphrases and one unrelated prompt:

| layer | paraphrase | unrelated | margin |
|-------|-----------:|----------:|-------:|
| L24 | 0.836 | 0.854 | **-0.018** |
| L36 | 0.709 | 0.712 | **-0.003** |
| L44 | 0.670 | 0.655 | +0.016 |
| L47 | 0.783 | 0.759 | +0.023 |

The *unrelated* prompt scored higher at L24 and L36, and the text bridges were the same
multilingual soup for "currency of Italy" and "why is the sky blue".

Cause: a handful of residual dimensions carry enormous content-independent magnitude
(outlier / attention-sink dimensions). Ranking by `|h_i|` finds those every time, so the key
described the model's furniture rather than the thought.

Fix: `baseline.rs` — per-layer mean and σ over a corpus, then `dim_signature` ranks the
z-score `(h-μ)/σ` and `text_bridge` unembeds the centred `h-μ`. Centring only for the
bridge: dividing by σ would rotate the residual out of the basis the unembedding expects.

### FINDING — the Gemma 3 chat template is not Gemma 4's

The first corrected run still looked wrong. Position `-1` resolved to a token `'start'`:
`<start_of_turn>` was being tokenised as *literal text*. Gemma 4 uses
`<|turn>user\n…\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
(`main.rs:277-287`), not Gemma 3's markers. The readouts had been describing template
gibberish. `read_prompts` now unescapes `\n` so a line-based prompt file can hold a
template.

### After both fixes — the keys discriminate

Same three prompts, correct template, 180-prompt baseline, read at the first generated
position:

| layer | paraphrase | unrelated | margin |
|-------|-----------:|----------:|-------:|
| L24 | 0.410 | 0.233 | +0.177 |
| L36 | 0.524 | 0.233 | **+0.291** |
| L44 | 0.221 | 0.165 | +0.056 |
| L47 | 0.142 | 0.062 | +0.080 |

Positive at every layer, strongest mid-stack at L36. This is a 3-prompt preview, **not** the
stability gate — that needs many paraphrase sets and a proper null distribution.

### The two doors report different things

`dim_signature` separates by subject. `text_bridge` at the first generated position reads
`hello / Welcome / Greetings / 👋` for all three prompts — a correct readout of the
disposition actually present there, which is "how do I open a reply", not "what is this
about". Mid-stack the logit lens shows the classic concept-before-token signature
(`also / également / también / аналоги` — one concept in four languages), but it never
surfaces subject words like *euro* or *lira*. That is the documented weakness of the raw
logit lens on Gemma-family models, and it is exactly the gap the paper's fitted transport
fills.

**Consequence for the architecture:** until fitting is unblocked, the cross-model door
(`text_bridge`) is weak on this model family, while the within-model door (`dim_signature`)
works. Cross-AI basins are the thing most blocked by the quantisation finding.

---

## 2026-08-02 — GATE 2 FAILS: finite differences do not work against candle's quantised path

**This is a negative result and it invalidates the fitting half of the plan as written.** The
crate builds, the unit tests pass, Gate 1 passes, and the estimator's bookkeeping is correct.
The obstruction is in the model, not the code.

### What was measured

`gemma-4-12b-it-Q4_K_M`, source layer 20 → target layer 47, 111 valid positions, random probe
directions, central differences:

| eps_rel | eps_abs | ‖J v‖ | ‖Σ Δh‖ | cos(prev) |
|--------:|--------:|------:|-------:|----------:|
| 1e-8 | 3.2e-8 | 2937843 | 20.5 | — |
| 1e-6 | 3.2e-6 | 37824 | 26.5 | -0.13 |
| 1e-4 | 3.2e-4 | 386 | 27.0 | -0.25 |
| 1e-2 | 3.2e-2 | 6.8 | 47.5 | -0.15 |
| 1e-1 | 3.2e-1 | 1.3 | 90.0 | 0.53 |
| 1e0 | 3.2e0 | 0.63 | 440 | 0.49 |

`‖Σ Δh‖` — the raw summed response, before the estimator's `1/(2ε·|band|)` scaling — is
**flat across eight decades of ε**. A Jacobian requires the response to be proportional to
the perturbation. This one is independent of it. `‖J v‖ ∝ 1/ε` is that floor divided by ε.

### It is not any of the obvious things

- **Not run-to-run noise.** Sweeping the same ε three times gives bit-identical results
  (cos = 1.00000, identical norms). The pipeline is deterministic.
- **Not f32 cancellation.** Target residual RMS is 0.2, so the f32 accumulation floor is
  ~1e-5. The observed floor is 3–90, six orders of magnitude larger.
- **Not depth or chaotic amplification.** The floor is present at a **single block** (layer
  46 → 47): 2.75 at eps_abs 1e-4, 3.82 at 1e-3, 4.49 at 1e-2. One block cannot decorrelate
  a trajectory.
- **Not the estimator.** Gate 1 (source == target) returns `J = I` to 0.000000.

### Cause

`candle-core-0.9.2/src/quantized/k_quants.rs:2296`, inside `matmul`:

```rust
T::VecDotType::from_float(lhs, lhs_b_mut)
```

Every quantised matmul **quantises its input activations** to 8-bit blocks (Q8_0 / Q8_1 /
Q8K, 32 values per shared scale) before the dot product. `QMatMul::forward` routes there for
every `QTensor` weight, which is every projection in the model.

So the GGUF forward pass is piecewise-constant in its input at the scale of the activation
quantisation step. A perturbation below that step is rounded away entirely; what survives is
whichever blocks happened to land on the far side of a rounding boundary — deterministic,
fixed in magnitude, arbitrary in direction. That is exactly the observed floor, and it
explains the otherwise-backwards result that **large ε is better than small ε**: cos only
starts climbing once eps_abs (~0.1–0.3) approaches the quantisation step.

There is no ε that works. The window where the perturbation is both above the quantisation
step and small enough to stay linear does not exist for this model.

### What this does NOT invalidate

- The estimator identity in `DESIGN.md` §3 — Gate 1 confirms the bookkeeping.
- `probe.rs`, `basis.rs`, `lens.rs`, `keys.rs`, the CLI, the workspace split, `unembed`.
- The **apply** path, which is exact arithmetic (transport + unembed) with no differencing.

The blocked part is precisely one thing: obtaining `J` by finite differences against
quantised weights.

### FINDING — Gate 3's model does not load

`Qwen3.5-0.8B-BF16.gguf` was the plan's cross-check against Python reverse-mode, because it
is the only unquantised GGUF on the box. It uses `qwen35.*` metadata keys; `llama.rs`
hardcodes `llama.*` and bails with `cannot find llama.attention.head_count in metadata`.
Gate 3 is blocked on a loader gap, independently of the above.

---

## 2026-08-02 — plan approved, first edits

### FINDING — the existing FD probe was returning zeros

`src/jacobian.rs:243-275` built its perturbation as `Tensor::zeros(...).add(&scalar)`, which
broadcasts ε across **every** dimension rather than writing it at `dims_to_measure[dim_idx]`. The
indexed reads `pos_val` / `neg_val` (lines 246, 252) were computed and discarded — the intended
write was dropped at some point. Compounding it, `neg_hidden = hidden.sub(&neg_perturbation)` with
`neg_perturbation` all `-ε` evaluates to `hidden + ε`, identical to `pos_hidden`, so
`pos_logits - neg_logits` was exactly zero and every entry of the sensitivity matrix was 0.

Live path: `main.rs:1154` → `measure_jacobian_step` (`main.rs:3880`) → `lens.measure`.

Consequence: **any `DimSignature` / `JacobianKey` / `MultiKeyAddress` produced before this fix is
degenerate** — the top-dims ranking was over an all-zero matrix, so it was arbitrary tie-breaking,
not a measurement. Any run log or research note quoting those keys needs re-reading with that in
mind.

Fixed: one-hot delta of magnitude ε at `dim`, `broadcast_add` / `broadcast_sub` for a real central
difference. `DType` dropped from the imports (it was only used by the two deleted `zeros` calls).

### FINDING — no autograd through GGUF

`QMatMul::forward` is `xs.apply_op1_no_bwd(...)` (`candle-core-0.9.2/src/quantized/mod.rs:861`).
jlens's reverse-mode estimator cannot be ported literally. Drove the entire forward-mode design in
`DESIGN.md` §3.

### FINDING — stale comment on the gemma4 hook surface

`gemma4.rs:709-713` documents only `PreLayer` / `PostMlp` / `FinalNorm` as hookable. `PostAttn` is
in fact live at `gemma4.rs:370` inside `Layer::forward`, which receives the hook. All four
`HookSite` variants work on gemma 4. Comment to be corrected; no code change needed.

### DEVIATION — phase bands added before implementation

Raised by Jason mid-plan: hooks that don't cover the phase-edge positions give incomplete keys even
when the smoke test passes. Site coverage turned out to be fine (see above), but the underlying
point held on the **position** axis — jlens averages source positions away (`fitting.py:198`), so a
literal port collapses first-thought / revise / settle into one transport.

Added `--position-groups paper|thirds|labels:<file.json>` as a first-class feature rather than a
follow-up. Restricting the perturbed set to a band and dividing by `|B|` keeps the estimator exact
within the band, at cost × n_bands. `paper` remains the default and reproduces jlens exactly.

### Model target

`gemma-4-12b-it-Q4_K_M.gguf` (dense) confirmed as first target. `gemma4.rs` has no expert/MoE
handling at all, so the A4B checkpoints in `~/models` cannot load through it regardless of the
routing-smoothness concern.
