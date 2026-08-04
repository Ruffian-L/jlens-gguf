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

---

## 2026-08-02 (late) — both earlier headline results were the BOS bug

`research-logs/2026-08-02_gemma4_missing_bos.md` invalidated the inputs to every
chat-templated run. Re-ran the two that mattered.

### Subject gate: 0.50 → 0.70

Same pre-registered protocol, same null, prompts the model can actually read:

| depth | L24 | L36 | L44 |
|---|---:|---:|---:|
| N=0 | **0.7042** | 0.6341 | 0.6377 |
| N=2 | 0.6138 | 0.6082 | 0.5936 |
| N=8 | 0.5775 | 0.5995 | 0.5795 |

Real signal, clearly above chance. **Still recorded as FAIL** — the pre-registered bar is
0.80 and it is not moving after the fact.

The shape is the interesting part: subject is sharpest at the **first** thought token and
decays with depth. We had assumed the opposite — that a thought needs to unfold before it
commits to a topic. It appears to commit immediately and then diffuse.

### Stance claim retracted

The unsupervised clusters were called "stances" on the strength of their exemplars. NMI says
otherwise — form 0.31–0.36, subject 0.33–0.36, and adjusted for what k=8 can achieve
(subject caps at 0.72) subject explains *more*. One cluster was five Silk Road prompts, i.e.
pure subject, and it was glossed over because only the first three members of each cluster
were printed.

Structure is real (all four metrics beat a shuffled null). What it is organised by is a
mixture of topic and opening move, neither dominant. The clean version was wrong.

---

## 2026-08-03 — replicated on Gemma 3. Stance is much stronger there.

Second architecture, same protocols. `gemma-3-4b-it-Q4_K_M`, 34 layers, d=2560, no thought
channel — it answers directly after `<start_of_turn>model`.

### Two more tokenizer traps, both fixed

- **Trailing `<eos>`.** Gemma 3's post-processor is `[<bos>, A, <eos>]`, so every encoded
  prompt *ends* with an end token. Reading "the last prompt position" read the disposition
  at `<eos>`. `hydrodynamic-swarm` strips this (`encode_prompt_no_trailing_eos`); the
  sidecar did not. Now routed through `tokens::encode_prompt`.
- **Wrong tokenizer silently loaded.** The loader prefers a tokenizer sitting next to the
  model, and a stray Llama 3.1 `tokenizer.json` in the models directory won — position 0 came
  back as `<|begin_of_text|>` (128000). Gemma 4 has a guard against this; Gemma 3 does not.
  Pass `--tokenizer` explicitly.

Also: the anchor rule now treats "no thought channel in the template" as *stream already
open*, instead of falling through to a step-8 heuristic written for Gemma 4.

### Structure: replicates

| layer | eff. dim (real/null) | silhouette (real/null) | continuation |
|---|---|---|---|
| L20 | 4.75 / 9.26 | 0.522 / -0.031 | 1.61× |
| L28 | 3.85 / 13.15 | 0.578 / -0.040 | 1.54× |

### What the clusters are: decisively **form**, not subject

| model | layer | form | subject |
|---|---|---:|---:|
| Gemma 4 12B | L36 | 0.360 | 0.345 |
| **Gemma 3 4B** | L12 | 0.476 | 0.177 |
| **Gemma 3 4B** | **L20** | **0.888** | 0.172 |

NMI 0.888 — the unsupervised clustering essentially recovered the eight question forms
without being told they exist. Exemplars are unambiguous: photosynthesis, the Silk Road and
tidal power cluster together because all three were asked for *a short overview*.

```
cluster 7   "here's a short overview of photosynthesis:"
            "here's a short overview of the Silk Road:"
            "here's a short overview of tidal power:"
cluster 4   "a beginner to ecology?" / "a beginner's to design" / "a beginner farmer"
```

### Subject gate: replicates as a near-miss

Peak **AUC 0.718** (N=2, L20) against Gemma 4's 0.70. Real signal, still under the
pre-registered 0.80 bar, still recorded as FAIL.

### Two caveats we'd want a reader to have

- Gemma 3 4B has **very formulaic openings** ("Okay, let's dive into…"). That plausibly
  inflates form NMI relative to a larger or less templated model. The comparison to Gemma 4
  is confounded by size as well as by architecture.
- The one clear architectural difference is that Gemma 4 has a thought channel and Gemma 3
  does not. Gemma 4's form signal is much weaker (0.36 vs 0.89). A thought block diffusing
  the opening stance is a plausible explanation and an untested one.

---

## 2026-08-03 — stance survives disjoint wording. Gate PASS.

A reviewer raised the sharpest objection so far: the cluster exemplars share literal strings
("at its simplest level", "teach X to a beginner"), so the form/stance signal might be
**string overlap** — residuals sitting close because the same words are about to be emitted,
not because a stance is encoded. If true, the headline collapses into "similar-sounding
questions produce similar-sounding answers", which is boring and true.

One correction to the objection: capture is at depth 0, so those shared strings are in the
model's *future*, not its past — there is almost no prefix at capture time. That doesn't
rescue the claim, it relocates it: at depth 0 the model has committed to how it will open,
and "committed to a stance" vs "committed to a string" are hard to tell apart.

### The test

Eight stances, each written **twice with deliberately disjoint wording**:

```
overview   "Give me a short overview of X."      | "In brief, what should I know about X?"
teaching   "How would you teach X to a beginner?"| "If someone knew nothing about X, where
                                                   would you start?"
```

Phrasing A covers subjects 0–5, phrasing B covers subjects 6–11, so a positive pair is
**same stance, different wording, different subject**. Plus a ninth group of arithmetic
("What is 7 + 5?") where the model has no stance latitude at all.

### Result: PASS

Restricted to pure cross-wording positives (`--pair-by stance --differ-on variant`):

| layer | n_pos | n_null | AUC |
|---|---:|---:|---:|
| L12 | 288 | 5184 | 0.7719 |
| **L20** | 288 | 5184 | **0.8553** ✅ |
| **L28** | 288 | 5184 | **0.8753** ✅ |

Above the pre-registered 0.80 bar, and **higher than the mixed-wording version** (0.8226) —
the opposite of what string overlap predicts. NMI confirms it:

| layer | stance | variant (wording) | subject |
|---|---:|---:|---:|
| L20 | 0.849 | 0.320 | 0.224 |
| L28 | **0.864** | 0.339 | 0.228 |

Stance 0.86, wording 0.34, subject 0.23.

The decode-time gate on the same corpus also passes: **N\* = 0, L20, AUC 0.8226**, decaying
to 0.58 by eight content tokens in — the signal is at the opening and fades, which is the
same shape seen everywhere else here.

### What this does and does not establish

It establishes that the geometry at the first generated token groups by *what kind of answer
is coming*, across disjoint phrasing and disjoint subjects, on a pre-registered threshold
written before any of these runs.

It does not establish that "stance" is a natural kind rather than a convenient name for
"the class of answer about to be produced". Operationally those are the same thing for
addressing memory, which is what this is for — but they are not the same claim.

Still owed: bootstrap CIs (a pass deserves the same rigour a fail got), a k sweep, and a
non-Gemma model. Both models so far are Gemma.

---

## 2026-08-03 — Llama 3.1 FAILS the stance gate. It is a Gemma 3 result.

Third model, first non-Gemma. `Meta-Llama-3.1-8B-Instruct-Q4_K_M`, 32 layers, d=4096.
Verified answering correctly before measuring ("The official currency of Italy is the Euro").
Same corpus construction, same pre-registered 0.80 bar.

### Cross-wording AUC: FAIL at every layer

| layer | AUC |
|---|---:|
| L12 | 0.7309 |
| L20 | 0.7383 |
| L28 | 0.7148 |

### And the NMI ordering inverts

| layer | stance | variant | subject |
|---|---:|---:|---:|
| L12 | 0.471 | 0.283 | 0.458 |
| L20 | 0.465 | 0.276 | **0.506** |
| L28 | 0.435 | 0.288 | **0.506** |

Adjusted for achievable ceilings (k=9 clusters; stance 9 values caps near 1.0, subject 13
values caps at 0.923): stance ~47%, subject ~55%. **Subject slightly wins.** The opposite of
Gemma 3, where it was stance 0.86 against subject 0.23.

### Where that leaves the headline

Across three models:

| model | cross-wording AUC | stance | subject |
|---|---|---:|---:|
| Gemma 3 4B | **0.875 PASS** | 0.864 | 0.228 |
| Gemma 4 12B | not run | 0.360 | 0.345 |
| Llama 3.1 8B | **0.738 FAIL** | 0.465 | 0.506 |

**The stance result does not generalise.** It is a Gemma 3 finding, and the strength of it
tracks something about the model rather than about language models. The caveat filed earlier
— that Gemma 3 4B has unusually formulaic openings ("Okay, let's dive into…") — now looks
load-bearing rather than decorative: templated openings plausibly *are* why stance dominates
its first-token geometry.

What survives all three models: **the geometry has structure** (every metric beats a
per-dimension shuffled null on every model), and **something is encoded at the first
generated token that predicts the continuation** (2.8–3.1× on Llama, the highest yet). What
that something *is* varies by model.

### Bug found and fixed

`cmd_structure` and the decode gate hardcoded Gemma's EOS ids (`1, 106, 50`). Llama's turn
ends at `<|eot_id|>` / `<|end_of_text|>`, so decodes ran past the end of the turn. EOS is now
resolved from the tokenizer. This does **not** change the numbers above — capture is at
depth 0, long before any EOS — but it corrupted the continuation strings.

---

## 2026-08-03 (later) — four models, same corpus. It is Gemma 3, and it is not size.

Ran the disjoint-wording stance test on every model that loads, all on the **same corpus**
(the earlier Gemma 4 number was from a different corpus and was not comparable).

| model | family | size | best cross-wording AUC | stance | variant | subject |
|---|---|---|---|---:|---:|---:|
| Gemma 3 4B | gemma3 | 4B | **0.8753** ✅ | 0.864 | 0.339 | 0.228 |
| Gemma 3 27B | gemma3 | 27B | **0.8926** ✅ | 0.811 | 0.293 | 0.238 |
| Gemma 4 12B | gemma4 | 12B | 0.6963 ❌ | 0.532 | 0.272 | 0.435 |
| Llama 3.1 8B | llama | 8B | 0.7383 ❌ | 0.465 | 0.276 | 0.506 |

### The size hypothesis is dead

Yesterday's explanation was that Gemma 3 4B's unusually formulaic openings ("Okay, let's
dive into…") were why stance dominated, and that a bigger, less templated model would show
less of it. **Gemma 3 27B is 7× larger and scores higher** (0.8926 vs 0.8753), with nearly
identical NMI (stance 0.81 / subject 0.24 against 0.86 / 0.23). The effect is stable across
a 7× size range within the family. Size is ruled out; so is templated-ness.

That is the second explanation of mine falsified by a run in two days. The first was
predicting the Gemma 4 clusters would be question-form.

### What the split actually tracks

Gemma 3 passes at both sizes. Gemma 4 and Llama 3.1 both fail, and in both the stance and
subject NMIs are close together rather than separated.

**The Llama comparison is clean.** Gemma 3 and Llama 3.1 both answer directly after the
turn header, so "the first generated token" is the same functional position in both. Gemma 3
separates stance from subject 0.86 vs 0.23; Llama does not, 0.47 vs 0.51.

**The Gemma 4 comparison is confounded and should not be read as a family result.** Gemma 4
opens a thought channel, so its depth-0 token sits *inside a thought block*, not at the start
of the answer. That is a different functional position, and the gap may be about where we
are reading rather than about the model. Testing that means capturing Gemma 4 at the first
token of its post-thought answer, which we have not done.

### Where this leaves it

A real, size-stable property of Gemma 3: the residual at the first generated token separates
*what kind of answer is coming* from *what it is about*, across disjoint phrasing. Llama 3.1
does not do this. We do not know why, and the honest scope is one model family.

Structure and continuation-prediction still hold on all four (continuation ratio 2.8–3.5×,
highest on the 27B at 3.49×).

### Models that could not be tested

`diffusiongemma-26B-A4B` — `general.architecture = diffusion-gemma`, 128 experts / 8 used.
No MoE support in the loader, and arch sniffing would route it to the Gemma 3 loader and
produce garbage. Conceptually it also has no "first generated token": a diffusion LM denoises
the whole sequence, so the framing needs rethinking before the measurement means anything.

### Loader footgun, hit twice

Tokenizer fallbacks in `loader.rs` are **relative paths**, so from any cwd but the repo root
they miss and a stray Llama `tokenizer.json` sitting next to the models wins the search.
Symptom the second time: 0 usable prompts, because `<bos>` and `<|turn>` tokenised as
garbage. Always pass `--tokenizer` with an absolute path.

---

## 2026-08-03 (later still) — CIs, and two of my own caveats retracted

### The Gemma 4 confound was imaginary

Yesterday I flagged Gemma 4's failure as confounded, on the grounds that its depth-0 token
sits inside a thought block rather than at the start of an answer. **It does not.** Hydro
appends the *thinking-off* generation prompt — `<|channel>thought\n<channel|>`, an empty
thought block already closed — so the model answers immediately:

```
"**Photosynthesis** is the biological process by which green plants, algae"
```

All four models were read at the same functional position: the first token of the answer.
Gemma 4's failure is genuine, which makes the result sharper — Gemma 3's own successor,
same lab, one generation later, lost the property.

### Bootstrap CIs (400 resamples over prompts, 95%)

Resampling over **prompts**, not pairs: pairs share prompts and are not independent, so a
pair-level bootstrap would report an interval far too narrow.

| model | best AUC | 95% CI | verdict |
|---|---:|---|---|
| Gemma 3 4B (L28) | 0.8753 | [0.828, 0.916] | **clear pass** |
| Gemma 3 27B (L37) | 0.8926 | [0.828, 0.946] | **clear pass** |
| Gemma 4 12B (L28) | 0.6963 | [0.619, 0.787] | **clear fail** |
| Llama 3.1 8B (L20) | 0.7383 | [0.657, 0.820] | **inconclusive** |

**Llama 3.1 is not a fail. It is underpowered.** Its interval spans the 0.80 bar, so at
n=108 we cannot distinguish it from passing. Calling it a failure — as this log did earlier
today — was reading a point estimate as a verdict. Retracted.

The clean, statistically supported contrast is **Gemma 3 vs Gemma 4**: both intervals are
entirely on their respective sides of the bar, and they point opposite ways.

### Where this actually leaves it

Supported: Gemma 3 (4B and 27B) separates *the kind of answer coming* from *what it is
about* at the first generated token, across disjoint phrasing, stable over a 7× size range,
with the CI clear of the pre-registered bar. Gemma 4 does not, with the CI clear of the bar
in the other direction.

Not supported either way: Llama 3.1, and therefore the whole question of whether this is a
Google-model property, a Gemma-3-specific one, or something broader. Settling Llama needs a
larger corpus, not a new model.

Unchanged and holding on all four: structure against a shuffled null, and first-token state
predicting the continuation (2.8–3.5×).
