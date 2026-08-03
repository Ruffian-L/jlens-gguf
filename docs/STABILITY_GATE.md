# Stability gate — protocol and thresholds

**Written before the run.** Thresholds declared in advance so the result cannot be graded
against whatever came out. If the gate fails, we diagnose; we do not move the bar.

---

## Question

Is a `dim_signature` key **re-hittable**? Two prompts that opened the same thought must key
closer together than two that didn't — reliably enough to build basins on.

Bit-repeatability is not the question and is already known (identical input → identical
output, cos = 1.00000). The question is **paraphrase robustness**: does the key survive a
rewording of the same subject while still rejecting a different subject?

---

## What is measured

**Commit position** (Grok's call, accepted): the residual at the **last prompt token**,
pre-first-generated-token, with Gemma 4's template
`<|turn>user\n…\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`.

First-thought ≠ first spoken word ≠ full answer. `key_captured_answer` is a *separate*
address, captured after the first assistant turn, and is **not** used here. Left null.

**Layers:** L36 primary (best margin in the 3-prompt preview). Sweep L24, L28, L32, L36, L40
so the choice of L36 is tested rather than assumed.

**Baseline:** per-layer μ/σ over the corpus, mandatory. Without it the key ranks the model's
constant outlier dimensions and does not discriminate at all.

---

## Populations

- **Positive pairs** — different paraphrases of the **same** subject. Same thought, opened
  in different words.
- **Null pairs** — paraphrases of **different** subjects. This is the distribution that
  matters: "0.4 similarity" means nothing without knowing what two unrelated prompts score.

Both scored with `weighted_jaccard` from `src/jacobian.rs:626`, the same function the picker
uses. Reusing it is the point — the gate must measure the metric that will actually index.

---

## Metric

**AUC** — the probability that a randomly chosen positive pair scores above a randomly
chosen null pair. Threshold-free, directly answers "can this rank a paraphrase above a
stranger", and unaffected by the absolute scale of the similarity, which drifts with layer.

Reported alongside: median positive, median null, and their ratio.

---

## Thresholds (binding)

| # | Criterion | Bar |
|---|-----------|-----|
| 1 | **AUC**, `dim_signature`, best layer | **≥ 0.80** to pass; ≥ 0.90 is strong |
| 2 | **Median ratio** positive ÷ null, same layer | **≥ 1.5×** |
| 3 | **Determinism**: identical prompt twice | **exactly 1.000** |
| 4 | **L36 specifically** must clear criterion 1 | else the preview was luck and the layer choice is unjustified |

Pass = 1, 2 and 3, with 4 reported separately. If the best layer passes but L36 doesn't, that
is a **partial pass**: the approach holds, the layer choice was wrong, and the preview
over-fit to three prompts.

`text_bridge` set-Jaccard is reported with the same statistics and **carries no pass/fail**.
It is expected to be weak on Gemma with the raw logit lens (the readout at the commit
position reads "how do I open a reply", not the subject). It is diagnostic for the
cross-model door, which is blocked on the fitted transport regardless.

---

## Corpus

24 subjects × 5 paraphrases = 120 prompts. Positive pairs: 24 × C(5,2) = **240**. Null pairs
sampled from the 24 × 23 / 2 × 25 cross-subject space, capped for runtime.

Paraphrases vary surface form while holding subject: question vs imperative, different verbs,
different word order. They are **not** synonym substitutions of a template, which would test
tokenizer robustness rather than semantic robustness.

---

## Failure modes to distinguish if it fails

- **AUC ≈ 0.5 at every layer** — the key carries no subject information. Something is still
  dominated by content-independent structure; check the baseline's sample count.
- **AUC high only at the last layer** — the key is reading the output distribution, not a
  disposition. Interesting but not first-thought.
- **High positive *and* high null** — the signature is saturated by shared template
  structure; the commit position sits in scaffolding common to every prompt.
- **Layer-dependent sign flips** — small-sample noise; needs more subjects before any claim.
