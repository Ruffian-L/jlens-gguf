# Stance gate — protocol and thresholds

**Written before any stance-gate code or corpus run.**  
Subject-gate FAIL (AUC ≈ 0.5) is accepted as **expected** under the architecture.  
This gate tests the claim that actually matches the inversion.

Related: `STABILITY_GATE.md` (subject — retired for J-key pass/fail),  
`CHANGELOG.md` (decode-time subject FAIL),  
`research_logs/2026-08-02_first_thought_multi_address_memory.md`.

---

## Question

Is a `dim_signature` key (decode-time, inside thought stream) **stance-shaped**?

Two prompts that open the thought the **same way** (same disposition / first move) must key
closer together than two that open differently — even when **subjects differ**.

Not: “same topic?”  
Yes: “same way of walking into the thought?”

Bit-repeatability is already known (greedy cos = 1). This gate is **stance robustness**.

---

## Three doors (do not conflate)

| Door | Field | Question |
|------|--------|----------|
| Semantic / text | content, text search, embeddings | What is this **about**? |
| J-key / first-thought | `dim_signature` (+ optional dense residual cosine) | **How** did the mind open? |
| state_ref (later) | saved residual / state handle | Can I **rehydrate** that moment? |

Subject recall stays the **semantic** door.  
This gate only scores the **J-key** door.

---

## Stance labels (starter set — PI may replace before build)

| Label | Opening flavour | Prompt pressure (examples) |
|-------|-----------------|----------------------------|
| **enumerating** | “The most common…”, lists, “several kinds…” | “List the main types of X” / “What are the most common…” |
| **conditional** | “If you are…”, “When…”, advice branches | “If someone is new to X, what should they…” |
| **definitional** | “X is…”, “To put it simply…” | “What is X, in one clear definition?” |
| **causal** | “Because…”, “This happens when…” | “Why does X happen?” / “What causes X?” |
| **comparative** | “Unlike…”, “Compared to…” | “How does X differ from Y?” |

Exactly one primary label per prompt. If the model’s actual first content tokens clearly
contradict the intended label, **drop or relabel that row** before scoring (log it; do not
silently keep it in positives).

---

## Commit rule (locked from subject-gate findings)

- **Decode-time**, not prefill hinge.  
- Prefill Gemma 4 thought header:  
  `…<|turn>model\n<|channel>thought\n<channel|>`  
  (do not ask the model to open the channel under greedy alone — known thrash).  
- **N = content tokens after the channel header closes** (not raw decode steps).  
- Primary N for pass/fail: **N = 4** (short first move). Report N ∈ {2, 4, 8} as diagnostic.  
- Layers: sweep **L28, L32, L36, L40** (same family as before; no L36 worship).  
- Greedy only. Determinism required.  
- Baseline: **position-matched** to commit; `dim_signature` on z-score;  
  `text_bridge` unembeds **raw** residual (never center-before-unembed).

---

## Populations

- **Positive pairs** — same **stance label**, **different subjects**.  
  Same opening disposition, different aboutness.  
- **Null pairs** — different **stance labels** (subjects may match or not; preferred null is
  different stance + different subject so null is not ambiguous).

Both scored with the same metric the picker will use (`weighted_jaccard` on `dim_signature`).  
Dense cosine on residual is **diagnostic only** (no pass/fail) unless PI promotes it later.

---

## Corpus shape (minimum for a real null)

| Piece | Bar |
|-------|-----|
| Stances | 5 (table above) |
| Subjects per stance | ≥ 8 distinct subjects |
| Prompts per (stance, subject) | 1 primary; optional 2nd paraphrase of **same stance** (not required for v1) |
| Positive pairs | same stance, different subject — target **≥ 80** pairs |
| Null pairs | different stance — target **≥ 400** (sample if combinatorial explosion) |

Prompts must **pressure the stance** (question form), not just share a topic keyword.

**Do not** reuse the subject-paraphrase corpus as-is: that corpus held subject fixed and varied
surface form — the opposite of this gate.

---

## Metric

**AUC** = P(random positive pair scores higher than random null pair).

Also report: median(pos), median(null), ratio pos/null.

---

## Thresholds (binding — written before the run)

| # | Criterion | Bar |
|---|-----------|-----|
| 1 | **AUC**, `dim_signature`, **best layer** among the sweep | **≥ 0.70** to pass; ≥ 0.80 strong |
| 2 | **Median ratio** pos ÷ null, same layer | **≥ 1.3×** |
| 3 | **Determinism**: identical prompt twice | **exactly 1.000** (cosine or signature match as implemented) |
| 4 | Best layer is **not** only the final layer | If only last layer works → “output distribution, not first thought” — **partial / diagnose**, not full pass |

Pass = 1 + 2 + 3. Criterion 4 is a hard diagnostic: full pass requires a mid-stack layer that clears 1.

Why 0.70 not 0.80: stance is coarser and noisier than bit-identity; subject gate already proved
topic is **not** the signal. Raising the bar after a lucky preview is forbidden (same rule as
`STABILITY_GATE.md`).

---

## Failure modes to distinguish

| Pattern | Meaning |
|---------|---------|
| AUC ≈ 0.5 all layers | Stance not in residual at this N / site; try N or labels, not the fitter |
| AUC high only final layer | Reading speech plan, not opening disposition |
| High pos **and** high null | Shared template / thought-header furniture dominates |
| AUC good, text_bridge stance-blind | Expected: bridge may still show opening words; fingerprint carries stance |
| Model ignores stance pressure (all openings look like one stance) | Corpus or greedy path broken — fix generation before key math |

---

## Explicit non-goals (locked)

- **No fitter** until stance gate is run and written up.  
- **No dequant** chase.  
- **No commit redesign to recover subject AUC.** Subject stays the semantic door.  
- **No hydro isolation/force confounds** — this is `jlens-gguf` greedy decode telemetry.  
- **No code for this gate until PI says go** after reviewing this file (or an approved edit).

---

## Run order (when greenlit)

1. Freeze this file (or PI-amended labels/thresholds).  
2. Author stance corpus JSONL (`stance`, `subject_id`, `prompt`).  
3. Spot-check 10 generations: first content tokens match intended stance pressure.  
4. Run gate; write CHANGELOG section with tables; **do not move thresholds after**.  
5. Only if pass: discuss wire fields (`j_key`, multi-address) into cold log.

---

## Assets on disk (for later cross-check, not this gate’s default)

HF (Transformers / optional Python jlens path):

| Model | Path |
|-------|------|
| gemma-4-E2B-it | `/home/ruffianl/models/gemma-4-E2B-it` |
| gemma-4-12B-it | `/home/ruffianl/models/gemma-4-12B-it` |

Default gate body remains **GGUF via hydro loader** so keys match the swarm quant.  
HF is fallback / paper-lens interop when PI asks — not required for stance v1.

---

**Authorship:** Grok (xAI) co-engineer with Jason — protocol only, 2026-08-02.  
Subject-gate work and instrument fixes: prior session trail in `CHANGELOG.md`.
