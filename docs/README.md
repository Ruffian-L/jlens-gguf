# `jlens-gguf`

A Jacobian-lens sidecar for GGUF models. Loads through hydro's own loader (`src/loader.rs`),
so it measures the same weights the swarm runs.

- `PLAN.md` — the approved plan, frozen as written.
- `DESIGN.md` — why it is built this way; the forward-mode derivation.
- `CHANGELOG.md` — what actually happened, including the results that went against the plan.

## Status

| piece | state |
|-------|-------|
| logit-lens telemetry (`readout`, `baseline`) | **works**, verified against `model.forward()` |
| transport + unembed + key schema | works (exact arithmetic) |
| subject stability gate | **FAIL expected** (keys are not subject-shaped) — see CHANGELOG |
| stance gate | **protocol ready** — `STANCE_GATE.md`; no code until PI go |
| fitting `J` by finite differences | **blocked** — see CHANGELOG, Gate 2 |

## Producing telemetry

Two steps. The baseline is not optional: without it the key ranks the model's constant
outlier dimensions and does not discriminate between prompts at all.

```bash
# 1. Per-layer residual statistics over a corpus. One prompt per line;
#    a literal \n in the file becomes a real newline, so a chat template fits on one line.
jlens-gguf baseline \
  --model ~/models/gemma-4-12b-it-Q4_K_M.gguf \
  --prompts corpus.txt --layers 24,36,44,47 \
  --out baseline.safetensors

# 2. Disposition snapshots as JSONL.
jlens-gguf readout \
  --model ~/models/gemma-4-12b-it-Q4_K_M.gguf \
  --prompt "$PROMPT" --layers 24,36,44,47 --positions=-1 \
  --baseline baseline.safetensors \
  --state-dir states/ --emit telemetry.jsonl --tag "turn-17"
```

`--verify` also runs the model's own `forward()` and prints its top-k. The last layer's
record must match it exactly; if it doesn't, the capture is wrong and the run is fiction.

**Gemma 4's chat template** is `<|turn>user\n…\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
(`src/main.rs:277-287`). Gemma 3's `<start_of_turn>` markers tokenise as *literal text* on
Gemma 4 and produce readouts of template gibberish.

## The record: one object, three doors

| door | field | scope | answers |
|------|-------|-------|---------|
| verbalizable | `text_bridge`, `text_bridge_hash` | **cross-model** | what it leaned toward saying |
| fingerprint | `dim_signature` | **within-model only** | which internal directions were live |
| rehydration | `state_ref` | within-model, exact | put the model back in this stance |

They are not interchangeable. `dim_signature` indexes raw residual dimensions, and
dimension 1523 in Gemma has no relationship to dimension 1523 in Qwen — residual bases are
per-model and arbitrary. Any basin that must hold across models has to form on
`text_bridge`, which is basis-independent because it is text. Same rule as the picker's
("a pick carries text; the host re-embeds in its own residual dim"), from the other side.

`lens` records what produced the numbers and is load-bearing: `logit` (no transport, ships
today), `jacobian` (fitted transport, blocked), `secant` (large-ε finite difference of the
quantised model — **not** the paper's `J`, and never labelled as it).

## What the readouts actually show

Measured on gemma-4-12b, mid-stack, first generated position:

- `dim_signature` separates subject — paraphrase beats unrelated at every layer, best at
  L36 (+0.291). Small sample; the stability gate is still owed.
- `text_bridge` reads the *opening move* (`hello / Welcome / 👋`), not the subject. That is a
  correct readout of the disposition at that position.
- Mid-stack the logit lens shows concepts before tokens — `also / également / también /
  аналоги`, one concept in four languages — but never surfaces subject words. That is the
  known weakness of the raw logit lens on Gemma, and the gap the fitted transport fills.
