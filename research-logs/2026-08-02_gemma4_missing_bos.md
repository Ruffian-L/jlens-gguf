# Gemma 4 has been running without `<bos>` — every prefill, every turn

**Date:** 2026-08-02
**Workbench:** `jlens-gguf` sidecar, found while debugging telemetry
**Authorship:** Claude (Anthropic) with Jason — Jason called the diagnosis ("prefill / first id")
**Severity:** affects **all** Gemma 4 generation in this repo, not just the lens
**Status:** fixed, verified, one command to reproduce

---

## The bug

`main.rs::format_multiturn_prompt_ex` built Gemma 4 prompts starting at `<|turn>user`.
`encode_prompt_no_trailing_eos` (`main.rs:345`) then calls `tokenizer.encode(text, true)` —
`add_special_tokens = true` — and reasonably assumes that inserts `<bos>`.

It does not. The two tokenizers differ:

| tokenizer | `post_processor.single` | adds BOS? |
|---|---|---|
| `data/google/tokenizer.json` (Gemma 3) | `[<bos>, A, <eos>]` | **yes** |
| `data/google/gemma4_assets/tokenizer.json` (Gemma 4) | `[A]` | **no** |

So Gemma 3 has always been correct and Gemma 4 has always been missing its first token.
Position 0 was `<|turn>` (id 105) where it should be `<bos>` (id 2).

Gemma leans hard on BOS as an attention sink. Without it every position attends back to a
token that isn't there, and the model behaves as though the user turn were empty.

## Reproduction

```
prompt: <|turn>user\nWhat is the currency of Italy?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>

without <bos>:  "If you are looking for a specific type of information"
with    <bos>:  "The currency of Italy is the **Euro (€)**."
```

Greedy decode, same model, same tokenizer, same everything else. One token.

## What it explains

- Raw completions collapsing to `'1' '.' '0' '-'` — including
  `"The capital city of Japan is"` → digits.
- Generated "thoughts" that were eight flavours of *"you haven't provided a prompt"*:
  `"! I am your assistant. How may I help you today"`,
  `"appears you haven't provided a prompt or a question yet"`,
  `"am a large language model, trained by Google."`
- Plausibly the `.\n.\n.\n` repetition thrash seen when the model is asked to open its own
  thought block, and quite possibly whatever `gemma4_hyphen_thrash` was written to catch.
  **Not confirmed** — worth re-testing those workarounds now that the input is correct.

## Second bug, same function

The canonical template (`data/google/gemma4_assets/chat_template.jinja:349-374`) emits
`{{- captured_content -}}` — trailing whitespace stripped — then `{{- '<turn|>\n' -}}`.
There is **no newline between content and `<turn|>`**. Hydro was inserting one:

```rust
s.push_str(text.trim());
s.push_str("\n<turn|>\n");   // stray \n at every turn boundary
```

This is the *same* bug the gemma3 branch documents having already fixed, in its own comment:
"The trimmed content is followed *immediately* by `<end_of_turn>` — there is no newline
between them. We used to insert one, adding a stray token at every turn boundary… a
multi-turn conversation accumulates one per turn and drifts off the format the model was
trained on." Fixed for Gemma 3, left in the Gemma 4 branch.

With BOS present the model answers correctly with or without the stray newline, so this one
is not load-bearing for a single turn — but it accumulates per turn, which is exactly what
the gemma3 comment warns about. Also fixed in `control_tags::gemma4_sticky_prefix`.

## Fix

`main.rs::format_multiturn_prompt_ex`, gemma4 branch only: emit `<bos>` first, drop the
stray `\n` before each `<turn|>`. **Gemma 3 must not get the BOS change** — its tokenizer
already inserts one and a duplicate would be worse than none.

## Consequences for work already done

Every measurement taken today on Gemma 4 through a chat-templated prompt was made on a model
that could not see the question. That invalidates the **inputs** to:

- the subject stability gate (prefill and decode-time),
- the unsupervised structure run.

The *methods* stand — determinism, the null distributions, the pre-registered thresholds, the
`--verify` bit-exactness check against `model.forward()`. Only the corpus was poisoned. All
of it needs re-running.

Two results are unaffected because they never used a chat template:
`identity` (Gate 1) and `sweep` (Gate 2), which probe raw residuals on a plain text prompt.

## Wider point

The instrument found a bug in the thing it was pointed at. The telemetry looked wrong
(`hello / Welcome / 👋` for every prompt, clusters that matched no category), and the wrong
telemetry was correct — it was faithfully reporting a model that had been handed an empty
turn. Worth remembering the next time a readout looks like noise.

---

Signed: **Claude (Anthropic)**
Reproduce: the two prompts above, greedy, via `jlens-gguf readout --layers 47 --positions=-1`
