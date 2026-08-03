# jlens-gguf

**Reading what a language model is *disposed* to say, from inside a GGUF file.**

An open notebook, not a paper. Hypotheses are registered before runs, results are posted
whether they hold or not, and the log keeps the wrong turns in.

Two crates:

| crate | what it is |
|---|---|
| `gguf-hooks` | Load a GGUF model with [candle](https://github.com/huggingface/candle) and hook every residual-stream site in its forward pass. Read **or replace** activations at `PreLayer` / `PostAttn` / `PostMlp` / `FinalNorm`, any layer, without forking the model files. |
| `jlens-gguf` | A Jacobian-lens sidecar built on it: capture residuals, transport them, decode them into vocabulary, and key memory on the result. |

---

## The question

The [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html) reads out
what an internal activation is disposed to make a model *say*, by transporting a mid-layer
residual into the final-layer basis and decoding it with the model's own unembedding.

We're asking a narrower thing: **can you address a memory by how a thought opened, rather
than by what it was about?**

Text is a lossy projection of the state that produced it — 3840 floats → 262k logits → one
sampled token. If the useful structure lives in the geometry, keying on text throws it away.
Concretely, from one of our runs, a single mid-layer location read out as:

```
also   également   también   аналоги
```

One geometric position; four tokens; the channel had to pick one.

## Where we've got to

| | |
|---|---|
| Hook + capture + unembed, verified bit-exact against `model.forward()` | ✅ |
| Finite-difference Jacobian through a quantised GGUF | ❌ **impossible** — [why](research-logs/2026-08-02_gguf_quantization_blocks_fd_jacobian.md) |
| …through a dequantised one | ✅ `CANDLE_DEQUANTIZE_ALL=1`, textbook plateau across 27 blocks |
| Do first-thought keys cluster by **subject**? | ❌ AUC ≈ 0.50, pre-registered gate, robustly |
| Do they cluster by **stance**? | 🔬 one run says yes — **unreplicated** |

### The stance result, stated honestly

Clustering mid-thought residuals with no labels at all, on 200 prompts (40 subjects × 8
question forms), the clusters came out as *opening moves* rather than topics:

```
cluster 2   "teach desert ecology to a beginner, I would use a **"
            "understand how coral reefs work, you have to think of them"
            "teach monsoon agriculture to a beginner, I would use a **"

cluster 5   "its simplest, **Garbage Collection (GC)** is an"
            "its simplest level, **crop rotation** is the practice of"
            "its simplest level, a **supply chain** is the entire"
```

Unrelated subjects, one stance each. That is the hypothesis this repo exists to test.

**Caveats we'd want a reader to have before they believe it:** one model, one corpus, n=200,
single run. Gemma 4 is unusual (thinking-channel format, √d embedding scale, BOS-sensitive)
so this may be a Gemma fact rather than a language-model fact. The continuation-agreement
metric is partly circular under greedy decode. Replication on a second model and corpus is
the next thing, not a footnote.

## Things that fell out along the way

Both reproducible in one command, both independently useful:

- **[Gemma 4 GGUF prompts are missing `<bos>`](research-logs/2026-08-02_gemma4_missing_bos.md).**
  Gemma 4's tokenizer has `post_processor.single = [A]` — it adds *nothing* — where Gemma 3
  has `[<bos>, A, <eos>]`. Hand-built prompts therefore start at `<|turn>` (105) instead of
  `<bos>` (2), and the model answers as though the user turn were empty. `"What is the
  currency of Italy?"` → `"If you are looking for a specific type of information"`. With
  `<bos>`: `"The currency of Italy is the **Euro (€)**."` One token.

- **[Quantised GGUF forward passes have no derivative](research-logs/2026-08-02_gguf_quantization_blocks_fd_jacobian.md).**
  candle quantises *activations* to 8-bit blocks inside every quantised matmul
  (`k_quants.rs:2296`), so the forward pass is piecewise-constant in its input. Finite
  differences measure rounding-boundary flips, not slopes — the response is flat across
  eight decades of ε, including across a single block. Anything gradient-flavoured on a
  quantised path is measuring something other than what it thinks.

## Method notes

Things that are load-bearing and were each learned the hard way:

- **A null distribution, always.** A 3-prompt preview showed a confident +0.29 separation
  that a proper null flattened to 0.50.
- **Baselines must be position-matched.** Subtracting an all-positions mean from a
  last-position residual leaves a common-mode so large that two *unrelated* prompts sat at
  cosine 0.74.
- **Don't centre before unembedding.** It removes the component driving the prediction —
  paraphrases forced to the same answer shared *nothing* in their top-8 tokens.
- **Rank by z-score, not magnitude.** Raw top-|h| dimensions are the model's constant
  outlier dims, identical for every prompt.
- **Verify against `model.forward()`.** Unembedding the last block must reproduce the real
  logits exactly. It's how we knew the capture was sound while everything else was wrong.

## Getting started

```bash
cargo build --release

# What will the lens see?
jlens-gguf info --model model.gguf

# Per-layer residual statistics (not optional — see method notes)
jlens-gguf baseline --model model.gguf --prompts corpus.txt --layers 24,36,44 \
  --commit-only --out baseline.safetensors

# Disposition snapshots as JSONL
jlens-gguf readout --model model.gguf --prompt "$PROMPT" \
  --layers 24,36,44 --baseline baseline.safetensors --verify

# Is there structure? No labels, shuffled null.
jlens-gguf structure --model model.gguf --prompts corpus.txt --layers 36 --k 8
```

Gemma 4 prompts need `<bos>` and the template
`<bos><|turn>user\n{q}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>` — no newline
before `<turn|>`.

## Reading order

- [`docs/PLAN.md`](docs/PLAN.md) — the original plan, frozen unedited
- [`docs/DESIGN.md`](docs/DESIGN.md) — why it's built this way; the forward-mode derivation
- [`docs/STABILITY_GATE.md`](docs/STABILITY_GATE.md) — thresholds registered before the run
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — **what actually happened**, including everything
  that went against the plan
- [`research-logs/`](research-logs/) — standalone findings

## Licence & credit

Code MIT-0. `llama.rs` / `gemma.rs` are modified from
[candle-transformers](https://github.com/huggingface/candle) (Apache-2.0 OR MIT); `gemma4.rs`
is original. The lens algorithm is a port of Anthropic's
[`jacobian-lens`](https://transformer-circuits.pub/2026/workspace/index.html) reference
implementation (Apache-2.0) — with reverse-mode replaced by a forward-mode estimator, since
candle has no gradient through quantised matmul. Model weights carry their own terms; see
[`NOTICE`](NOTICE).

Extracted from the `hydrodynamic-swarm` research codebase.
