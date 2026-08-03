# Executive decisions log

Jason handed over decision-making to keep momentum while resting (2026-08-02). Every call
made without him is recorded here with its reasoning, so any of them can be reversed on
review. Newest last.

---

**D1 — Proceed without Grok's "show me before any code runs" gate.**
Grok's paste asked for STANCE_GATE.md to be reviewed before implementation. Jason overrode
that in the same message ("make every executive decision you need... I trust you"). The doc
is still written and pre-registered *before* any run, so the protective intent is preserved;
only the human round-trip is skipped. **Reversible:** doc is committed unedited.

**D2 — Stance labels: adopt Grok's five starters unchanged.**
`enumerating | conditional | definitional | causal | comparative`. They are a reasonable
partition of opening moves and Jason called them "only a starter". Not worth burning his
absence inventing a taxonomy he may reject anyway. **Reversible:** labels live in one table
in `stance.rs`, swappable without touching the gate.

**D3 — Stances are *observed*, not assigned.**
We cannot instruct the model into a stance without changing the prompt, which would
confound the key with the instruction. So: generate the thought for every prompt, classify
the *observed* opening, then test whether same-stance keys cluster across different
subjects. This makes the corpus self-labelling and needs no hand annotation.

**D4 — Two arms, because the obvious design is circular.**
The residual at depth 0 is the state that emitted content token 1. Labelling stance from
tokens 0–3 and then keying on depth 0 would make a high AUC almost tautological — the key
would be "encoding the next token", dressed up as disposition.
- **Arm A (surface):** label from tokens 0–3, key at depth 0. Expected to pass. Documents
  that the key tracks imminent output. Not evidence of disposition.
- **Arm B (predictive):** label from tokens 8–15, key at depth 0. Non-circular — asks
  whether the *opening* state foreshadows how the thought later unfolds. **Arm B is the
  real test.** A pass on A with a fail on B means the key is a next-token echo, not a
  stance.
Plus a **shuffle control**: labels randomly permuted must give AUC ≈ 0.5, or the pipeline is
leaking.

**D5 — Stay on `gemma-4-12b-it-Q4_K_M.gguf`.**
The new unquantised safetensors downloads (E2B 9.6G, 12B 23G) matter for the *fitter*, which
is locked. Switching model now would invalidate every result gathered today for no gain on
the stance question. Noted for later: a BF16 GGUF would route through candle's
`QMatMul::Tensor`/`TensorF16` path, which does **no** activation quantisation — that is the
cheapest route past the Gate 2 blocker when the fitter unlocks, and it needs no dequantise
code at all. **Recorded, not acted on.**

**D6 — Keep every permanent fix from today.**
Header prefill, N = content tokens, position-matched baseline, raw-residual unembed,
determinism check. These are load-bearing and are not revisited.

**D7 — SplatRAG scope: emit, don't integrate.**
Jason asked for "jlens working in a way that matters for SplatRAG", explicitly not full
SplatRAG. The sidecar already emits JSONL with the three doors. The useful next step is a
stance-aware record the picker can consume, not wiring into the SplatRAG store on
`/media/ruffianl/ghost_team/...` while he is away and cannot check a write. **No writes
outside this repo and the scratchpad.**
