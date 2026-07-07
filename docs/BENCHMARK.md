# Benchmarks & honest evaluation

Tenet is evaluated on the standard **LongMemEval_S** benchmark (500 questions,
~115k-token multi-session histories) plus controlled capability tests. Every number is
**honest and reproducible** from `scripts/`. Where a strong baseline beats us, we say so.

> **Protocol.** A `gpt-4o` reader (the reader Mem0/Zep report against) + local embedder
> (`bge-small-en-v1.5`) + cheap `gpt-4o-mini` distiller, via OpenRouter. This validates the **architecture** (Tenet vs baselines under identical
> settings); the shipped product uses Qwen Cloud (`text-embedding-v4` + `qwen3.7-plus`)
> via `config.py` — swap by env, no code change. Numbers are indicative, compared only
> against baselines we run ourselves — never pasted onto the public gpt-4o leaderboard.

## TL;DR
- **Retrieval recall: on par with strong RAG** (95% vs 97.5% @k=10).
- **A frontier, not a point** (one `expand` knob): the **efficiency** point gives the
  **best accuracy-per-token** (49.2, 1.6× RAG) at *half* the context; the **parity** point
  **matches strong RAG's one-shot accuracy at equal-or-lower tokens** (57.5% = 57.5%, gpt-4o).
- **Dominates the long-horizon regime**: as a fact is updated many times, RAG collapses
  (100%→50%); **Tenet holds 100%**. This is the regime long-term memory is *for*.
- **Honest weakness**: multi-session synthesis — the one category still behind RAG
  (42.9 vs 57.1, up from 28.6). Documented in §6.

## 1. Retrieval recall — LongMemEval_S (`scripts/lme_recall.py`)
Session-level recall@10 over the full ~50-session haystack (n=40):

| System | recall@10 |
|---|---|
| naive-RAG | 95% |
| **Tenet** | **97.5%** |

Parity. (An earlier design scored 37% here; the hybrid index + stale-echo fixes below
closed the gap.)

## 2. QA answer-accuracy frontier (`scripts/lme_recall.py --qa`)
The metric production actually cares about — answer accuracy **per token of reader
context** (LongMemEval-V2's accuracy/latency direction). Tenet is a **frontier**: the
`--expand` knob spends spare context on belief-anchored evidence, so one system sits
anywhere from max-efficiency to accuracy-parity. n=40, gpt-4o reader, seed 2:

| System | mode | QA acc | reader tokens | **acc / 1k tokens** |
|---|---|---:|---:|---:|
| full-context (no memory) | — | 65%* | ~124,000 | 0.5* |
| RAG @k=10 | top-*k* turns | 57.5% | 2,101 | 27.4 |
| **Tenet** | efficiency (`--expand 0`) | 52.5% | **1,067** | **49.2** ← best/token |
| **Tenet** | parity (`--expand 20`, budget-capped) | **57.5%** | 2,083 | 27.6 |

At the **efficiency** point Tenet gives the highest accuracy per token (1.6× RAG, half its
context). **Belief-anchored evidence expansion** — refilling spare budget (capped at RAG's
own token count) with query-relevant raw turns from the sessions the belief state already
surfaced — brings raw accuracy to **parity with strong RAG at fewer tokens** (57.5% = 57.5%,
2,083 vs 2,101). Per-type at the parity point (gpt-4o):

| question type | RAG QA | Tenet QA |
|---|---|---|
| single-session-user | 83% | **100%** |
| knowledge-update | 67% | **83%** |
| temporal-reasoning | 33% | **40%** |
| multi-session | **57%** | 43% |

Tenet ≥ RAG on every type **except multi-session** (43 vs 57, up from 29 before expansion):
these questions need several evidence sessions, but expansion only deepens the sessions the
top-*k* already surfaced. Honest limitation, §6. On a cheaper `gpt-4o-mini` reader the parity
point edges ahead overall (Tenet 60.0 vs RAG 55.0). *(\*full-context under a weaker reader.)*

## 3. Long-horizon knowledge churn — where memory structurally wins (`scripts/bench_horizon.py`)
A fact updated N times over a long history, retrieval budget k=6, 15 distractors,
12 independent principals per point:

| # updates | naive-RAG | **Tenet** |
|---:|---:|---:|
| 2 | 100% | 100% |
| 4 | 100% | 100% |
| 6 | 100% | 100% |
| 8 | 67% | **100%** |
| 10 | 58% | **100%** |
| 12 | 50% | **100%** |

**RAG loses 50 points; Tenet loses 0.** Once the number of stale versions exceeds the
retrieval budget, RAG's top-k can't hold them all and the reader picks a wrong (old)
value; Tenet's bi-temporal supersession keeps exactly **one** current value regardless of
how many times the fact changed. This is the long-term-memory regime RAG cannot scale to.

## 4. Knowledge-update + the world-model mechanisms (`scripts/bench_knowledge_update.py`)
The first version of this test *refuted* the naive design (55% correct, 45% stale-leak vs
RAG 95%): the hybrid raw-slice pool reintroduced values the fact layer had retired. The
fix is a **world-model consistency rule** — the current facts are the belief state; a raw
slice echoing a *superseded* belief is stale evidence and is retired from current recall
(`_STALE_ECHO`). That took Tenet 55% → **100%**, matching RAG (0% stale-leak).

**World-model memory efficiency** — **surprise-gated writes** (predictive-coding
principle): an observation the store already predicts (cosine ≥ 0.97) carries no
information and isn't stored. Measured: **15% of turns dropped as redundant, no accuracy
loss.** RAG stores everything.

## 5. Capabilities proven deterministically (`scripts/test_memory.py`, `test_tenet_e2e.py`)
Supersession · time-travel (`recall(as_of=…)`) · forgetting sweep · context-budget recall
· distillation-driven consistent keys — all pass without any benchmark, demonstrating the
core value directly.

## 6. Honest limitations
- **Multi-session synthesis** is the one category where RAG still leads (43% vs 57%).
  Belief-anchored expansion lifted it from 29% but doesn't close it: these questions need
  evidence from *several* sessions, and expansion only deepens the sessions the top-*k*
  already surfaced — if a needed session isn't among them, its detail is still missing. Next
  step: session-diverse retrieval (guarantee coverage across distinct evidence sessions).
- **The frontier is a knob, not free lunch.** Parity accuracy costs RAG-equal tokens; the
  1.6× per-token win is at the efficiency point, which trades ~5pp of raw accuracy. One
  system spans both, but no single setting wins every axis at once.
- QA numbers are off-Qwen (gpt-4o / gpt-4o-mini readers), n=40, one seed; reader noise
  ≈±5–7pp, so the one-shot result is reported as *parity*, not a win. Shipped system uses Qwen
  Cloud (config flip). Churn result is reader-robust (identical on gpt-4o).

## Reproduce
```bash
python scripts/test_memory.py ; python scripts/test_tenet_e2e.py     # capabilities
python scripts/lme_recall.py --limit 40 --k 10 --qa --seed 2              # efficiency point
python scripts/lme_recall.py --limit 40 --k 10 --qa --seed 2 --expand 20  # parity point (budget-capped)
python scripts/bench_horizon.py --principals 12 --k 6 --updates 2,4,6,8,10,12   # long-horizon
python scripts/bench_knowledge_update.py --principals 4              # supersession + efficiency
# off-Qwen: prefix with  LLM_PROVIDER=openrouter EMBED_PROVIDER=local OPENROUTER_MODEL=openai/gpt-4o-mini
```
