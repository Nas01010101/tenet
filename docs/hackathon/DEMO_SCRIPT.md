# 3-minute demo video script

Goal: show Tenet *doing the hard things* (supersession, forgetting, time-travel, MCP),
not just "it stores text". Record screen + voice; upload public to YouTube.

**0:00 – 0:20 — Hook / problem**
> "LLM agents forget between sessions, and most memory tools just append and retrieve.
> The hard parts are facts that *change*, forgetting what's *stale*, and recalling under
> a small context window. That's Tenet."

**0:20 – 0:50 — Supersession (the money shot)**
Terminal, run the e2e test or a live snippet:
```python
m.ingest("Hi, I'm Alex, I live in Montreal and I'm vegetarian.")
m.ingest("Update: I moved to Toronto. Also my manager is now Sarah Chen.")
m.recall("where does the user live and who manages them?")
```
> "Watch — I told it Montreal, then Toronto. It doesn't keep both. The old value is
> *superseded* — retired to history — so it answers Toronto, only. Same for the manager."
Show `stats()`: current=3, superseded=2.

**0:50 – 1:15 — Time-travel + forgetting**
```python
m.recall("where did the user live?", as_of=<before the move>)   # → Montreal
m.forget_sweep()                                                 # stale low-value archived
```
> "History isn't lost — I can ask what it believed *before* the move. And stale,
> low-value memories get forgotten automatically; pinned identity facts never do."

**1:15 – 1:45 — Bi-temporal + how it works**
Show `docs/architecture.svg` on screen.
> "Every fact has event time and transaction time. Write-time distillation turns raw
> messages into atomic keyed facts so updates supersede reliably. The read path has no
> LLM — pure vector plus decay — so recall is fast."

**1:45 – 2:15 — Unplug the internet**
Turn off wifi on screen. Terminal, with `.env`/shell set to:
```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=tenet-distiller-1.5b-v2 EMBED_PROVIDER=ollama
```
```python
m.ingest("I just moved to Denver.")
m.ingest("Update: I moved to Austin.")           # supersedes, fully offline
m.uncertain_facts()                              # learned-dynamics confidence, zero-LLM (`tenet doubts` on the CLI)
m.recall("where did I live?", as_of=<before the move>)   # time-travel, zero-LLM
```
> "Wifi's off. This is our own LoRA-tuned distiller, trained on an RTX 3080, replacing
> Qwen Cloud for the one LLM call in the write path. Learn, supersede, doubt,
> time-travel — the whole loop, zero cloud calls."

**2:15 – 2:45 — MCP + Qwen Cloud**
Show Claude Desktop (or the MCP tool list) using `learn`/`recall`; show `smoke_test.py`.
> "Back online, it's MCP-native — drop it into any MCP client and your agent has
> persistent memory, powered by Qwen Cloud: distillation on qwen3.6-flash, retrieval
> on text-embedding-v4, reading on qwen3.7-plus."

**2:45 – 3:00 — Honest results + close**
Show the `docs/BENCHMARK.md` table.
> "It's competitive with strong RAG on raw recall — but its edge is answering with the
> *current* value of a changed fact, offline if you need it. That's Tenet: memory that
> manages itself."

---

## Final shot list (<3 min) — exact commands per beat

Timed beats mapped to the exact terminal commands to run while recording. Two env
workarounds matter on a machine like this one (large ML caches redirected to an
external/network volume): if that volume isn't mounted or writable, override the cache
vars inline rather than editing shell profile, and point the DB at a scratch path so the
recording starts from a clean store every take:

```bash
export TENET_DB_PATH=/tmp/tenet_demo.db          # clean store per take
export HF_HOME=~/.cache/huggingface               # only needed for local-embedder beats
export TRANSFORMERS_CACHE=~/.cache/huggingface     # transformers reads this before HF_HOME
```

| beat | time | on screen | command |
|---|---|---|---|
| Hook | 0:00–0:20 | title card / talking head | — (voiceover only, see script above) |
| Zero-key demo | 0:20–0:50 | terminal, local embedder, no key, no network | `pip install tenet-memory[local]` then `python examples/00_zero_key_demo.py` — **verified**: runs clean end-to-end offline with the env overrides above (supersession, time-travel, and a live `doubts` line all print); note it does need network on the *very first* run to fetch `bge-small-en-v1.5` once — pre-warm the cache before recording so the take itself is airplane-mode-safe |
| Web ledger + supersession/time-travel | 0:50–1:30 | `src/tenet/static/index.html` via the HTTP API | `uvicorn tenet.api:app --host 0.0.0.0 --port 8000` then drive `/ingest` + the ledger UI in a browser (struck-through history, time-travel scrubber) |
| Doubts + drift model | 1:30–1:50 | terminal, `tenet doubts` Rich table | `tenet doubts` (needs a store with some age-decayed facts — reuse the zero-key demo's seeded ledger, `TENET_DB_PATH` still pointed at it) |
| Unplug-the-internet, zero-cloud | 1:50–2:20 | wifi toggled off on screen, terminal | `LLM_PROVIDER=ollama OLLAMA_MODEL=tenet-distiller-1.5b-v2 EMBED_PROVIDER=ollama tenet remember "I moved from Boston to Seattle"` then `tenet doubts` — needs the ollama box reachable (local or `OLLAMA_BASE_URL` over Tailscale) *before* wifi drops if it's remote |
| Benchmark slide | 2:20–2:50 | `docs/BENCHMARK.md` §3 churn curve (`docs/horizon.svg`) + §6 FactConsolidation table, or the README "Results at a glance" table | no command — static slide/scroll |
| Close | 2:50–3:00 | logo / repo URL card | — |

**Verification status, stated plainly:** the zero-key demo (0:20–0:50 beat) was actually
executed end-to-end during this pass and confirmed working with the cache-var overrides
above. The web-ledger, `tenet doubts`, and offline-ollama beats were not re-executed here
(the ollama beat needs the GPU box up and the offline beat needs wifi actually toggled,
neither of which is safe to script blind) — they're the same commands already documented
and working elsewhere in this repo (README "Fully local / air-gapped", `tenet doubts`
CLI, `src/tenet/api.py`); dry-run each once before the real recording take.
