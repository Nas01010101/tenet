# Devpost submission package

Copy-paste ready. Fill the two bracketed links after you push the repo + upload the video.

---

## Track
**Track 1: MemoryAgent**

## Project name
**Mnemo — self-managing bi-temporal memory for AI agents**

## Elevator pitch (≤ 200 chars)
Persistent memory that stores, supersedes, forgets, and recalls under a context budget —
bi-temporal, no LLM in the read path, on Qwen Cloud, exposed over MCP + HTTP.

## Text description
**The problem.** LLM agents forget everything between sessions, and the memory layers
bolted on top mostly *append and retrieve* — they don't handle the hard parts: a fact
that **changes** over time, **forgetting** what's gone stale, and **recalling** the right
thing when the context window is small. Track 1 asks for exactly those three.

**Mnemo** is a memory service built around them:
- **Bi-temporal model** — every fact carries *event time* (`valid_at`/`invalid_at`) and
  *transaction time* (`created_at`/`expired_at`). When a fact changes, the old value is
  **superseded** (retired to history), not overwritten — so current recall returns only
  the truth *now*, while `recall(as_of=…)` can still answer "what did I believe in March".
- **Write-time distillation** — Qwen (`qwen3.6-flash`) turns raw messages into atomic
  facts with a stable `subject::attribute` key, so later updates reliably collide and
  supersede. A **hybrid index** also keeps raw verbatim slices so specific detail
  (durations, numbers) survives.
- **Timely forgetting** — salience-weighted recency decay; a sweep archives stale,
  low-value memories while pinned identity facts are never forgotten.
- **Recall under a budget** — `recall(char_budget=N)` fills to a token budget; the read
  path is pure vector + decay with **no LLM call**, so retrieval is fast.
- **MCP-native** — exposes `learn` / `recall` / `forget_stale` / `stats` so any MCP
  client (Claude Desktop, IDEs, other agents) gains persistent memory; also a FastAPI
  HTTP API.

**Built on Qwen Cloud** end-to-end: `text-embedding-v4` for retrieval, `qwen3.6-flash`
for distillation, `qwen3.7-plus` for reading — all via the OpenAI-compatible DashScope
(Alibaba Cloud Model Studio) API. Optional Alibaba Cloud OSS snapshots for durability.

**Honest evaluation.** On LongMemEval_S, raw retrieval recall is competitive with naive
RAG (we don't claim a leaderboard win); Mnemo's real, measured edge is **serving the
current value of a changed fact**, forgetting, and time-travel — capabilities RAG is
structurally unable to provide. Full numbers + limitations in `docs/BENCHMARK.md`.

**What's novel.** A memory system engineered for the *accuracy–latency–forgetting*
frontier rather than pure recall, with bi-temporal supersession + a stable semantic key
from distillation, exposed over MCP.

## Built with
`Qwen Cloud` (qwen3.7-plus, qwen3.6-flash, text-embedding-v4) · `Model Context Protocol` ·
`FastAPI` · `sqlite` · `NumPy` · `Alibaba Cloud OSS` · `Python`

## Links (fill in)
- **Code repository:** https://github.com/Nas01010101/mnemo (public, MIT license visible in About)
- **Demo video (≤3 min):** [YOUTUBE URL]
- **Architecture diagram:** `docs/architecture.svg` in the repo
- **Proof of Alibaba Cloud services/APIs:** `src/config.py` + `src/distill.py` +
  `src/memory.py` call `dashscope-intl.aliyuncs.com` (Alibaba Cloud Model Studio);
  optional OSS: `src/alicloud_oss.py`
- **Blog post (optional, Blog Post Prize):** [BLOG URL]

## Submission checklist
- [ ] Public repo + LICENSE visible in About section
- [ ] Alibaba Cloud services used (DashScope) + proof file linked
- [ ] Architecture diagram (`docs/architecture.svg`)
- [ ] ≤3-min demo video on YouTube (public)
- [ ] Text description (above)
- [ ] Track identified (Track 1)
- [ ] (optional) blog/social post linked
- [ ] (optional, for full "runs on Alibaba Cloud" credit) backend deployed to ECS/FC —
      see `docs/DEPLOY.md`
