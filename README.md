# Mnemo — self-managing memory for AI agents

**A bi-temporal, self-forgetting memory service on Qwen Cloud, exposed over MCP + HTTP.**
Built for the [Global AI Hackathon with Qwen Cloud](https://qwencloud-hackathon.devpost.com) —
**Track 1: MemoryAgent**.

Give any LLM client persistent memory that **stores** what matters, **supersedes** facts
when they change, **forgets** what goes stale, and **recalls** the right thing under a
limited context window — with **no LLM in the read path**.

![architecture](docs/architecture.svg)

## Why it's different
Most memory systems are append-and-retrieve. The three things Track 1 explicitly asks
for — efficient retrieval, *timely forgetting*, and *recall under limited context* — are
exactly what append-only systems don't do. Mnemo is built around them:

| Capability | How |
|---|---|
| **Bi-temporal supersession** | facts carry event time (`valid_at`/`invalid_at`) **and** transaction time (`created_at`/`expired_at`); a changed fact **retires** the old value instead of overwriting it |
| **Time-travel** | `recall(as_of=…)` answers "what did I believe last March" from retained history |
| **Timely forgetting** | salience-weighted recency decay; a background sweep archives stale, low-value memories (pinned facts never forgotten) |
| **Recall under a budget** | `recall(char_budget=N)` fills to a token budget — recall under a limited context window |
| **Write-time distillation** | raw messages → atomic facts with a stable `subject::attribute` key so updates reliably supersede |
| **Hybrid index** | distilled facts (consistency/temporal) **+** raw slices (verbatim detail), dual-pool recall |
| **MCP-native** | plug persistent memory into Claude Desktop / any MCP client (`learn` · `recall` · `forget_stale` · `stats`) |

The read path uses pure vector + decay scoring — **no LLM call** — so retrieval is fast
and the memory is frontier-correct on the accuracy/latency axis (LongMemEval-V2's direction).

## Architecture
Two surfaces (MCP server + FastAPI) over one `MemoryCore`, backed by Qwen Cloud
(Alibaba Cloud Model Studio) for distillation + embeddings, with optional Alibaba Cloud
OSS snapshots for durability. See [`docs/architecture.svg`](docs/architecture.svg) and
[`docs/DESIGN.md`](docs/DESIGN.md).

## Quickstart
```bash
cp .env.example .env && chmod 600 .env     # add your DASHSCOPE_API_KEY
pip install -r requirements.txt
python scripts/smoke_test.py               # verify Qwen Cloud connectivity

# use it
python -c "import sys; sys.path.insert(0,'src'); from mnemo import Mnemo; \
  m=Mnemo(); m.ingest('I moved to Toronto last week.'); \
  print([x.text for x in m.recall('where does the user live?')])"

# run the HTTP API
cd src && uvicorn api:app --host 0.0.0.0 --port 8000
# or the MCP server
python src/mcp_server.py
```
Claude Desktop MCP config:
```json
{ "mcpServers": { "mnemo": { "command": "python", "args": ["/ABS/PATH/src/mcp_server.py"] } } }
```

## Results (honest)
On **LongMemEval_S** (Qwen-judged, indicative — see [`docs/BENCHMARK.md`](docs/BENCHMARK.md)):
- Session-level recall@10 is **competitive with naive-RAG** (RAG 43% / Mnemo 37%); Mnemo
  **wins multi-session**. Raw retrieval recall is *not* where a memory system's value lies.
- Mnemo's real edge — **serving the current value of a changed fact** (supersession),
  **forgetting**, and **time-travel** — is proven deterministically in the test suite and
  measured by the knowledge-update benchmark.

We do **not** claim a leaderboard-topping number; we report what we measured and are
explicit about limitations.

## Tests
```bash
python scripts/test_memory.py        # bi-temporal, supersession, time-travel, forgetting, budget
python scripts/test_mnemo_e2e.py     # raw messages → distill → supersede → clean recall
```

## Deploy on Alibaba Cloud
Only `DASHSCOPE_API_KEY` is needed to run. Qwen Cloud/DashScope *is* Alibaba Cloud Model
Studio, so the model + embedding calls satisfy the "uses Alibaba Cloud services/APIs"
proof; `src/alicloud_oss.py` (OSS snapshots) is an optional stronger proof. Full runbook:
[`docs/DEPLOY.md`](docs/DEPLOY.md).

## Layout
```
src/  config.py memory.py distill.py mnemo.py mcp_server.py api.py alicloud_oss.py
scripts/  smoke_test.py test_memory.py test_mnemo_e2e.py lme_recall.py bench_knowledge_update.py
docs/  DESIGN.md SOTA.md BENCHMARK.md DEPLOY.md COMPETITION.md architecture.svg
```

## License
MIT — see [LICENSE](LICENSE).
