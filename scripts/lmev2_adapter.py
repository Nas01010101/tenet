"""LongMemEval-V2 memory-backend adapter for Tenet (hybrid, zero-generation read path).

Rebuilt retrieval layer (2026-07-08). Beats the 51.0% RAG baseline bar on the LME-V2-Small
deterministic dev slice by fixing the two things the first cut got wrong (embedder + ranking):

  insert(trajectory) -> structure-aware, HIGH-SIGNAL chunking (chunks_v2):
      • trajectory header (goal / outcome / env)
      • per-state ACTION+THOUGHT unit (kept whole, step/url prefixed)
      • the accessibility tree with the global-nav CHROME dropped (everything before the
        first `main` landmark) and only answer-bearing lines kept (StaticText, labelled
        controls, values, prices), short overlapping windows, cross-state DOM dedup.
      Each chunk is embedded (Qwen3-Embedding-0.6B via ollama /api/embed on the RTX) and
      also indexed for BM25.  ZERO LLM generation.

  query(q, img) -> HYBRID retrieval, ZERO LLM generation, sub-second (the LAFS latency lever):
      • the answer-format boilerplate ("Mark your final answer … \boxed{}") is stripped
      • dense = cosine over Qwen3 embeddings (query gets the instruction prefix)
      • lexical = BM25 over the same chunks
      • fused with Reciprocal Rank Fusion (equal weight — measured best), top-k under a
        char budget, returned as the harness's list[{"type":"text","value":...}].

Why hybrid: on the enterprise dev slice, dense (Qwen3-0.6b) lifts recall over BM25 on the
semantic "current-state" questions BM25 can't rank, and RRF fusion beats either component
(BM25 62.1% / dense 65.5% / hybrid 75.9% @48k budget; 89.7% @72k). Corpus recall ceiling is
~92%, so the win came from RANKING, not from storing more text.

Config JSON for the harness:
  {"memory_type": "tenet",
   "memory_params": {"k": 48, "char_budget": 48000,
                     "embed_url": "http://100.88.179.78:11434",
                     "embed_model": "qwen3-embedding:0.6b"}}

Embeddings/retrieval run locally/$0 against an ollama server (default the RTX box); nothing
paid. Embedding is ingest-time (excluded from LAFS latency); query-time cost is one short
embed call (~0.3s) + numpy cosine + a pure-python BM25 pass — well under the 1s LAFS corner.

Standalone mechanism smoke test (no reader, no judge):
  python scripts/lmev2_adapter.py --data-root ~/scratch/lmev2/data/longmemeval-v2 \
      --domain enterprise --n-trajectories 8 --n-questions 5
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------
# Defaults (measured winners on the LME-V2-Small dev slice).
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    k=48,                 # context items returned per query
    char_budget=48000,    # cap total returned context chars (reader-friendly ~13k tok)
    rrf_k=60,             # RRF constant
    pool=300,             # candidates per retriever before fusion
    win=1200,             # DOM window size (chars)
    overlap=150,          # DOM window overlap
    max_windows=6,        # max DOM windows per (deduped) page state
    embed_url=os.environ.get("LMEV2_EMBED_URL", "http://100.88.179.78:11434"),
    embed_model=os.environ.get("LMEV2_EMBED_MODEL", "qwen3-embedding:0.6b"),
    embed_batch=128,
)

# Qwen3-Embedding is instruction-aware: prepend a task instruction to QUERIES only.
Q3_QUERY_INSTRUCT = (
    "Instruct: Given a web-agent task question, retrieve the trajectory steps and page "
    "states that answer it.\nQuery: "
)

# ----------------------------------------------------------------------------
# Structure-aware, high-signal chunking (chunks_v2).
# ----------------------------------------------------------------------------
_WS = re.compile(r"\s+")
def _norm(s: str | None) -> str:
    return _WS.sub(" ", (s or "").strip())

def _url_tail(url: str) -> str:
    if not url:
        return ""
    return url.split("://", 1)[-1][:90]

_MAIN_RE = re.compile(r"(?:^|\n)\s*(?:\[\d+\]\s*)?main\b", re.I)
_SALIENT = re.compile(
    r"StaticText|textbox|combobox|option|checkbox|radio|menuitem|listitem|"
    r"heading|gridcell|\bcell '|\brow '|tab '|button '|link '|\$|value=|"
    r"\bselected\b|\bchecked\b", re.I)
_EMPTY_LABEL = re.compile(r"'\s*'")

def _salient_dom(tree: str) -> str:
    """Drop global-nav chrome (before the first `main` landmark) and keep only
    answer-bearing lines (labelled controls, StaticText, values, prices)."""
    m = _MAIN_RE.search(tree)
    body = tree[m.start():] if m else tree
    keep = []
    for ln in body.split("\n"):
        s = ln.strip()
        if not s or not _SALIENT.search(s):
            continue
        if _EMPTY_LABEL.search(s) and "StaticText" not in s and "$" not in s and "value=" not in s:
            if s.count("'") <= 2:
                continue
        keep.append(s)
    return "\n".join(keep)

def _window(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out = []; i = 0; step = max(1, size - overlap)
    while i < len(text):
        out.append(text[i:i + size]); i += step
    return out

def trajectory_chunks(tr: dict[str, Any], *, win: int, overlap: int,
                      max_windows: int) -> list[str]:
    tid = tr.get("id", "?")
    goal = _norm(tr.get("goal")); outcome = _norm(tr.get("outcome")); env = _norm(tr.get("environment"))
    out = [f"[traj {tid} | env={env} | outcome={outcome}] GOAL: {goal}"]
    seen: set[str] = set()
    for st in tr.get("states", []) or []:
        if not isinstance(st, dict):
            continue
        step = st.get("step"); url = _url_tail(str(st.get("url") or ""))
        thought = _norm(st.get("thought")); action = _norm(st.get("action"))
        prefix = f"[{tid} s{step} {url}]"
        if thought or action:
            body = (f" ACTION: {action}" if action else "") + (f" THOUGHT: {thought}" if thought else "")
            out.append(prefix + body)
        tree = st.get("accessibility_tree")
        if tree:
            sal = _norm(_salient_dom(tree))
            if not sal:
                continue
            sig = hashlib.blake2b(sal.encode(), digest_size=12).hexdigest()
            if sig in seen:
                continue
            seen.add(sig)
            for w in _window(sal, win, overlap)[:max_windows]:
                out.append(f"{prefix} PAGE: {w}")
    return [c for c in out if c.strip()]

# ----------------------------------------------------------------------------
# Query cleaning — strip the answer-format boilerplate that dilutes retrieval.
# ----------------------------------------------------------------------------
_MARK_RE = re.compile(r"\n*\s*Mark your final answer.*$", re.I | re.S)
def clean_query(q: str) -> str:
    return _MARK_RE.sub("", q).strip()

# ----------------------------------------------------------------------------
# Embeddings via ollama /api/embed (Qwen3-Embedding). Unit-normalised float32.
# ----------------------------------------------------------------------------
def _embed(texts: list[str], url: str, model: str, batch: int, *, is_query: bool) -> np.ndarray:
    if is_query:
        texts = [Q3_QUERY_INSTRUCT + t for t in texts]
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] for t in texts[i:i + batch]]
        body = json.dumps({"model": model, "input": chunk, "keep_alive": "30m"}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.load(r)
                embs = d.get("embeddings")
                if not embs:
                    raise RuntimeError(f"no embeddings: {str(d)[:200]}")
                out.extend(embs); break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
    a = np.asarray(out, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1.0
    return a / n

# ----------------------------------------------------------------------------
# Pure-python BM25 (no dependency).
# ----------------------------------------------------------------------------
_TOK = re.compile(r"[a-z0-9]+")
def _tok(s: str) -> list[str]:
    return _TOK.findall(s.lower())

class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [_tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(self.docs)
        self.dl = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avgdl = self.dl.mean() if self.N else 0.0
        df: dict[str, int] = collections.Counter()
        self.tf: list[collections.Counter] = []
        for d in self.docs:
            c = collections.Counter(d); self.tf.append(c)
            for t in c:
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: str) -> np.ndarray:
        sc = np.zeros(self.N, dtype=np.float32)
        for t in _tok(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(t, 0)
                if f:
                    sc[i] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return sc

# ----------------------------------------------------------------------------
# The registered backend. Import of memory_modules is deferred/optional so this
# file also runs standalone (mechanism smoke test) without the LME-V2 repo.
# ----------------------------------------------------------------------------
try:
    from memory_modules.memory import (  # type: ignore
        Memory, MemoryContextItem, register_memory, require,
    )

    @register_memory
    class TenetMemory(Memory):
        """Tenet as an LME-V2 memory backend — hybrid (dense+BM25 RRF), zero LLM generation."""

        memory_type = "tenet"

        def __init__(self, memory_params: dict[str, object]) -> None:
            super().__init__(memory_params)
            p = {**DEFAULTS, **{k: v for k, v in memory_params.items() if v is not None}}
            self.p = p
            self._texts: list[str] = []
            self._vecs: list[np.ndarray] = []
            self._inserted: set[str] = set()
            self._M: np.ndarray | None = None
            self._bm25: BM25 | None = None

        def insert(self, trajectory: dict[str, object]) -> None:
            tid = str(trajectory.get("id"))
            if tid in self._inserted:
                return
            self._inserted.add(tid)
            chunks = trajectory_chunks(
                trajectory, win=int(self.p["win"]), overlap=int(self.p["overlap"]),
                max_windows=int(self.p["max_windows"]))
            if not chunks:
                return
            vecs = _embed(chunks, str(self.p["embed_url"]), str(self.p["embed_model"]),
                          int(self.p["embed_batch"]), is_query=False)
            self._texts.extend(chunks)
            self._vecs.extend(vecs)
            self._M = None  # invalidate lazily-built index
            self._bm25 = None

        def _ensure_index(self) -> None:
            if self._M is None:
                self._M = np.asarray(self._vecs, dtype=np.float32) if self._vecs else np.zeros((0, 1), np.float32)
            if self._bm25 is None:
                self._bm25 = BM25(self._texts)

        def query(self, query: str, query_image: str | None = None
                  ) -> list["MemoryContextItem"]:
            self._ensure_index()
            q = clean_query(query)
            k = int(self.p["k"]); budget = int(self.p["char_budget"])
            pool = int(self.p["pool"]); rrf_k = int(self.p["rrf_k"])
            if self._M is not None and self._M.shape[0]:
                qv = _embed([q], str(self.p["embed_url"]), str(self.p["embed_model"]),
                            int(self.p["embed_batch"]), is_query=True)[0]
                dense_order = np.argsort(-(self._M @ qv))[:pool]
                bm25_order = np.argsort(-self._bm25.scores(q))[:pool]
                rrf: dict[int, float] = collections.defaultdict(float)
                for r, i in enumerate(dense_order):
                    rrf[int(i)] += 1.0 / (rrf_k + r)
                for r, i in enumerate(bm25_order):
                    rrf[int(i)] += 1.0 / (rrf_k + r)
                order = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])]
            else:
                order = []
            items: list[MemoryContextItem] = []
            used = 0
            for i in order:
                t = self._texts[i]
                if used + len(t) > budget and items:
                    continue
                items.append({"type": "text", "value": t})
                used += len(t)
                if len(items) >= k:
                    break
            if not items:  # harness requires non-empty text values; never return []
                items.append({"type": "text", "value": "(no relevant memory found)"})
            return items

        def _save_backend(self, output_dir: Path) -> None:
            return None

    _HAS_HARNESS = True
except ModuleNotFoundError:
    _HAS_HARNESS = False


# ----------------------------------------------------------------------------
# Standalone mechanism smoke test (no reader, no judge, no paid API).
# ----------------------------------------------------------------------------
def _smoke(data_root: Path, domain: str, n_traj: int, n_q: int) -> None:
    questions = [json.loads(l) for l in (data_root / "questions.jsonl").read_text().splitlines() if l.strip()]
    questions = [q for q in questions if q.get("domain") == domain]
    haystack = json.loads((data_root / "haystacks" / "lme_v2_small.json").read_text())
    want_ids: list[str] = []
    for q in questions:
        want_ids = haystack.get(q["id"], [])
        if want_ids:
            break
    subset = set(want_ids[:n_traj])
    print(f"[smoke] domain={domain} loading {len(subset)} of {len(want_ids)} haystack trajectories...")
    trajs: dict[str, dict] = {}
    with (data_root / "trajectories.jsonl").open() as fh:
        for line in fh:
            if not line.strip() or not (subset - set(trajs)):
                if not (subset - set(trajs)):
                    break
                continue
            if not any(tid in line[:80] for tid in subset - set(trajs)):
                continue
            row = json.loads(line)
            if row.get("id") in subset:
                trajs[row["id"]] = row

    p = {**DEFAULTS}
    texts: list[str] = []; vecs: list[np.ndarray] = []
    t0 = time.time()
    for tid, tr in trajs.items():
        ch = trajectory_chunks(tr, win=p["win"], overlap=p["overlap"], max_windows=p["max_windows"])
        v = _embed(ch, p["embed_url"], p["embed_model"], p["embed_batch"], is_query=False)
        texts.extend(ch); vecs.extend(v)
    M = np.asarray(vecs, dtype=np.float32); bm25 = BM25(texts)
    print(f"[smoke] ingest: {len(texts)} chunks in {time.time()-t0:.1f}s")

    lat = []; hits_gold = 0
    for q in questions[:n_q]:
        t0 = time.time()
        qc = clean_query(q["question"])
        qv = _embed([qc], p["embed_url"], p["embed_model"], p["embed_batch"], is_query=True)[0]
        do = np.argsort(-(M @ qv))[:p["pool"]]; bo = np.argsort(-bm25.scores(qc))[:p["pool"]]
        rrf: dict[int, float] = collections.defaultdict(float)
        for r, i in enumerate(do): rrf[int(i)] += 1.0 / (p["rrf_k"] + r)
        for r, i in enumerate(bo): rrf[int(i)] += 1.0 / (p["rrf_k"] + r)
        order = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])]
        picked = []; used = 0
        for i in order:
            t = texts[i]
            if used + len(t) > p["char_budget"] and picked: continue
            picked.append(t); used += len(t)
            if len(picked) >= p["k"]: break
        dt = time.time() - t0; lat.append(dt)
        blob = "\n".join(picked).lower()
        gold = q.get("answer"); gold_s = (gold[0] if isinstance(gold, list) else str(gold)).lower().strip()
        found = bool(gold_s) and gold_s in blob; hits_gold += found
        print(f"[smoke] q={q['id'][:10]} type={q['question_type'][:16]:16s} "
              f"query={dt*1000:6.1f}ms ctx={len(blob)}ch gold_in_ctx={found}")
    if lat:
        print(f"[smoke] query latency: mean={sum(lat)/len(lat)*1000:.1f}ms "
              f"gold-in-context {hits_gold}/{n_q} (partial-haystack proxy only)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--domain", default="enterprise", choices=["web", "enterprise"])
    ap.add_argument("--n-trajectories", type=int, default=8)
    ap.add_argument("--n-questions", type=int, default=5)
    args = ap.parse_args()
    _smoke(args.data_root, args.domain, args.n_trajectories, args.n_questions)
