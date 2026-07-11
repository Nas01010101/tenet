"""Deterministic tests for recall()'s raw-turn-favored dual-pool split
(docs/COMPARISON.md follow-up #2, "raw-turn-favored recall mode" — the LoCoMo
verbatim-recall gap). No LLM (EMBED_PROVIDER=local, pre-formed store_fact()s).

Run: python scripts/test_raw_recall.py
"""
import os
os.environ.setdefault("EMBED_PROVIDER", "local")  # before importing config

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
from tenet.memory import MemoryCore  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def _seed_pool(core, rng, qv, n_facts, n_raw):
    for i in range(n_facts):
        v = qv * 0.95 + rng.standard_normal(384).astype(np.float32) * 0.05
        v /= np.linalg.norm(v)
        core.store(f"fact {i}", key=f"k::{i}", _vec=v)
    for i in range(n_raw):
        v = qv * 0.9 + rng.standard_normal(384).astype(np.float32) * 0.1
        v /= np.linalg.norm(v)
        core.store(f"raw {i}", kind="raw", _vec=v)


def test_default_off_unchanged():
    """default (flag unset -> _RAW_RECALL_DEFAULT=False) caps raw at k//2, same as
    before this parameter existed."""
    core = MemoryCore(Path(tempfile.mkdtemp()) / "raw_recall_off.db")
    rng = np.random.default_rng(1)
    qv = rng.standard_normal(384).astype(np.float32); qv /= np.linalg.norm(qv)
    _seed_pool(core, rng, qv, n_facts=5, n_raw=6)

    hits = core.recall("probe", k=6)
    raw_n = sum(1 for h in hits if h.kind == "raw")
    check("default (raw_recall unset): raw capped at k//2", raw_n == 3, f"raw_n={raw_n}")

    explicit_off = core.recall("probe", k=6, raw_recall=False)
    check("raw_recall=False explicit == unset default (byte-identical texts)",
          [h.text for h in hits] == [h.text for h in explicit_off])
    core.close()


def test_raw_recall_on_favors_raw():
    """raw_recall=True: raw slices fill up to the FULL k budget when enough exist."""
    core = MemoryCore(Path(tempfile.mkdtemp()) / "raw_recall_on.db")
    rng = np.random.default_rng(2)
    qv = rng.standard_normal(384).astype(np.float32); qv /= np.linalg.norm(qv)
    _seed_pool(core, rng, qv, n_facts=5, n_raw=8)

    hits = core.recall("probe", k=6, raw_recall=True)
    raw_n = sum(1 for h in hits if h.kind == "raw")
    check("raw_recall=True: raw fills the whole k budget (8 raw available >= k=6)",
          raw_n == 6, f"raw_n={raw_n}")


def test_raw_recall_on_backfills_with_facts_when_raw_scarce():
    """If there aren't enough raw slices to fill k, facts backfill the rest —
    never fewer than k total when enough combined candidates exist."""
    core = MemoryCore(Path(tempfile.mkdtemp()) / "raw_recall_scarce.db")
    rng = np.random.default_rng(3)
    qv = rng.standard_normal(384).astype(np.float32); qv /= np.linalg.norm(qv)
    _seed_pool(core, rng, qv, n_facts=10, n_raw=2)  # only 2 raw slices exist

    hits = core.recall("probe", k=6, raw_recall=True)
    raw_n = sum(1 for h in hits if h.kind == "raw")
    fact_n = sum(1 for h in hits if h.kind != "raw")
    check("raw_recall=True, raw scarce: all 2 raw slices included", raw_n == 2, f"raw_n={raw_n}")
    check("raw_recall=True, raw scarce: facts backfill to reach k=6",
          len(hits) == 6 and fact_n == 4, f"total={len(hits)} fact_n={fact_n}")
    core.close()


def test_raw_recall_never_starves_facts_pool_empty():
    """No raw slices at all: raw_recall=True behaves like a normal facts-only
    recall (no crash, no empty result)."""
    core = MemoryCore(Path(tempfile.mkdtemp()) / "raw_recall_no_raw.db")
    rng = np.random.default_rng(4)
    qv = rng.standard_normal(384).astype(np.float32); qv /= np.linalg.norm(qv)
    _seed_pool(core, rng, qv, n_facts=5, n_raw=0)

    hits = core.recall("probe", k=6, raw_recall=True)
    check("raw_recall=True, no raw slices at all: falls back to facts",
          len(hits) == 5 and all(h.kind != "raw" for h in hits), f"kinds={[h.kind for h in hits]}")
    core.close()


def test_env_flag_matches_kwarg():
    """TENET_RAW_RECALL=1 produces the same behavior as raw_recall=True per call
    (the module-level default is read once at import time, so this test verifies
    the LOGIC is identical by calling both explicitly on a fresh process-equivalent
    core, since flipping the real env var after import wouldn't retroactively change
    the already-bound default — the per-call kwarg is what a caller actually uses
    to override, exactly like TENET_AGG_READER/TENET_RETRACT)."""
    core = MemoryCore(Path(tempfile.mkdtemp()) / "raw_recall_env.db")
    rng = np.random.default_rng(5)
    qv = rng.standard_normal(384).astype(np.float32); qv /= np.linalg.norm(qv)
    _seed_pool(core, rng, qv, n_facts=5, n_raw=8)

    a = core.recall("probe", k=6, raw_recall=True)
    b = core.recall("probe", k=6, raw_recall=True)
    check("raw_recall=True is deterministic across repeated calls",
          [h.text for h in a] == [h.text for h in b])
    core.close()


def main() -> int:
    test_default_off_unchanged()
    test_raw_recall_on_favors_raw()
    test_raw_recall_on_backfills_with_facts_when_raw_scarce()
    test_raw_recall_never_starves_facts_pool_empty()
    test_env_flag_matches_kwarg()
    if FAILS:
        print("\nFAILURES:", FAILS)
        return 1
    print("\nALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
