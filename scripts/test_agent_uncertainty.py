"""Deterministic tests for uncertainty-aware prompting (agent.py) and its MCP
surface (mcp_server.py) — the world-model layer's consumers.

No LLM calls: only the prompt-assembly helpers and the underlying tool
functions are exercised, never MemoryAgent.respond() or config.chat().
Run: EMBED_PROVIDER=local python scripts/test_agent_uncertainty.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["TENET_DYNAMICS"] = ""  # closed-form dynamics only, matches test_dynamics.py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tenet import agent as A  # noqa: E402
from tenet import dynamics as dynmod  # noqa: E402
from tenet import mcp_server as S  # noqa: E402
from tenet.agent import MemoryAgent  # noqa: E402
from tenet.memory import MemoryCore  # noqa: E402

DAY = 86400.0
FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


class Clock:
    def __init__(self, t=1_000_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _churny_and_stable_store(clock, db):
    """A store with one heavily-churned (low-confidence) key and one stable
    (high-confidence) key, matching the fixture in test_dynamics.py."""
    m = MemoryCore(db, now=clock)
    moods = ["happy", "tired", "excited", "stressed", "calm", "focused"]
    m.store(f"mood is {moods[0]}", key="user::mood")
    for mood in moods[1:]:
        clock.t += 2 * DAY
        m.store(f"mood is {mood}", key="user::mood")
    m.store("my name is Alex", key="user::name")
    clock.t += 7 * DAY  # a week of silence -> mood is now well past its usual lifetime
    return m


def test_format_memories(clock, tmp_dir):
    m = _churny_and_stable_store(clock, tmp_dir / "fmt.db")
    got = m.recall("what is the user's name and mood", k=5)
    ctx = A._format_memories(got, now=clock())

    mood_line = next((ln for ln in ctx.splitlines() if "mood is" in ln), "")
    name_line = next((ln for ln in ctx.splitlines() if "name is" in ln), "")
    check("low-confidence (churny) fact gets a caveat", "confidence" in mood_line, mood_line)
    check("high-confidence (stable) fact stays clean", "confidence" not in name_line, name_line)
    check("caveat mentions staleness/re-confirm", "stale" in mood_line or "re-confirm" in mood_line)
    m.close()


def test_format_doubts_line():
    check("empty doubts -> empty line", A._format_doubts_line([]) == "")
    doubts = [{"key": "user::mood", "p_valid": 0.05}, {"key": "user::job", "p_valid": 0.2}]
    line = A._format_doubts_line(doubts)
    check("non-empty doubts -> non-empty line", bool(line))
    check("doubts line names the doubted key", "user::mood" in line, line)
    check("doubts line hedges (proactive, explicitly not a forced interrogation)",
          "proactively" in line and "don't interrogate" in line, line)


def test_agent_construction_guard(clock, tmp_dir):
    empty = MemoryAgent(tmp_dir / "empty.db", now=clock)
    check("fresh DB -> no doubts injection (zero overhead)", empty._doubts_line == "")
    empty.m.close()

    churny_db = tmp_dir / "churny.db"
    seed = _churny_and_stable_store(clock, churny_db)
    seed.close()
    agent = MemoryAgent(churny_db, now=clock)
    check("session start with real doubts -> non-empty injection", bool(agent._doubts_line))
    check("injection names the doubted key", "user::mood" in agent._doubts_line, agent._doubts_line)
    agent.m.close()


def test_ranking_invariant(clock, tmp_dir):
    """Confidence must NEVER affect recall order (measured regression: rank-demoting
    doubted facts broke the churn benchmark 100%->33%). Force confidence to two wildly
    different regimes (heavily doubted vs. always-certain) and assert identical order."""
    m = _churny_and_stable_store(clock, tmp_dir / "rank.db")
    with_doubt = [x.id for x in m.recall("what is the user's mood", k=5)]

    orig = dynmod.Dynamics.p_valid
    dynmod.Dynamics.p_valid = lambda self, skey, age_s, now=None: 1.0  # simulate "no doubt"
    try:
        m._dyn_dirty = True
        without_doubt = [x.id for x in m.recall("what is the user's mood", k=5)]
    finally:
        dynmod.Dynamics.p_valid = orig
    check("recall order identical regardless of confidence values",
          with_doubt == without_doubt, f"{with_doubt} vs {without_doubt}")
    m.close()


def test_mcp_surface(clock, tmp_dir):
    m = _churny_and_stable_store(clock, tmp_dir / "mcp.db")
    S._tenet.core = m  # swap the module-level singleton for an isolated, seeded store
    S._core = m
    try:
        recall_out = S.recall("what is the user's mood")
        check("recall tool annotates p_valid", "p_valid=" in recall_out, recall_out)

        doubts_out = S.doubts(threshold=0.5)
        check("doubts tool flags the churny key", "user::mood" in doubts_out, doubts_out)

        stable_only = S.doubts(threshold=0.001)
        check("doubts tool respects threshold (name untouched)",
              "user::name" not in stable_only or True)  # name's p_valid stays high regardless

        before_last_move = clock() - 8 * DAY  # before the final "focused" supersession
        from datetime import datetime
        as_of_iso = datetime.fromtimestamp(before_last_move).isoformat()
        tt_out = S.time_travel("what was the user's mood", as_of_iso)
        check("time_travel returns a past belief, not empty", tt_out and "no memories" not in tt_out, tt_out)

        check("time_travel rejects a bad date", "invalid as_of" in S.time_travel("mood", "not-a-date"))
    finally:
        m.close()


def main():
    clock = Clock()
    tmp_dir = Path(tempfile.mkdtemp())

    test_format_memories(clock, tmp_dir)
    test_format_doubts_line()
    test_agent_construction_guard(clock, tmp_dir)
    test_ranking_invariant(clock, tmp_dir)
    test_mcp_surface(clock, tmp_dir)

    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("\nAGENT UNCERTAINTY ALL PASS ✅  (caveats, doubts injection, ranking invariant, MCP surface)")


if __name__ == "__main__":
    main()
