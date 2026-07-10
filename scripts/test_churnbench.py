"""Deterministic tests for ChurnBench's generator + scorer (scripts/bench_churn.py) —
NO LLM, no network. Fixed seed -> identical dataset hash; scorer correctness on
hand-made cases.

Run: python scripts/test_churnbench.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_churn import (  # noqa: E402
    ATTR_SPECS, CHUNK_SIZE, DISTRACTORS, _assert_no_substring_collisions,
    build_dataset, churn_half_life, score,
)

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    # --- determinism: same params/seed -> identical hash across independent calls ---
    d1 = build_dataset(updates_per_fact=8, n_facts=3, n_distractor_sessions=4, seed=42)
    d2 = build_dataset(updates_per_fact=8, n_facts=3, n_distractor_sessions=4, seed=42)
    check("same seed/params -> identical dataset hash", d1["hash"] == d2["hash"])
    check("same seed/params -> identical facts", d1["facts"] == d2["facts"])
    check("same seed/params -> identical questions", d1["questions"] == d2["questions"])

    # --- different seed -> (almost certainly) different hash ---
    d3 = build_dataset(updates_per_fact=8, n_facts=3, n_distractor_sessions=4, seed=43)
    check("different seed -> different dataset hash", d1["hash"] != d3["hash"])

    # --- different updates_per_fact -> different hash, different fact count ---
    d4 = build_dataset(updates_per_fact=4, n_facts=3, n_distractor_sessions=4, seed=42)
    check("different U -> different hash", d1["hash"] != d4["hash"])

    # --- structural invariants ---
    n_expected = 3 * 8 + 4 * 4  # n_facts*U + n_distractor_sessions*4 lines/block
    check("fact count matches n_facts*U + distractor lines",
          len(d1["facts"]) == n_expected, f"got {len(d1['facts'])}, want {n_expected}")
    check("questions count == n_facts", len(d1["questions"]) == 3)
    check("serials are 1..N contiguous",
          [s for s, _ in d1["facts"]] == list(range(1, len(d1["facts"]) + 1)))

    # sessions reconstruct the exact same chronological text sequence as facts
    flat_from_sessions = [line for chunk in d1["sessions"] for line in chunk]
    flat_from_facts = [t for _s, t in d1["facts"]]
    check("sessions concatenate back to the same order as facts",
          flat_from_sessions == flat_from_facts)
    check("session chunk sizes respect CHUNK_SIZE (last may be shorter)",
          all(len(c) <= CHUNK_SIZE for c in d1["sessions"]))

    # regression: no chunk may contain 2+ updates to the SAME attribute — a single
    # ingest_session/distill() call resolving a same-key conflict itself was measured
    # to sometimes pick the first (not final) value, corrupting tenet-arm supersession.
    def _attr_prefixes(n_facts):
        return {a: ATTR_SPECS[a]["update"].split("{v}")[0] for a in list(ATTR_SPECS)[:n_facts]}
    prefixes = _attr_prefixes(3)
    bad_chunk = None
    for chunk in d1["sessions"]:
        seen = set()
        for line in chunk:
            for attr, pfx in prefixes.items():
                if line.startswith(pfx):
                    if attr in seen:
                        bad_chunk = (chunk, attr)
                    seen.add(attr)
    check("no chunk contains two updates to the same attribute", bad_chunk is None,
          str(bad_chunk) if bad_chunk else "")

    # gold values are the LAST value in each attribute's chain, and each chain has
    # no repeated values (rng.shuffle + slice guarantees distinctness)
    for q in d1["questions"]:
        attr = q["attr"]
        chain_texts = [t for _s, t in d1["facts"]
                       if t.startswith(ATTR_SPECS[attr]["update"].split("{v}")[0][:8])]
        check(f"{attr}: gold is a non-empty value", bool(q["gold"].strip()))

    # --- no-substring-collision precondition holds for every attribute pool ---
    for attr, spec in ATTR_SPECS.items():
        try:
            _assert_no_substring_collisions(spec["pool"])
            check(f"no substring collisions in {attr} pool", True)
        except AssertionError as e:
            check(f"no substring collisions in {attr} pool", False, str(e))

    # --- pool sizes support the full sweep (U up to 32) ---
    for attr, spec in ATTR_SPECS.items():
        check(f"{attr} pool has >=32 values", len(spec["pool"]) >= 32,
              f"got {len(spec['pool'])}")

    # --- U larger than pool raises, doesn't silently truncate/repeat ---
    try:
        build_dataset(updates_per_fact=999, n_facts=1, n_distractor_sessions=0, seed=1)
        check("U > pool size raises ValueError", False, "did not raise")
    except ValueError:
        check("U > pool size raises ValueError", True)

    # --- n_facts beyond available attributes raises ---
    try:
        build_dataset(updates_per_fact=2, n_facts=99, n_distractor_sessions=0, seed=1)
        check("n_facts > available attrs raises ValueError", False, "did not raise")
    except ValueError:
        check("n_facts > available attrs raises ValueError", True)

    # =====================================================================
    # scorer — deterministic substring match, hand-made cases
    # =====================================================================
    check("exact match", score("Tokyo", "Tokyo"))
    check("case-insensitive", score("It's TOKYO now.", "Tokyo"))
    check("punctuation-robust", score("I believe it's Tokyo!", "Tokyo"))
    check("substring in a longer sentence", score("The current city is Tokyo, I think.", "Tokyo"))
    check("wrong value scores false", not score("Osaka", "Tokyo"))
    check("empty prediction scores false", not score("", "Tokyo"))
    check("multi-word gold, exact", score("senior analyst", "senior analyst"))
    check("multi-word gold, wrong level", not score("junior analyst", "senior analyst"))
    check("stale value does not accidentally match current (car pool)",
          not score("Honda Civic", "Honda Accord"))

    # --- churn_half_life ---
    sweep = [2, 4, 8, 16, 32]
    check("half-life picks largest passing U",
          churn_half_life({2: 1.0, 4: 1.0, 8: 0.95, 16: 0.5, 32: 0.3}, sweep) == 8)
    check("half-life: none pass -> '<min(sweep)>' sentinel",
          churn_half_life({2: 0.5, 4: 0.4, 8: 0.3, 16: 0.2, 32: 0.1}, sweep) == "<2")
    check("half-life: non-monotonic curve still takes the literal max passing U",
          churn_half_life({2: 1.0, 4: 0.5, 8: 1.0, 16: 0.5, 32: 0.5}, sweep) == 8)
    check("half-life: all pass -> largest U in sweep",
          churn_half_life({2: 1.0, 4: 1.0, 8: 1.0, 16: 1.0, 32: 1.0}, sweep) == 32)

    print(f"\n{len(FAILS)} failing checks" if FAILS else "\nall checks passed")
    for f in FAILS:
        print(f"  FAIL: {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
