"""Deterministic, LLM-free, torch-free test for tenet.navigate.navigate() and the
three public entry points wired on top of it (MemoryCore.navigate / Tenet.navigate /
the CLI `navigate` command / the MCP `navigate` tool).

Runs with NO network and NO torch: a hashed bag-of-words embedder is injected onto
the MemoryCore instance (overriding embed_batch, which both store() and recall() use),
giving fully controllable cosine geometry. This validates the *mechanism* — adaptive
associative descent + saturation stop — not the end-to-end LLM benchmark (see
scripts/bench_factcon.py + the spec for the FC-MH number-moving run).

Two falsifiable claims (mechanism, tested directly against tenet.navigate.navigate):
  1. MULTI-HOP REACH: a bridge fact that shares NO tokens with the query (so it is
     invisible to broad top-k recall) IS surfaced by navigate(), because an
     associative hop re-conditions the cue on an in-pool fact that DOES share a token
     with the bridge. Falsifier: navigate returns the same set as broad recall.
  2. EARLY STOP: a simple query whose evidence is fully in the broad pool makes
     navigate stop at hop 2 with a "saturated" trace entry (no over-fetch).
     Falsifier: navigate keeps descending to max_hops on a saturated query.

Plus wiring coverage (each entry point must reach the same underlying mechanism,
not a copy of it): MemoryCore.navigate, Tenet.navigate, cli.cmd_navigate,
mcp_server.navigate — called directly (no subprocess), asserting each returns/prints
the bridge fact for the same multi-hop setup as claim 1 above.

Run:  EMBED_PROVIDER=local python scripts/test_navigate.py   (exit 0 = pass)
"""
import argparse
import hashlib
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tenet.core import Tenet                  # noqa: E402
from tenet.memory import MemoryCore          # noqa: E402
from tenet.navigate import navigate          # noqa: E402

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)

_D = 64


def _fake_embed(texts):
    """Torch-free, network-free, process-stable hashed bag-of-words -> unit vectors.
    Shared tokens -> high cosine, so we can hand-build the recall geometry."""
    out = []
    for t in texts:
        v = np.zeros(_D, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _D
            v[h] += 1.0
        n = float(np.linalg.norm(v)) or 1.0
        out.append((v / n).astype(np.float32))
    return out


def _core():
    tmp = Path(tempfile.mkdtemp()) / "nav_test.db"
    c = MemoryCore(tmp)
    c.embed_batch = _fake_embed          # both store() and recall() route through this
    return c


def test_multihop_reach():
    c = _core()
    # Query shares tokens {sport, play, steve, sax} with the anchor fact only.
    query = "which sport does steve sax play"
    # Anchor: matches the query AND contains the bridge token "baseball".
    c.store("steve sax plays the sport baseball", key="steve_sax::sport")
    # Bridge: shares "baseball" with the anchor but NOTHING with the query.
    bridge_id = c.store("baseball originated in the country america",
                        key="baseball::origin")
    # Distractors: each shares a query token so they outrank the bridge in broad recall.
    c.store("maria plays the sport tennis", key="maria::sport")
    c.store("the sport of chess needs no play area", key="chess::info")
    c.store("steve enjoys play with his dog", key="steve::hobby")
    c.store("sax is a musical instrument you play", key="sax::music")

    broad = c.recall(query, k=5)
    broad_ids = {m.id for m in broad}
    assert bridge_id not in broad_ids, "bridge should be INVISIBLE to broad recall"

    nav, trace = navigate(c, query, k=5, max_hops=3, tau_gain=0.10)
    nav_ids = {m.id for m in nav}
    assert bridge_id in nav_ids, (
        f"navigate must surface the bridge via an associative hop; trace={trace}")
    assert any(t.get("adopted") and t["hop"] > 1 for t in trace), \
        f"a deeper hop must have been adopted; trace={trace}"
    print("PASS multihop_reach: bridge reached at hop >1")
    print("      trace:", trace)
    c.close()


def test_early_stop():
    c = _core()
    query = "what is the capital of france"
    c.store("the capital of france is paris", key="france::capital")
    # Unrelated facts: share no tokens with the query, so deeper hops find nothing
    # relevant to adopt -> navigate must stop early.
    for i, txt in enumerate([
        "the moon orbits the earth", "water boils at temperature",
        "guitars have six strings", "penguins live in cold regions",
    ]):
        c.store(txt, key=f"misc::{i}")

    nav, trace = navigate(c, query, k=5, max_hops=4, tau_gain=0.15)
    assert trace[-1].get("stop") == "saturated", f"expected early saturation; trace={trace}"
    assert trace[-1]["hop"] <= 3, f"should stop well before max_hops=4; trace={trace}"
    assert any(m.text == "the capital of france is paris" for m in nav), \
        "the answer fact must still be present"
    print("PASS early_stop: stopped at hop", trace[-1]["hop"], "-", trace[-1]["stop"])
    print("      trace:", trace)
    c.close()


def _seed_bridge(c) -> tuple[str, int]:
    """Same multi-hop bridge fixture as test_multihop_reach, factored out so the
    wiring tests below exercise an identical scenario through each entry point."""
    query = "which sport does steve sax play"
    c.store("steve sax plays the sport baseball", key="steve_sax::sport")
    bridge_id = c.store("baseball originated in the country america", key="baseball::origin")
    c.store("maria plays the sport tennis", key="maria::sport")
    c.store("the sport of chess needs no play area", key="chess::info")
    c.store("steve enjoys play with his dog", key="steve::hobby")
    c.store("sax is a musical instrument you play", key="sax::music")
    return query, bridge_id


def test_memorycore_navigate_entry():
    """MemoryCore.navigate() must reach the same result as calling the module
    function directly — not a divergent reimplementation."""
    c = _core()
    query, bridge_id = _seed_bridge(c)
    nav, trace = c.navigate(query, k=5, max_hops=3, tau_gain=0.10)
    check("MemoryCore.navigate surfaces the bridge fact",
          bridge_id in {m.id for m in nav}, f"trace={trace}")
    c.close()


def test_tenet_navigate_entry():
    """Tenet.navigate() (the top-level `from tenet import Tenet` surface) must
    delegate through to the same mechanism."""
    tmp = Path(tempfile.mkdtemp()) / "nav_tenet_test.db"
    t = Tenet(tmp)
    t.core.embed_batch = _fake_embed
    query, bridge_id = _seed_bridge(t.core)
    nav, trace = t.navigate(query, k=5, max_hops=3, tau_gain=0.10)
    check("Tenet.navigate surfaces the bridge fact",
          bridge_id in {m.id for m in nav}, f"trace={trace}")
    t.close()


def test_cli_navigate_entry(capsys):
    """`tenet navigate <query>` (cli.cmd_navigate) must render the bridge fact
    through the CLI's own output path, called directly (no subprocess).

    cmd_navigate opens its own Tenet(db) internally (can't inject a fake
    embedder onto that instance), so the fixture is seeded with the SAME
    embedder cmd_navigate will use: `config.embed_texts` is monkeypatched to
    the hashed bag-of-words embedder for the duration of this test only.
    """
    from tenet import cli as tenet_cli, config

    orig_embed = config.embed_texts
    config.embed_texts = _fake_embed
    try:
        tmp = Path(tempfile.mkdtemp()) / "nav_cli_test.db"
        c = MemoryCore(tmp)
        query, _bridge_id = _seed_bridge(c)
        c.close()

        args = argparse.Namespace(db=str(tmp), query=query, k=5, max_hops=3, tau_gain=0.10)
        rc = tenet_cli.cmd_navigate(args)
        out = capsys.readouterr().out if capsys else None
        check("cli.cmd_navigate returns 0", rc == 0, rc)
        if out is not None:  # only when run under pytest (capsys fixture available)
            check("cli.cmd_navigate prints the bridge fact",
                  "baseball originated in the country america" in out, out)
    finally:
        config.embed_texts = orig_embed


def test_mcp_navigate_tool():
    """The MCP `navigate` tool function, called directly (no MCP transport), must
    surface the bridge fact and report the hop count in its text output."""
    from tenet import mcp_server

    tmp = Path(tempfile.mkdtemp()) / "nav_mcp_test.db"
    mcp_server._core.close()
    mcp_server._core = MemoryCore(tmp)
    mcp_server._core.embed_batch = _fake_embed
    query, _bridge_id = _seed_bridge(mcp_server._core)

    out = mcp_server.navigate(query, budget=3)
    check("mcp navigate tool surfaces the bridge fact",
          "baseball originated in the country america" in out, out)
    check("mcp navigate tool reports hop count", "navigated" in out and "hop" in out, out)
    mcp_server._core.close()


if __name__ == "__main__":
    test_multihop_reach()
    test_early_stop()
    test_memorycore_navigate_entry()
    test_tenet_navigate_entry()
    test_cli_navigate_entry(capsys=None)
    test_mcp_navigate_tool()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        raise SystemExit(1)
    print("\nALL PASS")
