"""Deterministic tests for `tenet timeline` / `tenet export` / `tenet --version`
(src/tenet/cli_timeline.py + the argparse wiring in src/tenet/cli.py).

No LLM, no network: EMBED_PROVIDER=local, facts are seeded directly via
`Tenet.store_fact()` (bypasses distillation entirely — same pattern as
scripts/test_langgraph_store.py's TenetStore seeding). Covers: the
supersession chain rendering (current highlighted, superseded marked),
the default "current-only" vs `--all`/`KEY` "full history" views, `--json`
round-tripping through `json.loads`, an empty db producing a friendly
message instead of a crash, and `export`'s always-JSON dump.

Run: EMBED_PROVIDER=local python scripts/test_cli_timeline.py
"""
import argparse
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("EMBED_PROVIDER", "local")  # before importing tenet.*

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tenet import cli, cli_timeline  # noqa: E402
from tenet.core import Tenet  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def _timeline_args(db, key=None, all_=False, json_=False):
    return argparse.Namespace(db=str(db), key=key, all=all_, json=json_)


def _export_args(db, json_=False):
    return argparse.Namespace(db=str(db), json=json_)


def _run(fn, args):
    """Call a cmd_* function with list-collecting out/err, return (rc, out_lines, err_lines)."""
    out, err = [], []
    rc = fn(args, out.append, err.append)
    return rc, out, err


def test_empty_db():
    db = Path(tempfile.mkdtemp()) / "empty.db"

    rc, out, _ = _run(cli_timeline.cmd_timeline, _timeline_args(db))
    check("empty db: timeline returns 0", rc == 0, rc)
    check("empty db: timeline prints a friendly message, not a traceback",
          out and "empty" in out[0].lower(), out)

    rc, out, _ = _run(cli_timeline.cmd_timeline, _timeline_args(db, json_=True))
    check("empty db: timeline --json returns 0", rc == 0, rc)
    check("empty db: timeline --json is valid (empty) JSON", json.loads(out[0]) == [], out)

    rc, out, _ = _run(cli_timeline.cmd_export, _export_args(db))
    check("empty db: export returns 0", rc == 0, rc)
    check("empty db: export is valid (empty) JSON", json.loads(out[0]) == [], out)


def _seed(db) -> Tenet:
    m = Tenet(db)
    m.store_fact("I live in Boston", key="user::residence", source="chat")
    m.store_fact("I live in Seattle", key="user::residence", source="chat")  # supersedes
    m.store_fact("I like coffee", key="user::preference", source="onboarding")
    return m


def test_timeline_default_shows_current_only():
    db = Path(tempfile.mkdtemp()) / "seeded.db"
    m = _seed(db); m.close()

    orig_rich = cli._RICH
    cli._RICH = False  # force the plain-text renderer — deterministic, no ANSI
    try:
        rc, out, _ = _run(cli_timeline.cmd_timeline, _timeline_args(db))
        text = "\n".join(out)
        check("default timeline returns 0", rc == 0, rc)
        check("default timeline shows the CURRENT value", "Seattle" in text, text)
        check("default timeline hides the superseded value", "Boston" not in text, text)
        check("default timeline shows the other current key too", "coffee" in text, text)
    finally:
        cli._RICH = orig_rich


def test_timeline_key_shows_full_chain():
    db = Path(tempfile.mkdtemp()) / "seeded.db"
    m = _seed(db); m.close()

    orig_rich = cli._RICH
    cli._RICH = False
    try:
        rc, out, _ = _run(cli_timeline.cmd_timeline, _timeline_args(db, key="user::residence"))
        text = "\n".join(out)
        check("KEY timeline returns 0", rc == 0, rc)
        check("KEY timeline shows the current value", "Seattle" in text, text)
        check("KEY timeline shows the superseded value too", "Boston" in text, text)
        check("KEY timeline marks the superseded row", "[superseded]" in text, text)
        check("KEY timeline marks the current row", "[current]" in text, text)

        rc, out, _ = _run(cli_timeline.cmd_timeline, _timeline_args(db, all_=True))
        text = "\n".join(out)
        check("--all timeline (no KEY) also surfaces the superseded value",
              "Boston" in text and "Seattle" in text, text)
    finally:
        cli._RICH = orig_rich


def test_timeline_unknown_key():
    db = Path(tempfile.mkdtemp()) / "seeded.db"
    m = _seed(db); m.close()
    rc, _, err = _run(cli_timeline.cmd_timeline, _timeline_args(db, key="no::such"))
    check("unknown key returns nonzero", rc != 0, rc)
    check("unknown key prints a helpful message", err and "no::such" in err[0], err)


def test_timeline_json():
    db = Path(tempfile.mkdtemp()) / "seeded.db"
    m = _seed(db); m.close()

    rc, out, _ = _run(cli_timeline.cmd_timeline, _timeline_args(db, all_=True, json_=True))
    check("timeline --all --json returns 0", rc == 0, rc)
    rows = json.loads(out[0])
    check("timeline --json parses to a list", isinstance(rows, list), rows)
    check("timeline --json has all 3 belief rows (2 residence history + 1 preference)",
          len(rows) == 3, rows)
    expected_keys = {"id", "key", "text", "valid_at", "created_at", "expired_at",
                      "source", "status", "p_valid"}
    check("timeline --json rows carry provenance (created_at/source)",
          all(expected_keys <= set(r.keys()) for r in rows), rows)
    check("timeline --json: exactly one 'current' row per key",
          sum(1 for r in rows if r["status"] == "current") == 2, rows)
    check("timeline --json: source is preserved",
          any(r["source"] == "onboarding" for r in rows), rows)


def test_export():
    db = Path(tempfile.mkdtemp()) / "seeded.db"
    m = _seed(db); m.close()

    rc, out, _ = _run(cli_timeline.cmd_export, _export_args(db))
    check("export (pretty) returns 0", rc == 0, rc)
    pretty = "\n".join(out)
    rows = json.loads(pretty)
    check("export dumps current + superseded (3 rows)", len(rows) == 3, rows)
    check("export includes the superseded row", any(r["status"] == "superseded" for r in rows), rows)
    check("export is pretty-printed by default (multi-line)", "\n" in pretty, repr(pretty[:80]))

    rc, out, _ = _run(cli_timeline.cmd_export, _export_args(db, json_=True))
    check("export --json returns 0", rc == 0, rc)
    compact = out[0]
    check("export --json is single-line (compact)", "\n" not in compact, repr(compact[:80]))
    check("export --json round-trips the same rows", len(json.loads(compact)) == 3, compact)


def test_cli_main_dispatch():
    """Exercise the real argparse wiring (KEY nargs='?', --all, --json, --db),
    not just cmd_timeline() called directly — catches parser-plumbing bugs."""
    db = Path(tempfile.mkdtemp()) / "seeded.db"
    m = _seed(db); m.close()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["timeline", "--all", "--json", "--db", str(db)])
    check("cli.main(['timeline', ...]) returns 0", rc == 0, rc)
    rows = json.loads(buf.getvalue())
    check("cli.main timeline --json parses and has 3 rows", len(rows) == 3, rows)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["export", "--db", str(db)])
    check("cli.main(['export', ...]) returns 0", rc == 0, rc)
    check("cli.main export output parses as JSON", len(json.loads(buf.getvalue())) == 3, buf.getvalue())


def test_version():
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            cli.main(["--version"])
            rc = 0  # argparse's "version" action normally raises SystemExit(0)
        except SystemExit as e:
            rc = e.code
    out = buf.getvalue().strip()
    check("--version exits 0", rc == 0, rc)
    check("--version output names the tool and a version string",
          out.startswith("tenet ") and len(out.split()) == 2, repr(out))


def main() -> int:
    test_empty_db()
    test_timeline_default_shows_current_only()
    test_timeline_key_shows_full_chain()
    test_timeline_unknown_key()
    test_timeline_json()
    test_export()
    test_cli_main_dispatch()
    test_version()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("\nCLI TIMELINE/EXPORT ALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
