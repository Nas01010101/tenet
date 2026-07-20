"""`tenet timeline` / `tenet export` — the bi-temporal belief ledger, rendered.

The whole project's mechanism is *supersession is invisible until you go
looking* (bi-temporal history, but `recall()` only ever shows you "now").
These two commands make it a first-class artifact: one query layer
(`MemoryCore.list_beliefs()`, the same call `/state` and `get_all()` use) and
two renderers — `timeline` for a human glancing at a terminal, `export` for a
machine. Split out of cli.py to keep that file under the repo's 500-line cap.
"""
from __future__ import annotations

import json
from datetime import datetime


def _fmt(ts: float | None) -> str:
    return "-" if ts is None else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _group_by_key(rows: list[dict]) -> dict[str, list[dict]]:
    # list_beliefs() already returns rows ORDER BY skey, valid_at — grouping
    # here just buckets that existing order, no re-sort needed.
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["key"], []).append(r)
    return groups


def cmd_timeline(args, out, err) -> int:
    from . import cli
    m = cli._open(args.db)
    try:
        rows = m.core.list_beliefs()  # as_of=None: full ledger (current + superseded)
    finally:
        m.close()

    if args.key:
        rows = [r for r in rows if r["key"] == args.key]
        if not rows:
            if args.json:
                out("[]")
                return 0
            err(f"no beliefs found for key {args.key!r} — run `tenet timeline` "
                f"with no key to see all current keys")
            return 1
    elif not args.all:
        rows = [r for r in rows if r["status"] == "current"]

    if not rows:
        if args.json:
            out("[]")
        else:
            out("(empty — no beliefs stored yet. `tenet remember \"...\"` to add one)")
        return 0

    if args.json:
        out(json.dumps(rows, indent=2))
        return 0

    groups = _group_by_key(rows)
    if cli._RICH:
        _render_rich(cli._console, groups)
    else:
        _render_plain(out, groups)
    return 0


def _render_rich(console, groups: dict[str, list[dict]]) -> None:
    from rich.text import Text
    for key, entries in groups.items():
        console.print(Text(key, style="bold cyan"))
        for e in entries:
            current = e["status"] == "current"
            marker = "●" if current else "○"
            # markup=False equivalent: Text(str) never parses its content as markup,
            # so arbitrary/LLM-produced belief text (e.g. "[mcp]") renders verbatim —
            # same reasoning as cli.py's cmd_recall table.
            line = Text(f"  {marker} ", style="bold green" if current else "dim")
            line.append(e["text"], style="bold green" if current else "dim strike")
            line.append(f"  (valid_at={_fmt(e['valid_at'])}, created_at={_fmt(e['created_at'])}, "
                         f"src={e['source'] or '-'})", style="dim")
            if not current:
                line.append("  [superseded]", style="dim")
            console.print(line)
        console.print()


def _render_plain(out, groups: dict[str, list[dict]]) -> None:
    for key, entries in groups.items():
        out(key)
        for e in entries:
            tag = "current" if e["status"] == "current" else "superseded"
            out(f"  [{tag}] {e['text']}  (valid_at={_fmt(e['valid_at'])}, "
                f"created_at={_fmt(e['created_at'])}, src={e['source'] or '-'})")
        out("")


def cmd_export(args, out, err) -> int:
    from . import cli
    m = cli._open(args.db)
    try:
        rows = m.core.list_beliefs()  # full ledger: current + superseded, for audit
    finally:
        m.close()
    # --json: compact single-line (piping to jq/other tools); default: indent=2
    # for a human reading it straight off stdout. Same rows either way.
    out(json.dumps(rows) if args.json else json.dumps(rows, indent=2))
    return 0
