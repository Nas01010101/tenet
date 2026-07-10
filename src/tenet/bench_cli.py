"""tenet bench — thin dispatcher over the scripts/bench_*.py source-of-truth harnesses.

Design: this module does NOT reimplement any benchmark. Each `tenet bench run <name>`
shells out to the exact scripts/*.py the paper numbers come from, forwarding any extra
flags verbatim, and appends a one-line JSON record (timestamp/config/git-sha/exit) to
data/bench_runs.jsonl. `tenet bench results` renders that log. Keeps the scripts the
single source of truth — the CLI only wires env + args and captures provenance.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = _REPO / "scripts"
_RUNS_LOG = _REPO / "data" / "bench_runs.jsonl"

# name -> (script, one-line description, docs/paper section it reproduces)
BENCHMARKS: dict[str, tuple[str, str, str]] = {
    "churn": (
        "bench_horizon.py",
        "Long-horizon knowledge churn — RAG collapses as a fact is updated N times, Tenet holds.",
        "BENCHMARK.md §3 / README Fig. 1",
    ),
    "knowledge-update": (
        "bench_knowledge_update.py",
        "Controlled supersession: current-correct vs stale-leak on facts that change over sessions.",
        "BENCHMARK.md §4",
    ),
    "lme-recall": (
        "lme_recall.py",
        "LongMemEval_S retrieval recall@k + QA answer-accuracy/token frontier (--qa).",
        "BENCHMARK.md §1-2",
    ),
    "factcon": (
        "bench_factcon.py",
        "MemoryAgentBench FactConsolidation (SubEM, official reader) — standardized supersession.",
        "BENCHMARK.md §6 (arXiv:2507.05257)",
    ),
    "mab-ar": (
        "bench_mab_ar.py",
        "MemoryAgentBench Accurate-Retrieval: RULER-QA / LongMemEval(S*) / EventQA.",
        "BENCHMARK.md §7 (arXiv:2507.05257)",
    ),
    "lmev2": (
        "lmev2_adapter.py",
        "LongMemEval-V2 backend adapter — zero-LLM read-path mechanism smoke test.",
        "scripts/lmev2_adapter.py docstring",
    ),
    "churnbench": (
        "bench_churn.py",
        "Parametric high-churn benchmark — updates-per-fact dial, 4 arms "
        "(tenet/rag/mem0/hipporag), churn half-life headline metric.",
        "BENCHMARK.md §9 (ChurnBench)",
    ),
}

# --provider presets -> env overrides (keyless local paths for cheap reproduction).
_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "qwen": {},  # defaults (Qwen Cloud) — needs DASHSCOPE_API_KEY
    "local": {"EMBED_PROVIDER": "local"},  # keyless embeddings; LLM left as-is
    "ollama": {"EMBED_PROVIDER": "local", "LLM_PROVIDER": "ollama",
               "OLLAMA_MODEL": "qwen2.5:7b"},  # fully keyless end-to-end
    "openrouter": {"EMBED_PROVIDER": "local", "LLM_PROVIDER": "openrouter"},
}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True, timeout=5).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        return "?"


def _forwarded_args(args, extra: list[str]) -> list[str]:
    """Map the recognized convenience flags to their script arg + append passthrough.

    Every declared flag is only emitted when the user set it, so a benchmark that
    doesn't accept it (e.g. churn has no --qpc) only errors if the user actually asks.
    Unknown flags are forwarded verbatim via `extra` (parse_known_args)."""
    fwd: list[str] = []
    for flag, val in (("--qpc", args.qpc), ("--cells", args.cells),
                      ("--k", args.k), ("--seed", args.seed),
                      ("--principals", args.principals)):
        if val is not None:
            fwd += [flag, str(val)]
    return fwd + list(extra)


def cmd_bench_list(args, out, _err) -> int:
    from . import cli
    rows = [(n, s[1], s[2]) for n, s in BENCHMARKS.items()]
    if cli._RICH:
        from rich.table import Table
        t = Table(show_header=True, header_style="bold")
        t.add_column("benchmark", style="cyan"); t.add_column("what it measures", overflow="fold")
        t.add_column("reproduces", style="dim")
        for n, desc, ref in rows:
            t.add_row(n, desc, ref)
        cli._console.print(t)
    else:
        for n, desc, ref in rows:
            out(f"{n:<17} {desc}\n{'':<17} ({ref})")
    return 0


def cmd_bench_run(args, extra, out, err) -> int:
    spec = BENCHMARKS.get(args.name)
    if spec is None:
        err(f"unknown benchmark {args.name!r}. Try: tenet bench list")
        return 2
    script, desc, ref = spec
    script_path = _SCRIPTS / script
    if not script_path.exists():
        err(f"benchmark script missing: {script_path}")
        return 1

    env = dict(os.environ)
    preset = _PROVIDER_PRESETS.get(args.provider, {}) if args.provider else {}
    env.update(preset)
    for kv in args.env or []:
        if "=" not in kv:
            err(f"--env expects KEY=VAL, got {kv!r}"); return 2
        key, _, val = kv.partition("=")
        env[key] = val

    cmd = [sys.executable, str(script_path)] + _forwarded_args(args, extra)
    env_shown = {**preset, **{k.split("=")[0]: k.split("=", 1)[1] for k in (args.env or [])}}
    env_prefix = " ".join(f"{k}={v}" for k, v in env_shown.items())
    pretty = (env_prefix + " " if env_prefix else "") + " ".join(cmd)

    if args.dry_run:
        out(f"[dry-run] {args.name} → {ref}")
        out(pretty)
        return 0

    out(f"running {args.name} ({ref})")
    out(pretty)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_REPO, env=env)  # stream child stdout/stderr live
    dt = time.time() - t0

    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": args.name, "script": script, "git_sha": _git_sha(),
        "provider": args.provider, "args": _forwarded_args(args, extra),
        "env": env_shown, "exit_code": proc.returncode, "seconds": round(dt, 1),
    }
    _RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _RUNS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    out(f"{'ok' if proc.returncode == 0 else 'FAILED'} in {dt:.1f}s "
        f"· logged to {_RUNS_LOG.relative_to(_REPO)}")
    return proc.returncode


def cmd_bench_results(args, out, err) -> int:
    from . import cli
    if not _RUNS_LOG.exists():
        out("(no runs yet — try: tenet bench run churn --provider ollama --principals 2)")
        return 0
    records = []
    for line in _RUNS_LOG.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records = records[-args.limit:]
    if cli._RICH:
        from rich.table import Table
        t = Table(show_header=True, header_style="bold")
        for c in ("when", "benchmark", "sha", "provider", "args", "sec", "exit"):
            t.add_column(c, overflow="fold", style="dim" if c in ("sha", "args") else None)
        for r in records:
            t.add_row(r.get("ts", "?").replace("T", " ").replace("+00:00", ""),
                      r.get("benchmark", "?"), r.get("git_sha", "?"),
                      r.get("provider") or "-", " ".join(r.get("args", [])) or "-",
                      str(r.get("seconds", "?")),
                      str(r.get("exit_code", "?")))
        cli._console.print(t)
    else:
        for r in records:
            out(f"{r.get('ts','?')}  {r.get('benchmark','?'):<16} sha={r.get('git_sha','?')} "
                f"provider={r.get('provider') or '-'} exit={r.get('exit_code','?')} "
                f"{r.get('seconds','?')}s  {' '.join(r.get('args', []))}")
    return 0
