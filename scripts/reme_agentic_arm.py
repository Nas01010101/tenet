"""Supplementary h2h arm: ReMe's OWN agentic answerer, end-to-end.

    PYTHONPATH=scripts python scripts/reme_agentic_arm.py --n 100 \
        --reme-venv scratchpad/reme-venv --workspace-root docs_scratch/reme_ws \
        --out docs_scratch/reme_agentic_n100.jsonl

Why this exists: the main harness (bench_reme_h2h.py) scores every arm through
the SAME single-shot reader so the memory system is the only variable. That
protocol deliberately does not exercise ReMe's own answering agent — a ReAct
loop with vector_search / bm25_search / python_execute / session-pivot tools
(reme.steps.benchmark.lme.agentic_answer, its own eval pipeline's job #4).
This script closes the obvious objection ("you nerfed ReMe's reader") by ALSO
running ReMe fully as-shipped on the SAME ingested workspaces: its own vector+
keyword index (LME_EMBEDDING_STORE=default), its own agent, its own answer —
judged by the same qa_judge as every other arm. Same instance selection
(seed=0 slice) as the main harness; resume-safe via JSONL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench_reme_h2h import ANSWER_MODEL, DISTILL_MODEL_REME, EMBED_MODEL, guard_not_oracle, DATA
from lme_recall import qa_judge
from reme_h2h_driver import reme_env, reme_run_job
from reme_preingest import load_instances


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reme-venv", required=True)
    ap.add_argument("--reme-config", default=str(Path(__file__).parent / "reme_h2h_config.yaml"))
    ap.add_argument("--workspace-root", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    guard_not_oracle(DATA)
    instances = load_instances(args.n, args.seed, "")
    root = Path(args.workspace_root)
    reme_bin = str(Path(args.reme_venv) / "bin" / "reme")
    env = reme_env(embed_model=EMBED_MODEL, answer_model=ANSWER_MODEL,
                   distill_model=DISTILL_MODEL_REME)
    # The one deviation from the main harness env: enable ReMe's own vector
    # store so its agent gets the full vector+BM25 retrieval it ships with.
    env["LME_EMBEDDING_STORE"] = "default"

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in out.open() if l.strip()}
    t0 = time.time()
    with out.open("a") as f:
        for i, inst in enumerate(instances):
            qid = inst["question_id"]
            if qid in done:
                continue
            ws = root / qid
            row = {"qid": qid, "type": inst["question_type"]}
            try:
                # Rebuild the index WITH embeddings (idempotent; local +
                # embedding calls over ~dozens of notes, cheap).
                reme_run_job(reme_bin, args.reme_config, "update_index", ws, env,
                             timeout=args.timeout)
                reme_run_job(reme_bin, args.reme_config, "agentic_answer", ws, env,
                             timeout=args.timeout)
                mem = json.loads((ws / "mem_answer.json").read_text())
                pred = str(mem.get("answer", "") or "").strip()
                row["pred"] = pred
                row["ok"] = qa_judge(inst["question"], inst["answer"], pred) if pred else None
            except Exception as e:  # keep going; a failed question is a row, not a crash
                row["ok"], row["pred"], row["error"] = None, None, str(e)[:500]
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_done = len(done) + sum(1 for _ in [1])  # this row
            print(f"[{i + 1}/{len(instances)}] {qid} ok={row['ok']} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in out.open()]
    scored = [r for r in rows if r["ok"] is not None]
    k = sum(bool(r["ok"]) for r in scored)
    print(f"\nreme-agentic: {k}/{len(scored)} scored correct "
          f"({len(rows) - len(scored)} unscored/error)")


if __name__ == "__main__":
    main()
