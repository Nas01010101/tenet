"""Score a predictions JSONL (from bench_mab_ar.py --dump-preds) with MAB's
official LLM-judge (anscheck prompts verbatim, via judge_correct). Decouples
reading (RTX, always available) from judging (needs a strong model) so a quota
outage never loses reader work.

Usage: QWEN_JUDGE_MODEL=qwen-max python scripts/judge_preds.py preds.jsonl
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bench_factcon import wilson_ci             # noqa: E402
from bench_mab_ar import judge_correct          # noqa: E402


def main():
    path = sys.argv[1]
    out_path = path.replace(".jsonl", ".judged.jsonl")
    done = set()
    if Path(out_path).exists():                  # resumable: skip already-judged rows
        for line in open(out_path):
            done.add(json.loads(line)["q"][:200])
    out_f = open(out_path, "a")
    per_cell: dict[str, list[int]] = {}
    for line in open(path):
        r = json.loads(line)
        s = per_cell.setdefault(r["cell"], [0, 0, 0, 0])   # rag, tenet, n, err
        if r["q"][:200] in done:
            continue
        if not r["rag"].strip() or not r["tenet"].strip():
            s[3] += 1
            continue
        r_ok = judge_correct(r["q"], r["gold"], r["rag"], r["qtype"], r["abs"])
        t_ok = judge_correct(r["q"], r["gold"], r["tenet"], r["qtype"], r["abs"])
        if r_ok is None or t_ok is None:
            s[3] += 1
            continue
        s[0] += r_ok; s[1] += t_ok; s[2] += 1
        out_f.write(json.dumps(r | {"rag_ok": bool(r_ok), "tenet_ok": bool(t_ok)}) + "\n")
        out_f.flush()
        if s[2] % 25 == 0:
            print(f"  {r['cell']}: [{s[2]}] rag={s[0]} tenet={s[1]} err={s[3]}", flush=True)
    # rebuild the table from the full judged file (old + new rows)
    per_cell = {}
    for line in open(out_path):
        r = json.loads(line)
        s = per_cell.setdefault(r["cell"], [0, 0, 0, 0])
        s[0] += r["rag_ok"]; s[1] += r["tenet_ok"]; s[2] += 1

    print("\n=== judged (MAB anscheck, official) ===")
    R = T = N = 0
    for cell, (rr, tt, n, e) in sorted(per_cell.items()):
        R += rr; T += tt; N += n
        print(f"{cell:>22} | RAG {100*rr/n:5.1f}% | TENET {100*tt/n:5.1f}% | n={n}")
    if N:
        lo, hi = wilson_ci(T / N, N)
        print(f"{'pooled':>22} | RAG {100*R/N:5.1f}% | TENET {100*T/N:.1f}% [{100*lo:.1f},{100*hi:.1f}] n={N}")


if __name__ == "__main__":
    main()
