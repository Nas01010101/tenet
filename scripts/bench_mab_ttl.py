"""MemoryAgentBench Test-Time Learning split (ICL) — the third MAB competency
(ICLR 2026, arXiv:2507.05257; HF ai-hyz/MemoryAgentBench, split Test_Time_Learning).

Test-Time Learning measures whether a memory system can *learn a task at inference*
from thousands of in-context demonstrations it has ingested — no weight updates. Each
cell is a text classification dataset whose "context" is 5,900-8,300 labelled
demonstrations of the form  `<query>\nlabel: <int>` ; the questions are held-out
queries and the gold is the integer class label. A memory system passes by RETRIEVING
the demonstrations most relevant to a held-out query and letting the reader copy the
right label. This is a purely retrieval-bound competency: there is no supersession,
churn, or conflict to resolve, so Tenet's bi-temporal machinery is inert here — the
value of running it is BREADTH (covering 3 of MAB's 4 core competencies) and showing
that Tenet's zero-LLM embedding-only ingestion is not *worse* than a strong RAG control
on the one axis its supersession design can't help. We say so plainly.

Cells (5 ICL classification tasks; recsys_redial is excluded — different metric
(_process_recsys_dataset, ranking not classification) and a 5.6M-char context):
  icl_banking77   (77 classes, 5,897 demos)   icl_clinic150  (150, 7,050)
  icl_nlu         (68,  8,296)                 icl_trec_coarse (6,   6,600)
  icl_trec_fine   (50,  6,400)

Arms (matched backbone, identical reader prompt + top-k):
  rag    — top-k cosine over demonstration-query embeddings                [control]
  tenet  — same demos ingested as raw slices; MemoryCore.recall() top-k    [ours]
Both embed the demo QUERY (not the label) so retrieval is query-to-query nearest
neighbours; the label rides along in `source`. Because there is no conflict to
resolve, the two arms retrieve near-identically by design — the honest expectation
is parity, and the comparison is against MAB's PUBLISHED TTL numbers, not a win claim.

Protocol fidelity (copied VERBATIM from HUST-AI-HYZ/MemoryAgentBench):
  * reader prompt  = the ICL 'rag_agent' template (utils/templates.py)
  * scoring        = parse_output() + substring_exact_match after normalize_answer,
                     max over golds (utils/eval_other_utils.py; _process_icl_dataset).
MAB's headline ICL metric is substring_exact_match: the reader emits "label: N", the
gold is "N", and drqa exact_match on the *normalized* string "label n" != "n" would be
all-zero — substring is the metric that scores it. We ALSO report a stricter
integer-exact metric (parse the emitted integer, compare equality) because
substring-on-integers can false-positive on single-digit golds (gold "1" is a
substring of "label 18"); the strict number is the conservative floor.

Reader = local qwen2.5:7b (ollama) + local bge-small embedder => $0, exactly the stack
that produced docs/factcon_results.json. API failures exclude the question (never
scored wrong). Resume-safe: --dump-preds appends per-question JSONL as it goes.

Usage:
  LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b EMBED_PROVIDER=local \
    python scripts/bench_mab_ttl.py --cells icl_banking77 --qpc 10   # smoke
  LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b EMBED_PROVIDER=local \
    python scripts/bench_mab_ttl.py --qpc 100                        # full 5x100
"""
from __future__ import annotations

import argparse, hashlib, json, re, string, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from tenet import config  # noqa: E402
from tenet import Tenet  # noqa: E402
from bench_factcon import wilson_ci  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "mab_ttl"

# --------------------------------------------------------------------------
# MAB official scoring — copied VERBATIM from utils/eval_other_utils.py
# (normalize_answer, substring_exact_match_score, parse_output). The ICL metric
# in _process_icl_dataset is parse_output() -> substring_exact_match (max over golds).
# --------------------------------------------------------------------------
def normalize_answer(answer_text):
    text = answer_text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def substring_exact_match_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def parse_output(output_text, answer_prefix="Answer:"):
    extraction_patterns = [
        re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in extraction_patterns:
        match = pattern.search(output_text)
        if match:
            extracted_text = match[1].strip()
            clean_answer = re.sub(f'^{re.escape(answer_prefix)}', '', extracted_text,
                                  flags=re.IGNORECASE).strip()
            return clean_answer
    return None


def icl_substring(pred: str, golds) -> bool:
    """MAB ICL headline metric: parse_output then substring_exact_match, max over golds."""
    parsed = parse_output(pred) or pred
    gs = golds if isinstance(golds, list) else [golds]
    gs = [g for sub in gs for g in sub] if gs and isinstance(gs[0], list) else gs
    return max(substring_exact_match_score(parsed, str(g)) for g in gs)


_INT_RE = re.compile(r"-?\d+")


def icl_int_exact(pred: str, golds) -> bool:
    """Stricter secondary: the FIRST integer emitted must equal a gold integer exactly.
    Guards against substring false positives on single-digit labels."""
    parsed = parse_output(pred) or pred
    m = _INT_RE.search(parsed)
    if not m:
        return False
    p = m.group(0)
    gs = golds if isinstance(golds, list) else [golds]
    gs = [g for sub in gs for g in sub] if gs and isinstance(gs[0], list) else gs
    return any(p == str(g).strip() for g in gs)


# MAB official ICL reader prompt ('rag_agent' in utils/templates.py) — VERBATIM.
# ({{label}} is a literal "{label}" placeholder in their template, kept as-is.)
_ICL_PROMPT = ('Use the provided mapping from the context to numerical label to assign a '
               'numerical label to the context. Only output "label: {{label}}" and nothing '
               'else. \n\nQuestion:{question} \n\n label:')


# --------------------------------------------------------------------------
# Demonstration parsing / ingestion
# --------------------------------------------------------------------------
_LABEL_RE = re.compile(r"^label:\s*(-?\d+)\s*$")


def parse_demos(context: str) -> list[tuple[str, str]]:
    """Split the ICL context into (query, label) demonstrations. Format is
    <query lines>\nlabel: <int>, demos separated by blank lines."""
    demos, cur = [], []
    for line in context.splitlines():
        m = _LABEL_RE.match(line.strip())
        if m:
            q = "\n".join(cur).strip()
            if q:
                demos.append((q, m.group(1)))
            cur = []
        elif line.strip():
            cur.append(line)
    return demos


def build_rag(cache_id: str, demos: list[tuple[str, str]], embedder) -> np.ndarray:
    """Cache demo-query embeddings (label excluded from the embedded text)."""
    npz = CACHE / f"{cache_id}.rag.npz"
    if npz.exists():
        mat = np.load(npz)["v"]
        if len(mat) == len(demos):
            return mat
    vecs = []
    B = 256
    qs = [q for q, _ in demos]
    for i in range(0, len(qs), B):
        vecs.extend(embedder(qs[i:i + B]))
    mat = np.array(vecs)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, v=mat)
    return mat


def build_tenet(cache_id: str, demos: list[tuple[str, str]], mat: np.ndarray) -> Tenet:
    """Ingest each demo as a raw slice: text = demo query, source = label. Reuses the
    already-computed rag embeddings (identical vectors) so the two arms are matched."""
    dbp = CACHE / f"{cache_id}.db"
    if dbp.exists():
        return Tenet(dbp)
    m = Tenet(dbp)
    for (q, lab), v in zip(demos, mat):
        m.core.store(q, kind="raw", salience=0.5, source=str(lab), _vec=v)
    return m


def pool_from(pairs: list[tuple[str, str]]) -> str:
    """Render retrieved (query, label) demos as MAB's 'context->label mapping'."""
    return "\n\n".join(f"{q}\nlabel: {lab}" for q, lab in pairs)


def read_label(pool: str, question: str) -> str:
    return config.chat(
        [{"role": "user", "content": pool + "\n\n" + _ICL_PROMPT.format(question=question)}],
        qwen_default=config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"), max_tokens=16)


CELLS = ["icl_banking77", "icl_clinic150", "icl_nlu", "icl_trec_coarse", "icl_trec_fine"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma list of ICL cell prefixes (default all 5)")
    ap.add_argument("--qpc", type=int, default=100, help="questions per cell")
    ap.add_argument("--k", type=int, default=20, help="retrieved demonstrations")
    ap.add_argument("--dump-preds", default="", help="append per-question JSONL (resume-safe)")
    ap.add_argument("--out", default="", help="write results JSON evidence artifact here")
    args = ap.parse_args()

    from datasets import load_dataset
    ttl = load_dataset("ai-hyz/MemoryAgentBench", split="Test_Time_Learning")
    want = [c for c in args.cells.split(",") if c] or CELLS

    core = Tenet(CACHE / "emb_host.db")
    embedder = core.core.embed_batch

    # resume: previously scored (cell, question) rows in the preds file are not re-run;
    # their outcomes are replayed into the counters so the final report covers them.
    prior: dict[tuple[str, str], dict] = {}
    if args.dump_preds and Path(args.dump_preds).exists():
        for line in Path(args.dump_preds).open():
            try:
                row = json.loads(line)
                prior[(row["cell"], row["q"])] = row
            except Exception:
                pass
        if prior:
            print(f"[resume] {len(prior)} prior predictions loaded from {args.dump_preds}", flush=True)
    preds_f = open(args.dump_preds, "a") if args.dump_preds else None

    results = {}
    t0 = time.time()
    for ex in ttl:
        source = ex["metadata"]["source"]
        cell = next((c for c in want if source.startswith(c)), None)
        if not cell:
            continue
        demos = parse_demos(ex["context"])
        cache_id = "ttl" + hashlib.md5(ex["context"].encode()).hexdigest()[:12]
        print(f"\n=== {cell}: {len(demos)} demos (cache {cache_id}) ===", flush=True)
        mat = build_rag(cache_id, demos, embedder)
        m = build_tenet(cache_id, demos, mat)
        demo_labels = [lab for _, lab in demos]

        qs = list(zip(ex["questions"], ex["answers"]))[: args.qpc]
        st = {"rag": [0, 0, 0], "tenet": [0, 0, 0]}  # [substr_ok, int_ok, n]
        errors = 0
        for qi, (q, gold) in enumerate(qs):
            prev = prior.get((cell, q))
            if prev is not None:
                st["rag"][0] += prev["rag_ss"]; st["rag"][1] += prev["rag_int"]; st["rag"][2] += 1
                st["tenet"][0] += prev["tenet_ss"]; st["tenet"][1] += prev["tenet_int"]; st["tenet"][2] += 1
                continue
            qv = np.asarray(embedder([q])[0])
            top = np.argsort(-(mat @ qv))[: args.k]
            rag_pairs = [(demos[i][0], demo_labels[i]) for i in top]
            hits = m.core.recall(q, k=args.k)
            tenet_pairs = [(h.text, str(h.source)) for h in hits]
            rp = read_label(pool_from(rag_pairs), q)
            tp = read_label(pool_from(tenet_pairs), q)
            if not rp.strip() or not tp.strip():
                errors += 1
                continue
            r_ss, r_ie = icl_substring(rp, gold), icl_int_exact(rp, gold)
            t_ss, t_ie = icl_substring(tp, gold), icl_int_exact(tp, gold)
            st["rag"][0] += r_ss; st["rag"][1] += r_ie; st["rag"][2] += 1
            st["tenet"][0] += t_ss; st["tenet"][1] += t_ie; st["tenet"][2] += 1
            if preds_f:
                preds_f.write(json.dumps({"cell": cell, "q": q, "gold": gold,
                                          "rag": rp, "tenet": tp,
                                          "rag_ss": bool(r_ss), "tenet_ss": bool(t_ss),
                                          "rag_int": bool(r_ie), "tenet_int": bool(t_ie)}) + "\n")
                preds_f.flush()
            if (qi + 1) % 20 == 0:
                print(f"  [{qi+1}/{len(qs)}] rag={st['rag'][0]}/{st['rag'][2]} "
                      f"tenet={st['tenet'][0]}/{st['tenet'][2]} (substr)", flush=True)
        m.close()
        results[cell] = {"n": st["rag"][2], "errors": errors,
                         "rag": {"substr": st["rag"][0], "int": st["rag"][1]},
                         "tenet": {"substr": st["tenet"][0], "int": st["tenet"][1]}}

    # ---- report ----
    print(f"\n=== MAB Test-Time Learning (ICL, substring_exact_match, k={args.k}, qpc={args.qpc}) ===")
    print(f"{'cell':>16} | {'RAG substr':>18} | {'TENET substr':>20} | {'RAG int':>8} | {'TEN int':>8} | err")
    pool = {"rag": [0, 0], "tenet": [0, 0], "n": 0}
    for cell, r in sorted(results.items()):
        n = r["n"]
        pool["n"] += n
        rows = {}
        for arm in ("rag", "tenet"):
            pool[arm][0] += r[arm]["substr"]; pool[arm][1] += r[arm]["int"]
            p = r[arm]["substr"] / n if n else 0.0
            lo, hi = wilson_ci(p, n)
            rows[arm] = f"{100*p:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]"
        ri = 100 * r["rag"]["int"] / n if n else 0.0
        ti = 100 * r["tenet"]["int"] / n if n else 0.0
        print(f"{cell:>16} | {rows['rag']:>18} | {rows['tenet']:>20} | {ri:7.1f}% | {ti:7.1f}% | {r['errors']}")
    print("-" * 92)
    n = pool["n"]
    for arm in ("rag", "tenet"):
        if n:
            p = pool[arm][0] / n
            lo, hi = wilson_ci(p, n)
            pi = 100 * pool[arm][1] / n
            print(f"{arm:>16} POOLED substr {100*p:.1f}% [{100*lo:.1f},{100*hi:.1f}] "
                  f"(n={n})  | int-exact {pi:.1f}%")
    print(f"wall={time.time()-t0:.0f}s")

    if args.out:
        prov = {"backbone": config.LLM_PROVIDER, "reader_model": config.get("OLLAMA_MODEL", "")
                if config.LLM_PROVIDER == "ollama" else config.get("QWEN_ANSWER_MODEL", "qwen3.7-plus"),
                "embedder": config.EMBED_PROVIDER, "k": args.k, "qpc": args.qpc,
                "wall_s": round(time.time() - t0)}
        per_cell = {}
        for cell, r in results.items():
            nc = r["n"]
            row = {"n": nc, "errors": r["errors"]}
            for arm in ("rag", "tenet"):
                p = r[arm]["substr"] / nc if nc else 0.0
                lo, hi = wilson_ci(p, nc)
                row[arm] = {"substr_acc": round(100 * p, 1),
                            "substr_ci_lo": round(100 * lo, 1),
                            "substr_ci_hi": round(100 * hi, 1),
                            "int_acc": round(100 * r[arm]["int"] / nc, 1) if nc else 0.0,
                            "substr_correct": r[arm]["substr"], "int_correct": r[arm]["int"]}
            per_cell[cell] = row
        pooled = {}
        for arm in ("rag", "tenet"):
            p = pool[arm][0] / n if n else 0.0
            lo, hi = wilson_ci(p, n)
            pooled[arm] = {"substr_acc": round(100 * p, 1), "substr_ci_lo": round(100 * lo, 1),
                           "substr_ci_hi": round(100 * hi, 1),
                           "int_acc": round(100 * pool[arm][1] / n, 1) if n else 0.0, "n": n}
        out = {"config": {
                   "benchmark": "MAB Test-Time Learning / ICL (MemoryAgentBench, arXiv:2507.05257)",
                   "split": "Test_Time_Learning", "cells": want,
                   "reader": prov["reader_model"], "embedder": "local bge-small",
                   "scoring": "MAB official ICL: parse_output + substring_exact_match, max over golds (verbatim)",
                   "prompt": "MAB official ICL 'rag_agent' template, verbatim",
                   "reproduce": "LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b EMBED_PROVIDER=local "
                                f"python scripts/bench_mab_ttl.py --qpc {args.qpc} --k {args.k}"},
               "provenance": prov, "per_cell": per_cell, "pooled": pooled}
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    if preds_f:
        preds_f.close()


if __name__ == "__main__":
    main()
