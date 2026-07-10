"""Render the ChurnBench accuracy-vs-U curve (one line per arm, Wilson CI shaded) from
the results JSON written by `bench_churn.py --out`.

Usage: python scripts/plot_churnbench.py <results.json> [--out docs/churnbench_curve.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_STYLE = {
    "tenet": {"color": "#1b7a3d", "marker": "o", "label": "Tenet"},
    "rag": {"color": "#8a8a8a", "marker": "s", "label": "RAG"},
    "mem0": {"color": "#c2703a", "marker": "^", "label": "Mem0-style"},
    "hipporag": {"color": "#5b6fc2", "marker": "D", "label": "HippoRAG-v2-style"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="path to results JSON from bench_churn.py --out")
    ap.add_argument("--out", default="docs/churnbench_curve.png")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    curve = data["curve"]
    sweep = data["config"]["updates"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    for arm, style in ARM_STYLE.items():
        if arm not in curve:
            continue
        us = sorted(int(u) for u in curve[arm])
        acc = [100 * curve[arm][str(u)]["acc"] for u in us]
        lo = [100 * curve[arm][str(u)]["ci_lo"] for u in us]
        hi = [100 * curve[arm][str(u)]["ci_hi"] for u in us]
        ax.plot(us, acc, color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=6)
        ax.fill_between(us, lo, hi, color=style["color"], alpha=0.15)

    ax.axhline(90, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(sweep[-1], 91, "90% (half-life threshold)", ha="right", va="bottom",
            fontsize=8, color="black", alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sweep)
    ax.set_xticklabels([str(u) for u in sweep])
    ax.set_xlabel("updates per fact (U)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(-2, 104)
    ax.set_title("ChurnBench: accuracy vs. updates-per-fact")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
