"""Autoresearch HP-sweep target — tune Tenet's three supersession thresholds
against a deterministic, LLM-FREE churn-retrieval metric.

The driver (sweep.py) proposes {consistency_threshold, tau_key, text_floor} and
writes it to $AUTORESEARCH_CONFIG; this target builds a FIXED, seeded synthetic
knowledge-churn corpus, ingests it (embeddings only — bge-small local, NO LLM,
NO cloud call, NO ollama), runs recall(), and writes the scalar ERROR to
$AUTORESEARCH_RESULTS (lower is better). The number the driver reads comes from
the results file, never this script's stdout.

The JUDGE — the synthetic corpus generator, the fixed seed, and the scoring rule
below — is NOT the search surface: the config only sets three numeric thresholds.
There is no knob here to reward-hack; a "win" that required changing the corpus or
the scorer would be void.

Proxy honesty: this synthetic corpus models the distiller-key-drift + stale-raw-echo
regime that the real supersession thresholds exist for (a subject's attribute is
updated U times, each update keyed with a DIFFERENT synonym for the same attribute,
plus a verbatim raw echo of each old value). Any config that wins here MUST be
re-verified on the real benchmarks (bench_churn / bench_supersession_firing) before
being adopted as a default — it is a fast search proxy, not the final judge.
"""
import json
import os
import tempfile

# ── read driver-proposed config, set the thresholds via env BEFORE importing tenet ──
cfg_path = os.environ.get("AUTORESEARCH_CONFIG")
cfg = json.loads(open(cfg_path).read()) if cfg_path and os.path.exists(cfg_path) else {}
consistency_threshold = float(cfg.get("consistency_threshold", 0.70))
tau_key = float(cfg.get("tau_key", 0.78))
text_floor = float(cfg.get("text_floor", 0.66))

# key-resolution thresholds are read at import from these env vars (memory.py)
os.environ["TENET_KEY_RESOLUTION"] = "on"
os.environ["TENET_KEY_RESOLUTION_TAU"] = str(tau_key)
os.environ["TENET_KEY_RESOLUTION_TEXTFLOOR"] = str(text_floor)
os.environ["EMBED_PROVIDER"] = "local"          # bge-small, ~130MB, deterministic, RAM-safe
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.pop("OLLAMA_MODEL", None)             # belt-and-suspenders: never a local LLM
os.environ.pop("OLLAMA_BASE_URL", None)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from tenet.memory import MemoryCore  # noqa: E402

# ── JUDGE: fixed, seeded synthetic knowledge-churn corpus (do not edit) ──────────
SEED = 0
S = 15          # subjects
U = 4           # updates per attribute (churn depth)
# synonym sets for one attribute — the distiller-key-drift the thresholds must absorb
ATTR_SYNONYMS = ["residence", "home_city", "city", "location", "home_town"]
BASE_ATTR = "residence"
CITIES = ["Boston", "Seattle", "Denver", "Austin", "Chicago", "Portland",
          "Miami", "Dallas", "Phoenix", "Atlanta", "Newark", "Fresno",
          "Tucson", "Reno", "Tulsa", "Akron", "Provo", "Ogden", "Salem", "Erie"]


def _rng(seed):
    import random
    r = random.Random(seed)
    return r


def build_and_score() -> float:
    r = _rng(SEED)
    dbdir = tempfile.mkdtemp(prefix="ar_tenet_")
    clock = [1000.0]
    core = MemoryCore(os.path.join(dbdir, "t.db"), now=lambda: clock[0])
    truth = {}   # subject -> current (last) value string
    for s in range(S):
        subj = f"subj{s}"
        # U successive values for this subject's residence, each keyed with a DIFFERENT
        # synonym (distiller drift) + a verbatim raw echo of the value at that time.
        vals = r.sample(CITIES, U)
        for i, v in enumerate(vals):
            clock[0] += 100.0
            syn = ATTR_SYNONYMS[i % len(ATTR_SYNONYMS)]
            core.store(f"{subj} lives in {v}", key=f"{subj}::{syn}", valid_at=clock[0])
            core.store(f"{subj} said: I live in {v}", kind="raw",
                       source=subj, valid_at=clock[0], surprise_gate=None)
        truth[subj] = vals[-1]
    # score: for each subject, does recall surface the CURRENT city and rank no STALE
    # city of that subject above it? (pure retrieval; LLM-free)
    correct = 0
    for s in range(S):
        subj = f"subj{s}"
        cur = truth[subj]
        mems = core.recall(f"where does {subj} live now",
                           k=5, consistency_threshold=consistency_threshold)
        # earliest-first stale cities for this subject
        ranked_cities = []
        for m in mems:
            if subj not in m.text:
                continue
            for c in CITIES:
                if c in m.text:
                    ranked_cities.append(c)
                    break
        if ranked_cities and ranked_cities[0] == cur:
            correct += 1
    core.close()
    return 1.0 - correct / S   # ERROR, lower is better


err = build_and_score()

results = os.environ.get("AUTORESEARCH_RESULTS")
if results:
    with open(results, "w") as f:
        json.dump({"metric": err}, f)
print(f"error={err:.4f}  (consistency={consistency_threshold} tau_key={tau_key} text_floor={text_floor})")
