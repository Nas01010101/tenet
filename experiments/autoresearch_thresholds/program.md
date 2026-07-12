# Autoresearch program — Tenet supersession thresholds

## Objective
Find the setting of Tenet's three supersession-governing thresholds that MINIMIZES
knowledge-churn retrieval error — i.e. maximizes how reliably `recall()` surfaces the
*current* value of a churned attribute and suppresses stale ones — WITHOUT any LLM call.

## Metric (ungameable, runner-owned)
`error = 1 - (subjects whose recall ranks the CURRENT city first) / S`, computed by
`target.py` on a FIXED seeded synthetic corpus and written to `$AUTORESEARCH_RESULTS`.
Lower is better. The corpus generator, seed, and scorer are the JUDGE and are out of bounds.

## Editable surface
ONLY the three thresholds, supplied by the driver via `$AUTORESEARCH_CONFIG`:
- `consistency_threshold` ∈ [0.55, 0.85]   (read-time belief–evidence consistency; default 0.70)
- `tau_key`               ∈ [0.65, 0.88]   (embedding key-resolution attribute-cosine; default 0.78)
- `text_floor`            ∈ [0.50, 0.78]   (fact-text cosine floor for key resolution; default 0.66)

`target.py` itself (corpus, scoring) is NOT a search knob — it only translates a config to a metric.

## Boundaries (hard rails — learned the expensive way)
- **Embeddings only.** `EMBED_PROVIDER=local` (bge-small, ~130 MB). **NO LLM reader, NO Qwen
  Cloud call, NO ollama / local large model.** A 14B local model on this 16 GB Mac is forbidden.
- Time-box per eval via the harness; fail-closed (error/no-metric → reverted, never kept).
- Aborts on `~/.claude/STOP`; stop when `mcp__budget_guard__check` != ok.

## Proxy caveat
This is a fast SEARCH PROXY for the distiller-key-drift + stale-raw-echo regime, not the final
judge. Any winning config MUST be re-verified on the real benchmarks (`scripts/bench_churn.py`,
`scripts/bench_supersession_firing.py`) and must not regress the deterministic suites before it
is adopted as a default. Current shipped defaults (0.70 / 0.78 / 0.66) are the incumbent baseline.

## Run 1 result (2026-07-12) — a proxy-infidelity NEGATIVE, do NOT adopt
16-eval sweep (6 random + 10 BO, bge-small, LLM-free): default (0.70/0.78/0.66) → error 0.667;
BO converged to error 0.0 at ~{consistency 0.81, tau_key 0.66, text_floor 0.70}. **This is a
reward-hacked proxy win, NOT a real improvement.** The proxy contains only synonym-drift POSITIVES
(keys that *should* collapse) and NO adversarial distinct-attribute NEGATIVES (keys that must NOT
collapse — e.g. `pet` vs `pet_name`). So it rewards ever-more-aggressive key-resolution (lower
`tau_key`), which is precisely the direction that re-introduces the false-supersession bug that
`tau_key=0.78` + `text_floor=0.66` were tuned to prevent on the *labeled* firing set (which DOES
have negatives, `scripts/bench_supersession_firing.py`). **Defaults left unchanged.**
**Fix before Run 2:** add distinct-attribute negatives to the corpus and make the metric a
FIRE-precision/recall F-score (penalize false collapses), so the proxy can't win by over-firing.
Then re-sweep, then verify on the real firing benchmark before adopting anything.

## Run
```
AUTORESEARCH_MAXIMIZE=0 python ~/code/claude-config/scripts/autoresearch/sweep.py experiments/autoresearch_thresholds
```
