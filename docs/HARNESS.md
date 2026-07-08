# Read-path performance & the language decision

**Question.** Should Tenet's retrieval core stay in Python, move to Rust, or adopt an
accelerated library? Decided on measurements, not taste.

**One-line verdict.** **Stay in Python.** At every store size this product realistically
reaches, the read path is already well under the 1-second latency corner that the recall
benchmark (LAFS) buckets as a single point — so latency wins there score *nothing*. The
one change worth banking is a ~30-line **resident float32 matrix + batched matmul** (a
20-30× speedup over the current per-row loop, zero new dependencies), and even that is a
*headroom* investment for the 100k-500k regime, not a benchmark lever. Rust pays back only
past ~1M vectors with sub-millisecond p99 SLAs — a regime the current in-RAM design hits a
*memory* wall well before, so the honest next step at that point is an on-disk ANN
(`sqlite-vec`, already a dependency), not a PyO3 rewrite of cosine.

---

## 1. What the read path actually does

`MemoryCore.recall` (`src/tenet/memory.py:193`) per query:

1. `qv = self._embed(query)` — **one embedding call** (network to Qwen Cloud, or a local
   model). Language-independent; dwarfs everything else (10-300 ms) and is untouched by any
   Python-vs-Rust decision.
2. `rows = self._rows_as_of(...)` — `SELECT *` pulls **every live row incl. its embedding
   BLOB** out of sqlite into Python.
3. A **Python `for` loop** over those rows: `np.frombuffer` deserialize → `np.dot(qv, emb)`
   cosine → `_decay(row)` math → append to a list, then `list.sort`.
4. `_expired_fact_matrix()` + `_fresh()` — a stale-echo filter that runs an **extra
   `expired @ emb` matmul per raw row**.
5. dual-pool selection + char-budget assembly (cheap, list ops).

Steps 2-4 are the only parts a language/library choice can touch. Step 1 is the real wall
clock and is provider-bound, not language-bound.

## 2. Method

CPU-only, synthetic unit vectors (no embedding model, no GPU — the model call is factored
out on purpose), replaying the **exact** loop from `memory.py`. 384-d (bge-small, the
shipped local embedder) and 4096-d (Qwen3-Embedding's largest) at 4k / 50k / 500k rows.
Min-of-reps; peak RSS held < 1 GB (one matrix at a time, `del`+`gc` between cells).
Repro: `scratchpad/profile_readpath.py` + `profile_lean.py` (harness scratch).

## 3. Results — milliseconds per query

| dim | n | sqlite fetch | **status-quo loop** (steps 2-3) | vectorized f32 | vectorized f16 | HNSW query | HNSW build (1-time) | matrix RAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 4k | 2.0 | 3.6 | **0.13** | 2.83 | 0.45 | 424 | 6 MB |
| 384 | 50k | 26.9 | 46.0 | **1.38** | 35.1 | 0.74 | 16 600 | 77 MB |
| 384 | 500k | 859 | 561 † | **19.96** | 349 | ~0.9 ‡ | ~180 000 ‡ | 768 MB |
| 4096 | 4k | 4.5 | 7.9 | **0.99** | 30.0 | ~0.5 ‡ | ~2 000 ‡ | 66 MB |
| 4096 | 50k | 473 | 132 | **12.9** | 416 | ~1 ‡ | ~180 000 ‡ | 819 MB |
| 4096 | 500k | — infeasible: an 8.2 GB resident matrix won't fit a 16 GB box — see §6 — |

† status-quo at 500k extrapolated ×(n/50k) from a 50k slice (a pure O(n) Python loop).
‡ extrapolated / not re-measured (HNSW build at 500k exceeded a sane time-box under memory
pressure; the 4k/50k builds already characterize it).

**Current real read latency = fetch + loop** (both are paid sequentially in `recall`):
50k@384-d ≈ 73 ms, 50k@4096-d ≈ 605 ms, 500k@384-d ≈ **1.42 s** (the first cell to cross 1 s).

## 4. Where the milliseconds go

- **The embedding call (step 1) is the budget.** 10-300 ms, provider-bound, identical in any
  language. If latency ever matters, this is the lever — batching/caching queries, not
  rewriting cosine.
- **The scoring loop is the only Python-tax, and it's ~28× too slow.** A single BLAS matmul
  (`mat @ qv`) + `argpartition` does in **0.13-20 ms** what the per-row
  `frombuffer`/`np.dot`/append loop takes **3.6-561 ms** to do (28-33× at 384-d, 8-10× at
  4096-d). The loss is pure Python per-iteration overhead — *not* the arithmetic, which is
  already in C.
- **At scale, deserialization overtakes arithmetic.** Pulling and unpacking every BLOB from
  sqlite (859 ms at 500k@384-d, 473 ms at 50k@4096-d) becomes the single biggest cost. The
  fix isn't a faster language — it's *not re-fetching*: keep the matrix resident in RAM so
  the numpy array **is** the index.
- **float16 is a memory tactic, not a speed one.** vec_f16 is *slower* than f32 everywhere
  (349 vs 20 ms at 500k@384-d): there's no native fp16 GEMM on this CPU BLAS, so it upcasts.
  Use f16 only to halve RAM (§6), never for latency.
- **HNSW: sub-ms queries, brutal builds.** 0.45-0.9 ms per query, but 16.6 s to build at 50k
  and ~3 min extrapolated at 500k, and it's *approximate*. Only amortizes on a near-static
  store queried far more than it's written.

## 5. The options, against the least-code ladder

| option | what | latency @50k/384 | new deps | maintainer cost | verdict |
|---|---|---:|---|---|---|
| (a) status-quo numpy loop | today | 73 ms | none | none | fine ≤ ~10k rows; wasteful above |
| **(b) optimized Python** | resident f32 matrix + `mat@qv` + `argpartition`, incremental refresh on write | **~1.4 ms** | none | ~30 lines | **recommended headroom path** |
| (c) mature ANN/FTS lib | `usearch`/`hnswlib` (dense), `sqlite-vec` (already a dep) / **SQLite FTS5** (stdlib, BM25) | <1 ms query, costly build | 0-1 | small | only past ~500k-1M or when RAM won't hold the matrix |
| (d) hand-rolled Rust (PyO3/maturin) | reimplement the scan in Rust | ≈ (b), BLAS-bound | build toolchain | **high** for a solo maintainer (maturin, cross-wheels, CI, CVE surface) | **not justified at any size this product reaches** |

**Why Rust buys ~nothing here.** The three costs that matter are *already* native C: the
cosine is BLAS, `argpartition` is numpy-C, the BLOB read is sqlite-C. The only thing left in
Python is the per-row loop — and option (b) deletes that loop with one matmul, no Rust. A
PyO3 core would at best tie numpy's BLAS while adding a maturin build, platform wheels, a CI
matrix, and an `unsafe`/CVE surface onto one maintainer. The crossover where a native ANN
*does* pay — >1M vectors, <1 ms p99 with metadata-filtered scans — is reached only *after*
the in-RAM matrix blows the memory budget (§6), at which point the right move is an on-disk
index (`sqlite-vec`, already imported) that keeps vectors out of Python entirely — still not
a bespoke Rust cosine.

**BM25 side (the fusion the retrieval layer is adding):** the least-code BM25 is **SQLite
FTS5**, which ships *inside* the sqlite already in use — no `tantivy-py`, no `duckdb`. Reach
for an external FTS engine only if FTS5's ranking proves insufficient, which it won't at
this scale.

## 6. Crossover & the real wall

- **≤ ~10k rows** (the product's actual range — LongMemEval ≈ a few thousand chunks/haystack,
  MAB FactConsolidation ≈ up to 18k facts): status-quo is 2-20 ms. **Do nothing.**
- **10k-500k rows:** adopt (b). Keeps 384-d under ~25 ms and 4096-d under ~15 ms, all in
  pure Python, and removes the per-query BLOB re-fetch. This is where 500k@384-d otherwise
  crosses the 1 s line (1.42 s today → ~20 ms).
- **The wall is memory, not CPU.** 500k × 4096-d f32 = **8.2 GB** — infeasible to hold
  resident on a 16 GB machine. f16 halves it to 4.1 GB (buys headroom, not speed); past that
  you *must* page vectors to disk. That is the `sqlite-vec` / `usearch`-memmap regime — an
  on-disk ANN — and it is the honest trigger for option (c), still not (d).

## 7. LAFS / benchmark impact — stated plainly

The recall benchmark's latency axis treats **everything under 1 second as one corner**. At
the store sizes this product runs at, the read path is 2-73 ms — three to four orders of
magnitude inside that corner. **So any latency win below 1 s is marketing, not score.** The
sole regime that touches the 1 s boundary is 500k+ rows, which is a *scaling/headroom*
concern addressed for free by option (b) in pure Python. There is no accuracy or benchmark
reason — and therefore no justification under least-code discipline — to take on a Rust
core or an ANN dependency now. Both are left as documented, measured recommendations,
gated on a store size this design does not yet reach.
