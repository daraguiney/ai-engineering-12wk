# Stage 2 — RAG with citations (Weeks 3–5) · Portfolio piece #1

**Checkpoint:** ablation table, 3 failures fixed (and how), 1 failure deliberately not fixed (and why). Design doc in `DESIGN.md`.

## Ablation
Retrieval only so far: 34 answerable questions, so one question is 2.9 points and differences under
~16 points are noise. Same index (bge-small, 1500/500 chars) for every row. Reproduce:
`PYTHONPATH=scripts uv run --active python scripts/eval_ablation.py` (results in `results/ablation_retrieval.json`).

| Variant | recall@5 | paper recall@5 | Answer quality | Cost/query | Latency (median, retrieval) |
|---|---|---|---|---|---|
| Baseline dense | 0.353 | 0.853 | not yet judged | ~$0.008 (generation) | 0.014 s |
| BM25 only | 0.441 | 0.882 | | | 0.012 s |
| + BM25 hybrid (RRF) | 0.382 | 0.912 | | | 0.024 s |
| + Reranker (bge-reranker-base, top 20) | **0.559** | 0.941 | | | 7.8 s (CPU) |
| + Reranker (MiniLM-L-12, top 20) | 0.441 | 0.941 | | | 7.2 s (CPU) |
| + Reranker (MiniLM-L-6, top 20) | 0.441 | 0.941 | | | 2.7 s (CPU) |
| + Contextual chunks, dense | 0.559 | 0.853 | | one-time $1.97 to index | 0.012 s |
| + Contextual chunks, BM25 | 0.559 | 0.912 | | (same) | 0.010 s |
| + Contextual chunks, hybrid (RRF) | 0.559 | 0.912 | | (same) | 0.024 s |
| + Contextual chunks, hybrid + reranker | **0.618** | 0.882 | | (same) | 6.9 s (CPU) |

**Reading it**
- The reranker is the only change that clearly helps: versus dense it gains 8 questions and loses 1 at recall@5.
- BM25 alone beats dense (gains 6, loses 3): these questions use exact terms (model names, numbers).
- Plain RRF hybrid (+2/-1 vs dense) is within noise and does not beat BM25 alone.
- Cost: the best reranker makes retrieval ~500x slower on CPU (7.4 s/query). The smaller MiniLM models are
  faster (L-6: 2.7 s) but only match BM25 at recall@5 (0.441); they do better at recall@10 (0.647 vs 0.588).
  Latency is dominated by the 20 long (1500-char) passages, not by model size alone.
- **Contextual chunking** (Haiku 5.5 writes a 1-2 sentence note per chunk, prepended before embedding / BM25; `scripts/contextualize_corpus.py`)
  lifts plain dense search from 0.353 to 0.559 (gains 7 questions, loses 0). That is as good as the reranker's gain without
  the query-time latency. One-time cost: $1.94 for 3,378 chunks (my pre-run estimate of ~$1.2 was low; the tokenizer
  counts ~30% more tokens than I assumed) plus ~8.5 min to re-embed. Notes are generated from the paper, so a stale
  note means re-running it when chunking changes. Quote matching and reranking use the original chunk text.
- Contextual hybrid has the best top-1 (0.441) and recall@10 (0.735) at 0.024 s per query. Adding the reranker on top
  gives 0.618 at recall@5, but +3/-1 over the plain reranker is within noise and costs ~7 s per query.
- Cross-paper questions are still 0/5 at recall@5 with every variant, contextual ones included: top-k retrieval returns chunks for one paper and does not fix questions that need two. Next idea (untested): retrieve per paper, then merge.
