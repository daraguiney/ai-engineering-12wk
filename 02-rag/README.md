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
| + Contextual chunks | | | | | |

**Reading it**
- The reranker is the only change that clearly helps: versus dense it gains 8 questions and loses 1 at recall@5.
- BM25 alone beats dense (gains 6, loses 3): these questions use exact terms (model names, numbers).
- Plain RRF hybrid (+2/-1 vs dense) is within noise and does not beat BM25 alone.
- Cost: the reranker makes retrieval ~300x slower on CPU (7.8 s/query). A smaller model (`ms-marco-MiniLM-L-6`) is untested.
- Cross-paper questions are still 0/5 at recall@5 (one reaches rank 9): reranking top-20 does not fix questions that need two papers.
