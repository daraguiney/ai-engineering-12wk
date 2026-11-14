# Stage 2 — RAG with citations (Weeks 3–5) · Portfolio piece #1

**Checkpoint:** ablation table, 3 failures fixed (and how), 1 failure deliberately not fixed (and why). Design doc in `DESIGN.md`.

## Ablation
Retrieval only so far: 34 answerable questions, so one question is 2.9 points and differences under
~16 points are noise. Same index (bge-small, 1500/500 chars) for every row. Reproduce:
`PYTHONPATH=scripts uv run --active python scripts/eval_ablation.py` (results in `results/ablation_retrieval.json`).

| Variant | recall@5 | paper recall@5 | Answer quality | Cost/query | Latency (median, retrieval) |
|---|---|---|---|---|---|
| Baseline dense | 0.353 | 0.853 | **0.426** | $0.0081 | 0.014 s retrieval, 1.65 s end to end |
| BM25 only | 0.441 | 0.882 | | | 0.012 s |
| + BM25 hybrid (RRF) | 0.382 | 0.912 | | | 0.024 s |
| + Reranker (bge-reranker-base, top 20) | **0.559** | 0.941 | | | 7.8 s (CPU) |
| + Reranker (MiniLM-L-12, top 20) | 0.441 | 0.941 | | | 7.2 s (CPU) |
| + Reranker (MiniLM-L-6, top 20) | 0.441 | 0.941 | | | 2.7 s (CPU) |
| + Contextual chunks, dense | 0.559 | 0.853 | | one-time $1.97 to index | 0.012 s |
| + Contextual chunks, BM25 | 0.559 | 0.912 | | (same) | 0.010 s |
| + Contextual chunks, hybrid (RRF) | 0.559 | 0.912 | **0.632** | $0.0081 (+ one-time index) | 0.024 s retrieval, 1.67 s end to end |
| + Contextual chunks, hybrid + reranker | **0.618** | 0.882 | | (same) | 6.9 s (CPU) |

Answer quality = Opus 5.5 judge against the reference answer and evidence quote (correct 1, partial 0.5, wrong or
abstained 0), averaged over the 34 answerable questions; Sonnet 5.5 answers from the top 5 chunks. Judged for two
variants so far (`scripts/eval_answers.py`, `results/answers_*.json`); other rows are retrieval-only.

**Reading it**
- **Answer quality moves with retrieval:** 0.426 -> 0.632 from dense to contextual hybrid (8 questions better, 1 worse).
  Neither variant gave a wrong answer: 0 incorrect of 34. Every loss is the model abstaining when the evidence was not
  retrieved (18 of 34 abstentions for dense, 12 for contextual hybrid). Traps: 6/6 abstained for both.
- The judge has not been checked against human grading yet, and 34 questions is small (one question = 2.9 points).
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

## Judge check (human grading, 2026-11-14)
12 answers graded blind by the owner (`review_app/grade_server.py`; items `testset/grading_items.json`, judge key
`results/grading_key.json`, comparison `results/judge_vs_human.json`). The sample is every judge-"partial" answer (4)
plus 8 random judge-"correct" ones, so it is not a random draw of all answers.
- **Exact agreement 7/12.** Judge "correct" was confirmed 7 of 8 times by the human.
- **The judge is too lenient on answers that wrongly say "the sources do not state X" when X is in the reference.**
  3 of the 4 judge-"partial" answers were graded "incorrect" by the human, all of this kind (g06, g09, g11).
- One judge "partial" was graded "correct" (g03: vague "exceeding 80%" where the reference has 85% vs 81%), and one
  judge "correct" was graded "incorrect" (g05: the answer adds an off-topic Wikipedia exception to the right threshold).
- Effect on the headline: counting only "correct" (no half credit) gives dense 0.382 vs contextual hybrid 0.618,
  the same direction and gap as the 0.426 vs 0.632 above. Half credit for partials flatters both variants slightly.
