# Stage 2 — RAG with citations (Weeks 3–5) · Portfolio piece #1

**Checkpoint:** ablation table, 3 failures fixed (and how), 1 failure deliberately not fixed (and why). Design doc in `DESIGN.md`.

## Ablation
34 answerable questions, so one question is 2.9 points and differences under ~16 points are noise. Same
embedding model and chunking (bge-small, 1500/500 chars) for every row. Reproduce retrieval:
`PYTHONPATH=scripts uv run --active python scripts/eval_ablation.py` (and `eval_contextual.py`); answers and judging:
`scripts/eval_answers.py` (results in `results/`).

| Variant | recall@5 | paper recall@5 | Answer quality | Serving cost/query | End-to-end latency (median) |
|---|---|---|---|---|---|
| Baseline dense | 0.353 | 0.853 | 0.397 | $0.0081 | 1.65 s |
| BM25 only | 0.441 | 0.882 | 0.544 | $0.0081 | 1.75 s |
| + BM25 hybrid (RRF) | 0.382 | 0.912 | 0.471 | $0.0080 | 1.87 s |
| + Reranker (bge-reranker-base, top 20) | 0.559 | 0.941 | 0.662 | $0.0083 | 21.04 s |
| + Reranker (MiniLM-L-12, top 20) | 0.441 | 0.941 | not judged | | retrieval alone 7.2 s |
| + Reranker (MiniLM-L-6, top 20) | 0.441 | 0.941 | not judged | | retrieval alone 2.7 s |
| + Contextual chunks, dense | 0.559 | 0.853 | not judged | one-time $1.97 to index | retrieval alone 0.012 s |
| + Contextual chunks, BM25 | 0.559 | 0.912 | not judged | (same) | retrieval alone 0.010 s |
| + Contextual chunks, hybrid (RRF) | 0.559 | 0.912 | **0.618** | $0.0081 (+ one-time index) | 1.67 s |
| + Contextual chunks, hybrid + reranker | **0.618** | 0.882 | **0.691** | $0.0082 (+ one-time index) | 17.86 s |

Answer quality = Opus 5.5 judge (prompt `judge_v2`) against the reference answer and evidence quote (correct 1,
partial 0.5, wrong or abstained 0), averaged over the 34 answerable questions; Sonnet 5.5 answers from the top 5 chunks.
The four rows named in the plan (baseline, hybrid, reranker, contextual) plus BM25 and the best combination are judged;
rows marked "not judged" are retrieval-only. Trap questions: 6/6 abstained for every variant. Every variant answered
all 40 questions (4 transient network/judge errors were re-run).

**Reading it**
- **Answer quality moves with retrieval:** 0.397 -> 0.618 from dense to contextual hybrid (8 questions better, 1 worse; judge v2, see Judge check).
  Almost no wrong answers (judge v2: 2 incorrect for dense, 1 for contextual hybrid, all of the kind 'the sources do not say'). Nearly every loss is the model abstaining when the evidence was not
  retrieved (18 of 34 abstentions for dense, 12 for contextual hybrid). Traps: 6/6 abstained for both.
- **Ranking by answer quality:** contextual hybrid + reranker 0.691 > plain hybrid + reranker 0.662 > contextual hybrid 0.618
  > BM25 0.544 > hybrid 0.471 > dense 0.397. Adding the reranker on top of contextual hybrid (+0.07, about 2 questions) is
  within noise, but costs ~16 s more per query on CPU. **Best trade-off: contextual hybrid** (0.618 at 1.7 s, no extra per-query cost).
- Wrong answers stay rare in every variant (0-2 of 34); what changes is how often the system declines to answer
  because the passage was not retrieved (18 for dense down to 8 for contextual hybrid + reranker).
- The reranker rows' end-to-end latency (17-21 s) is higher than the 7 s measured for retrieval alone; I have not isolated why
  (the run overlapped other work on the machine), so treat the exact figure loosely. It is slow either way.
- The judge is checked against 12 human grades only (see Judge check); 34 questions is small (one question = 2.9 points).
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

**Judge v2 (added after the check above).** One rule: an answer that says the sources do not state something the
reference does state is *incorrect*, not partial. Re-judged all saved answers (`scripts/rejudge.py`, v1 verdicts kept as
`judge_v1`): agreement with the 12 human grades rose from 7/12 to **10/12**. The two remaining disagreements are g03
(judge stricter than the human on vague numbers) and g05 (extra Wikipedia clause); I left the rubric alone for those
rather than tune to 12 samples. Headline answer quality under v2: dense 0.397, contextual hybrid 0.618. The judge is
"checked against 12 human grades", not validated.
