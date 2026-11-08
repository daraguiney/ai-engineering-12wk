# Stage 1 — Typed extractor + eval harness (Weeks 1–2)

**Checkpoint:** show a metric that moved from X to Y because of a specific prompt/schema change, and explain why.

Extracts a typed `PaperExtraction` (title, authors, problem, method, datasets, key results,
limitations) from arXiv abstracts. Prompts live in `prompts/`, golden data in `evals/golden/`
(50 papers), rules in `evals/RUBRIC.md`.

## Results (50 reviewed records, extractor `claude-sonnet-5-5`, judge `claude-opus-5-5`)

| Field | v1 | v2 | v3 (not adopted) |
|---|---|---|---|
| title | 1.000 | 1.000 | 1.000 |
| authors | 1.000 | 1.000 | 1.000 |
| **datasets** | **0.780** | **0.975** | 0.955 |
| key_results | 0.977 | 0.977 | 0.987 |
| limitations | 0.930 | 0.920 | 0.900 |
| problem (judge) | 0.960 | 0.950 | 0.930 |
| method (judge) | 0.940 | 0.930 | 0.960 |

Scores are after the owner's spot-check of 11 records (earlier run: datasets 0.744 → 0.975).

**The change:** v2 adds explicit rules (datasets = explicitly named datasets/benchmarks only,
never task categories; limitations only if stated; key_results as short sentences).
**Why it moved:** v1 listed task categories ("arithmetic reasoning"), models and its own data
collections as datasets, which failed ~16 of 50 records. v2 failed 2.

## Error analysis (v2 remaining misses)
- datasets: v2 lists models (`BERT-large`, `GPT-2`) as datasets in 2205.14135 → candidate v3 rule: "models are not datasets". 2005.11401: model omits Wikipedia (corpus rule is a judgment call).
- limitations: 2305.14314 and 2306.05685 — model misses limitations the gold records (authors' failure analysis, judge biases). 2005.14165 is a wording difference that exact-string F1 punishes → the scorer is stricter than a human would be.
- method (judge) fell ~5 points for BOTH versions after the spot-check, on mostly the same records (6 shared, 4 of them edited by the owner). Likely cause: the corrected gold `method` text is more detailed than the prompt asks for, so the judge calls the shorter extraction "partial". This measures style match as much as correctness → fix by aligning the prompt/gold detail level, not by trusting the drop.
- limitations 0.95 → 0.92 for v2: owner added a stated HELM limitation the model misses. Within noise at n=50.

## v3 experiment (mixed — v2 stays the default)
v3 added: models are not datasets, a named-corpus clause, a ~40-word `method` rule, and a broader
`limitations` rule. Outcome:
- Worked: `method` judge 0.93 → 0.96 (the length/detail rule); models like BERT-large no longer listed as datasets.
- Backfired: the corpus clause adds `Wikipedia` where gold has none (REALM, ReAct); the broadened limitations rule
  makes the model list weak results of *tested* models as limitations (Mamba/MMLU); `MLPerf 1.1` now counts as a dataset.
- Differences of 1–2 points are within single-run noise at n=50. Only the `method` gain looks real.
- v3 rules were written after studying this same gold set, so any gain here would overstate generalisation.
- Next: keep only the method rule, and measure on 10–15 fresh papers not used so far (held-out), instead of tuning more on these 50.

## Caveats — read before trusting these numbers
1. **Gold labels: 14 of 50 reviewed by the owner** (3 first, then 11 in a spot-check of the riskiest records); the other 36 were reviewed by Claude (see `reviewed_by`). The owner changed fields in all 11 spot-checked records, so Claude's labels were not reliable on the hard cases.
2. **Anchoring:** 40 of the 50 drafts were produced by the v2 prompt, then edited. This favours v2. On just the 10 v1-drafted records (the fairer comparison) datasets goes 0.70 → 0.90: smaller, but the same direction.
3. The judge was only sanity-checked on obviously wrong candidates, not on subtle errors.
4. Scores are single runs; no variance estimate.

## Reproduce
```
uv run --env-file ../.env python scripts/run_eval.py v1 --judge
uv run --env-file ../.env python scripts/run_eval.py v2 --judge
```
