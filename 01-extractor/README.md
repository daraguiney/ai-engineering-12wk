# Stage 1 — Typed extractor + eval harness (Weeks 1–2)

**Checkpoint:** show a metric that moved from X to Y because of a specific prompt/schema change, and explain why.

Extracts a typed `PaperExtraction` (title, authors, problem, method, datasets, key results,
limitations) from arXiv abstracts. Prompts live in `prompts/`, golden data in `evals/golden/`
(50 papers), rules in `evals/RUBRIC.md`.

## Results (50 reviewed records, extractor `claude-sonnet-5-5`, judge `claude-opus-5-5`)

| Field | v1 | v2 |
|---|---|---|
| title | 1.000 | 1.000 |
| authors | 1.000 | 1.000 |
| **datasets** | **0.744** | **0.975** |
| key_results | 0.973 | 0.977 |
| limitations | 0.930 | 0.950 |
| problem (judge) | 0.980 | 0.960 |
| method (judge) | 0.990 | 0.980 |

**The change:** v2 adds explicit rules (datasets = explicitly named datasets/benchmarks only,
never task categories; limitations only if stated; key_results as short sentences).
**Why it moved:** v1 listed task categories ("arithmetic reasoning"), models and its own data
collections as datasets, which failed 16 of 50 records. v2 failed 2.

## Error analysis (v2 remaining misses)
- datasets: v2 lists models (`BERT-large`, `GPT-2`) as datasets in 2205.14135 → candidate v3 rule: "models are not datasets". 2005.11401: model omits Wikipedia (corpus rule is a judgment call).
- limitations: 2305.14314 and 2306.05685 — model misses limitations the gold records (authors' failure analysis, judge biases). 2005.14165 is a wording difference that exact-string F1 punishes → the scorer is stricter than a human would be.
- problem/method judge scores dipped slightly in v2 (2 records); likely noise, not investigated.

## Caveats — read before trusting these numbers
1. **Gold labels were mostly reviewed by Claude, not a human** (47 of 50; see `reviewed_by`). The 3 owner-reviewed records are the only independent ones.
2. **Anchoring:** 40 of the 50 drafts were produced by the v2 prompt, then edited. This favours v2. On just the 10 v1-drafted records (the fairer comparison) datasets goes 0.70 → 0.90: smaller, but the same direction.
3. The judge was only sanity-checked on obviously wrong candidates, not on subtle errors.
4. Scores are single runs; no variance estimate.

## Reproduce
```
uv run --env-file ../.env python scripts/run_eval.py v1 --judge
uv run --env-file ../.env python scripts/run_eval.py v2 --judge
```
