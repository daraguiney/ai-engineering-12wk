| round | change | target | limitations | method | spend |
|-------|--------|--------|-------------|--------|-------|
| 0 | baseline (v2, 50 records x 3 reps, no split: directional) | 0.942 | 0.920 | 0.963 | $2.96 |

Best so far: baseline. No round 1 run.

Analyzer (read the baseline failures) recommended no prompt change. Only 9 of 50 records are below
1.0 on target. The largest prompt-fixable behavior (omitting a limitation the abstract states
indirectly, 7 of 150 rows) is worth at most 0.023 on target, 0.013 if the gold-flagged 2306.05685
is excluded. Noise is +-0.044 (rep-to-rep about 0.01), and a more aggressive limitations rule would
risk false positives on empty-gold records (2009.03300). The rest of the loss is gold/scorer:
exact-string F1 penalising wording on limitations (2005.14165, 2211.09110) and gold choices about
where a sentence belongs (2305.14314 method). Those files are off-limits for this loop.
