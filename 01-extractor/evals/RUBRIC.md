# Golden-set rubric

Rules for what counts as correct in `expected`. Applied by hand when reviewing
and encoded in `prompts/extract_paper_v2.md`.

- **datasets**: only datasets/benchmarks explicitly named in the text (e.g. GSM8K,
  WMT 2014). Not task categories like "arithmetic reasoning".
- **limitations**: empty unless the text itself states a limitation. Never infer one.
- **key_results**: one short sentence per result, keeping numbers exactly as written
  (e.g. "28.4 BLEU on WMT 2014 EN-DE"). No surrounding explanation.
- All fields: use only information in the text; do not guess.
