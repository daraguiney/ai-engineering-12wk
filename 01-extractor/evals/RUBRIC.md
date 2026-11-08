# Golden-set rubric

Rules for what counts as correct in `expected`. Applied by hand when reviewing
and encoded in `prompts/extract_paper_v2.md`.

- **datasets**: only datasets/benchmarks explicitly named in the text (e.g. GSM8K,
  WMT 2014). Not task categories like "arithmetic reasoning".
- **limitations**: empty unless the text itself states a limitation. Never infer one.
- **key_results**: one short sentence per result, keeping numbers exactly as written
  (e.g. "28.4 BLEU on WMT 2014 EN-DE"). No surrounding explanation.
- **datasets**: models (e.g. BERT-large, GPT-2) are never datasets.
- **method**: 2-3 sentences (~40 words): the core technique, its named components, and how it
  was evaluated or at what scale, as stated in the text.
- **limitations**: also include weaknesses the authors themselves report about the approach
  (failure cases, known biases, missing coverage), not only the word "limitation".
- All fields: use only information in the text; do not guess.
- **datasets (corpora)**: a named corpus the method actually uses, e.g. Wikipedia as a
  retrieval index, counts. A corpus mentioned only as an example ("such as Wikipedia") does not.
- **key_results**: findings only. Setup facts (model size, training data size, release notes)
  belong in `method`, not here.

## Provenance
Records with `"reviewed_by": "claude"` were reviewed by Claude against this rubric,
not by the repo owner. Records without `reviewed_by` were reviewed by the owner.
