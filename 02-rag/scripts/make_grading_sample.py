"""Draw the blind human-grading sample (12 answers) from the judged Q&A results.

Usage: uv run --active python scripts/make_grading_sample.py [--seed 20261114]
Writes testset/grading_items.json (served to the grading page, no judge verdicts)
and results/grading_key.json (hidden judge verdicts; do not open before grading).
"""
import argparse
import json
from pathlib import Path

from rag.grading import build_grading_sample

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    questions = json.loads((ROOT / "testset" / "questions.json").read_text(encoding="utf-8"))
    kept = json.loads((ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8"))["kept"]
    records = {
        name: json.loads((ROOT / "results" / f"answers_{name}.json").read_text(encoding="utf-8"))["records"]
        for name in ("dense", "ctx_hybrid")
    }
    items, key = build_grading_sample(records, questions, {k["arxiv_id"]: k["title"] for k in kept}, seed=args.seed)
    (ROOT / "testset" / "grading_items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n")
    (ROOT / "results" / "grading_key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"{len(items)} items written (seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
