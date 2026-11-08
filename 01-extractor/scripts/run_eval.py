"""Score a prompt version on the reviewed golden records.

Usage: uv run --env-file ../.env python scripts/run_eval.py v2 [--judge]
Writes evals/results/<version>.json and prints per-field means.
"""
import json
import sys
from pathlib import Path

from extractor.evaluate import evaluate, load_reviewed
from extractor.extract import extract_paper, make_client
from extractor.judge import judge_record

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "evals" / "golden"
RESULTS_DIR = ROOT / "evals" / "results"


def main(argv: list[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1 or flags - {"--judge"}:
        print(__doc__, file=sys.stderr)
        return 2
    version = args[0]
    client = make_client()
    records = load_reviewed(GOLDEN_DIR)
    result = evaluate(
        records,
        lambda text: extract_paper(text, client=client, prompt_version=version),
        judge_fn=(
            (lambda src, exp, pred: judge_record(src, exp, pred, client=client))
            if "--judge" in flags
            else None
        ),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{version}.json"
    out.write_text(json.dumps({"prompt_version": version, **result}, indent=2) + "\n")
    print(f"prompt {version}: {len(records)} reviewed records, {len(result['failures'])} failures")
    for field, value in result["mean"].items():
        print(f"  {field:<12} {value:.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
