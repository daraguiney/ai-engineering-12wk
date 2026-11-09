"""Run one hillclimb round: every reviewed record x reps, recording transcripts and usage.

Usage: uv run --env-file ../.env python scripts/run_hillclimb_round.py <round_dir> <prompt_version>
           [--reps 3] [--concurrency 4] [--limit N] [--dry-run]

Writes <round_dir>/results.jsonl and <round_dir>/traces/. Re-running the same command resumes.
"""
import argparse
from pathlib import Path

from extractor.evaluate import load_reviewed
from extractor.extract import make_client
from extractor.hillclimb import load_done, run_case, run_round

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "evals" / "golden"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("prompt_version")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="first N records only (canary)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_reviewed(GOLDEN_DIR)[: args.limit]
    done = load_done(args.round_dir / "results.jsonl")
    remaining = len(records) * args.reps - len(done)
    print(
        f"{len(records)} records x {args.reps} reps, prompt {args.prompt_version}: "
        f"{remaining} attempts to run ({len(done)} already recorded)"
    )
    if args.dry_run:
        return 0

    client = make_client()
    ran = run_round(
        records,
        args.round_dir,
        reps=args.reps,
        concurrency=args.concurrency,
        run_one=lambda record, rep: run_case(
            record, rep, client=client, prompt_version=args.prompt_version
        ),
    )
    print(f"recorded {ran} attempts in {args.round_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
