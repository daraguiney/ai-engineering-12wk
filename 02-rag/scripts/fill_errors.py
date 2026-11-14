"""Re-run only the questions that errored (answer error or judge error) in saved answer results.

Usage: PYTHONPATH=scripts uv run --active --env-file ../.env python scripts/fill_errors.py
Replaces each errored record in results/answers_<variant>.json and refreshes the summary.
"""
import json
from pathlib import Path

from eval_answers import make_variants, run_question, summarise
from eval_retrieval import load_questions

from rag.index import FastEmbedder
from rag.qa import make_client

ROOT = Path(__file__).resolve().parents[1]
K = 5


def main() -> int:
    questions = {q.id: q for q in load_questions()}
    files = {p.stem.removeprefix("answers_"): p for p in sorted((ROOT / "results").glob("answers_*.json"))}
    todo = {}
    for name, path in files.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        bad = [r["id"] for r in data["records"] if "error" in r or "judge_error" in r]
        if bad:
            todo[name] = (data, bad)
    if not todo:
        print("nothing to fill")
        return 0
    retrievers = make_variants(FastEmbedder(), list(todo))
    client = make_client()
    for name, (data, bad) in todo.items():
        fresh = {i: run_question(questions[i], retrievers[name], K, client) for i in bad}
        records = [fresh.get(r["id"], r) for r in data["records"]]
        summary = {**data["summary"], **summarise(name, records)}
        files[name].write_text(json.dumps({"summary": summary, "records": records}, indent=1, ensure_ascii=False) + "\n")
        print(name, "refilled", bad, "-> still erroring:", [i for i, r in fresh.items() if "error" in r or "judge_error" in r])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
