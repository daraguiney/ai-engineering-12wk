"""Re-judge the saved answers with the current judge prompt (no new answers generated).

Usage: PYTHONPATH=scripts uv run --active --env-file ../.env python scripts/rejudge.py
Keeps each earlier verdict under `judge_v1` and rewrites `judge`, then refreshes results/answers_*.json summaries.
"""
import json
from pathlib import Path

from eval_answers import summarise
from eval_retrieval import load_questions

from rag.judge import JUDGE_PROMPT_VERSION, judge_answer
from rag.qa import make_client

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    notes = {q.id: q for q in load_questions()}
    client = make_client()
    for path in sorted((ROOT / "results").glob("answers_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records = []
        for r in data["records"]:
            if "judge" not in r:
                records.append(r)
                continue
            q = notes[r["id"]]
            new = judge_answer(q.question, q.notes, [e.quote for e in q.evidence], r["answer"], client=client)
            records.append({**r, "judge_v1": r.get("judge_v1", r["judge"]), "judge": new})
        summary = {**summarise(data["summary"]["variant"], records), "judge_prompt": JUDGE_PROMPT_VERSION}
        path.write_text(json.dumps({"summary": summary, "records": records}, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
