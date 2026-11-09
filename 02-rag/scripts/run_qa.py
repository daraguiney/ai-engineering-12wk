"""Run cited Q&A over the test set using the dense baseline index and score what can be scored.

Usage: uv run --env-file ../.env python scripts/run_qa.py [--k 5]
Writes results/baseline_qa.json. Answer correctness needs a judge (Week 4); this reports:
  - abstention on the trap (unanswerable) questions
  - wrongly abstaining on answerable questions
  - whether the cited chunks contain the labelled evidence quote(s)
  - whether the evidence was even in the retrieved chunks (separates retrieval from generation)
"""
import argparse
import json
from pathlib import Path

from rag.index import FastEmbedder, open_index, search
from rag.qa import ANSWER_MODEL, answer_question, make_client
from rag.testset import Question, quote_in_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "baseline_qa.json"
PRICE_IN, PRICE_OUT = 2.0, 10.0  # claude-sonnet-5-5, $ per million tokens


def _contains_all(question: Question, chunks: list[dict]) -> bool:
    return all(
        any(c["arxiv_id"] == e.arxiv_id and quote_in_text(e.quote, c["text"]) for c in chunks)
        for e in question.evidence
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    rows = json.loads((ROOT / "testset" / "questions.json").read_text(encoding="utf-8"))
    questions = [Question(**{k: v for k, v in r.items() if k in Question.model_fields}) for r in rows]
    table, embedder, client = open_index(ROOT / "index" / "lancedb"), FastEmbedder(), make_client()

    records = []
    for q in questions:
        hits = search(table, embedder, q.question, k=args.k)
        try:
            result = answer_question(q.question, hits, client=client)
        except Exception as exc:  # noqa: BLE001  recorded and counted in the output, never hidden
            records.append({"id": q.id, "type": q.type, "error": f"{type(exc).__name__}: {exc}"})
            continue
        cited_chunks = [h for h in hits for c in result["cited"]
                        if (h["arxiv_id"], h["chunk_index"]) == (c["arxiv_id"], c["chunk_index"])]
        records.append({
            "id": q.id, "type": q.type, "question": q.question, **result,
            "evidence_in_retrieved": None if q.type == "unanswerable" else _contains_all(q, hits),
            "evidence_in_cited": None if q.type == "unanswerable" else _contains_all(q, cited_chunks),
        })

    errors = [r for r in records if "error" in r]
    ok = [r for r in records if "error" not in r]
    traps = [r for r in ok if r["type"] == "unanswerable"]
    answerable = [r for r in ok if r["type"] != "unanswerable"]
    cost = sum(r["usage"]["input_tokens"] * PRICE_IN + r["usage"]["output_tokens"] * PRICE_OUT for r in ok) / 1e6
    summary = {
        "model": ANSWER_MODEL, "k": args.k, "errors": len(errors),
        "trap_abstained": f"{sum(not r['answerable'] for r in traps)}/{len(traps)}",
        "answerable_wrongly_abstained": f"{sum(not r['answerable'] for r in answerable)}/{len(answerable)}",
        "evidence_in_retrieved": f"{sum(bool(r['evidence_in_retrieved']) for r in answerable)}/{len(answerable)}",
        "answered_and_cited_the_evidence": f"{sum(bool(r['evidence_in_cited']) for r in answerable if r['answerable'])}/{len(answerable)}",
        "cost_per_query_usd": round(cost / max(len(ok), 1), 4), "total_cost_usd": round(cost, 3),
    }
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))
    for r in errors:
        print("ERROR", r["id"], r["error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
