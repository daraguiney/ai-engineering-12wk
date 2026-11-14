"""Answer-quality eval: cited Q&A over each retrieval variant, graded by an LLM judge.

Usage: PYTHONPATH=scripts uv run --active --env-file ../.env python scripts/eval_answers.py [--k 5]
Writes results/answers_<variant>.json. Per variant it reports:
  - judged correctness on answerable questions (correct=1, partial=0.5, incorrect/abstained=0)
  - abstention on trap questions, wrong abstentions, errors
  - serving cost per query (answer model only) and end-to-end latency (retrieval + answer)
Judge tokens are reported separately: judging is an eval cost, not a serving cost.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from eval_retrieval import load_questions

from rag.hybrid import DEFAULT_POOL, Bm25Index, bm25_search, hybrid_search
from rag.index import FastEmbedder, open_index, search
from rag.judge import JUDGE_MODEL, judge_answer
from rag.qa import ANSWER_MODEL, answer_question, make_client

ROOT = Path(__file__).resolve().parents[1]
PRICE_IN, PRICE_OUT = 2.0, 10.0  # claude-sonnet-5-5, $ per million tokens


def make_variants(embedder):
    plain = open_index(ROOT / "index" / "lancedb")
    ctx = open_index(ROOT / "index" / "lancedb_ctx")
    rows = ctx.to_arrow().select(["arxiv_id", "chunk_index", "text", "contextual_text"]).to_pylist()
    bm25 = Bm25Index(rows, text_key="contextual_text")
    dense_ctx = lambda q, k: search(ctx, embedder, q, k=k)
    sparse_ctx = lambda q, k: bm25_search(bm25, q, k=k)
    return {
        "dense": lambda q, k: search(plain, embedder, q, k=k),
        "ctx_hybrid": lambda q, k: hybrid_search(q, dense=dense_ctx, sparse=sparse_ctx, k=k, pool=DEFAULT_POOL),
    }


def run_question(q, retrieve, k, client):
    started = time.monotonic()
    hits = retrieve(q.question, k)
    try:
        result = answer_question(q.question, hits, client=client)
    except Exception as exc:  # noqa: BLE001  recorded and counted in the summary, never hidden
        return {"id": q.id, "type": q.type, "error": f"{type(exc).__name__}: {exc}"}
    record = {"id": q.id, "type": q.type, "question": q.question, **result,
              "latency_s": round(time.monotonic() - started, 2)}
    if q.type != "unanswerable" and result["answerable"]:
        try:
            record["judge"] = judge_answer(
                q.question, q.notes, [e.quote for e in q.evidence], result["answer"], client=client
            )
        except Exception as exc:  # noqa: BLE001
            record["judge_error"] = f"{type(exc).__name__}: {exc}"
    return record


def summarise(name, records):
    ok = [r for r in records if "error" not in r]
    traps = [r for r in ok if r["type"] == "unanswerable"]
    ans = [r for r in ok if r["type"] != "unanswerable"]
    scores = [r["judge"]["score"] if "judge" in r else 0.0 for r in ans]
    cost = sum(r["usage"]["input_tokens"] * PRICE_IN + r["usage"]["output_tokens"] * PRICE_OUT for r in ok) / 1e6
    judged = [r["judge"]["usage"] for r in ans if "judge" in r]
    return {
        "variant": name, "answer_model": ANSWER_MODEL, "judge_model": JUDGE_MODEL,
        "answerable": len(ans), "errors": len(records) - len(ok),
        "judge_errors": sum("judge_error" in r for r in ok),
        "correctness": round(sum(scores) / len(scores), 3) if scores else None,
        "verdicts": {v: sum(r.get("judge", {}).get("verdict") == v for r in ans) for v in ("correct", "partial", "incorrect")},
        "wrongly_abstained": sum(not r["answerable"] for r in ans),
        "trap_abstained": f"{sum(not r['answerable'] for r in traps)}/{len(traps)}",
        "serving_cost_per_query_usd": round(cost / max(len(ok), 1), 4),
        "median_latency_s": round(statistics.median(r["latency_s"] for r in ok), 2) if ok else None,
        "judge_tokens": {"input": sum(u["input_tokens"] for u in judged), "output": sum(u["output_tokens"] for u in judged)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="first N questions only (canary)")
    args = parser.parse_args()
    questions, embedder, client = load_questions()[: args.limit], FastEmbedder(), make_client()
    for name, retrieve in make_variants(embedder).items():
        records = [run_question(q, retrieve, args.k, client) for q in questions]
        summary = summarise(name, records)
        out = ROOT / "results" / f"answers_{name}.json"
        out.write_text(json.dumps({"summary": summary, "records": records}, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps(summary), flush=True)
        for r in records:
            if "error" in r:
                print("ERROR", r["id"], r["error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
