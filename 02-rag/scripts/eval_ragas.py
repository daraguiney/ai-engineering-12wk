"""Run Ragas metrics on saved answers and set them beside our judge and the human grades.

Usage: PYTHONPATH=scripts uv run --active --env-file ../.env python scripts/eval_ragas.py [--variant ctx_hybrid] [--n 8]
Metrics: faithfulness (answer vs retrieved text), context precision / recall (retrieved text vs reference),
factual correctness (answer vs reference). Ragas LLM: Sonnet 5.5. Tokens are tallied so the cost is measured.
Writes results/ragas_<variant>.json.
"""
import argparse
import asyncio
import json
from pathlib import Path

from anthropic import AsyncAnthropic
from eval_answers import make_variants
from eval_retrieval import load_questions
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from rag.index import FastEmbedder

ROOT = Path(__file__).resolve().parents[1]
RAGAS_MODEL = "claude-sonnet-5-5"
K = 5


def pick_sample(records: list[dict], n: int) -> list[dict]:
    """A spread of answered-correct, answered-incorrect and abstained questions (traps excluded)."""
    ans = [r for r in records if "error" not in r and r["type"] != "unanswerable"]
    correct = [r for r in ans if r["answerable"] and r.get("judge", {}).get("verdict") == "correct"]
    wrong = [r for r in ans if r["answerable"] and r.get("judge", {}).get("verdict") in ("incorrect", "partial")]
    abstained = [r for r in ans if not r["answerable"]]
    quota = [(correct, n // 2), (wrong, 1), (abstained, n - n // 2 - 1)]
    chosen = [r for pool, take in quota for r in pool[:take]]
    return chosen[:n]


def human_grades() -> dict[tuple[str, str], str]:
    key = json.loads((ROOT / "results" / "grading_key.json").read_text())
    items = {i["id"]: i for i in json.loads((ROOT / "testset" / "grading_items.json").read_text())}
    verdicts = json.loads((ROOT / "testset" / "grading_verdicts.json").read_text())
    return {(key[i]["question_id"], items[i]["answer"]): v["verdict"] for i, v in verdicts.items()}


def _value(outcome) -> float | str:
    if isinstance(outcome, Exception):
        return f"ERROR {type(outcome).__name__}: {outcome}"
    return round(float(outcome.value), 3)


async def score_one(metrics, q, answer, contexts):
    f, p, r, c = metrics
    out = await asyncio.gather(
        f.ascore(user_input=q.question, response=answer, retrieved_contexts=contexts),
        p.ascore(user_input=q.question, reference=q.notes, retrieved_contexts=contexts),
        r.ascore(user_input=q.question, retrieved_contexts=contexts, reference=q.notes),
        c.ascore(response=answer, reference=q.notes),
        return_exceptions=True,
    )
    names = ("faithfulness", "context_precision", "context_recall", "factual_correctness")
    return {n: _value(o) for n, o in zip(names, out, strict=True)}


async def main_async(args) -> int:
    questions = {q.id: q for q in load_questions()}
    records = json.loads((ROOT / "results" / f"answers_{args.variant}.json").read_text())["records"]
    sample = pick_sample(records, args.n)
    retrieve = make_variants(FastEmbedder(), [args.variant])[args.variant]
    humans = human_grades()

    tally = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    client = AsyncAnthropic()
    original = client.messages.create

    async def counted(*a, **kw):
        resp = await original(*a, **kw)
        tally["input_tokens"] += resp.usage.input_tokens
        tally["output_tokens"] += resp.usage.output_tokens
        tally["calls"] += 1
        return resp

    client.messages.create = counted
    llm = llm_factory(RAGAS_MODEL, provider="anthropic", client=client)
    metrics = (Faithfulness(llm=llm), ContextPrecision(llm=llm), ContextRecall(llm=llm), FactualCorrectness(llm=llm))

    rows = []
    for r in sample:
        q = questions[r["id"]]
        contexts = [h["text"] for h in retrieve(q.question, K)]
        scores = await score_one(metrics, q, r["answer"], contexts)
        rows.append({"id": r["id"], "answered": r["answerable"], "our_judge": r.get("judge", {}).get("verdict"),
                     "human": humans.get((r["id"], r["answer"])), "ragas": scores})
        print(json.dumps(rows[-1]), flush=True)
    out = {"variant": args.variant, "ragas_model": RAGAS_MODEL, "tokens": tally, "rows": rows}
    (ROOT / "results" / f"ragas_{args.variant}.json").write_text(json.dumps(out, indent=1) + "\n")
    print("TOKENS", json.dumps(tally))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="ctx_hybrid")
    parser.add_argument("--n", type=int, default=8)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
