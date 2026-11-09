"""Build the dense-retrieval baseline index and score it on the labelled test set.

Usage: uv run python scripts/eval_retrieval.py [--reuse-index]
Writes results/baseline_dense.json (config, recall@k, per-question ranks).
"""
import argparse
import json
import time
from pathlib import Path

from rag.chunking import DEFAULT_OVERLAP, DEFAULT_SIZE
from rag.index import EMBED_MODEL, FastEmbedder, build_index, open_index
from rag.retrieval_eval import KS, recall_at_k, score_question
from rag.testset import Question, validate_questions

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "index" / "lancedb"
OUT = ROOT / "results" / "baseline_dense.json"


def load_questions() -> list[Question]:
    rows = json.loads((ROOT / "testset" / "questions.json").read_text(encoding="utf-8"))
    return [Question(**{k: v for k, v in r.items() if k in Question.model_fields}) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-index", action="store_true")
    args = parser.parse_args()

    kept = json.loads((ROOT / "corpus" / "manifest.json").read_text())["kept"]
    texts = {k["arxiv_id"]: (ROOT / "corpus" / "text" / f"{k['arxiv_id']}.txt").read_text() for k in kept}
    questions = load_questions()
    problems = validate_questions(questions, texts)
    if problems:
        raise SystemExit(f"test set has invalid questions: {problems}")

    embedder = FastEmbedder()
    started = time.monotonic()
    table = open_index(DB) if args.reuse_index else build_index(texts, embedder, DB)
    index_s = time.monotonic() - started
    print(f"index: {table.count_rows()} chunks over {len(texts)} papers ({index_s:.0f}s)")

    results = [score_question(q, table, embedder, max_k=max(KS)) for q in questions]
    recall = {f"recall@{k}": round(recall_at_k(results, k), 3) for k in KS}
    paper_recall = {f"paper_recall@{k}": round(recall_at_k(results, k, "paper_rank"), 3) for k in KS}
    answerable = sum(1 for r in results if r["type"] != "unanswerable")
    print(f"answerable questions scored: {answerable}")
    for name, value in {**recall, **paper_recall}.items():
        print(f"  {name:<18} {value:.3f}")

    OUT.write_text(json.dumps({
        "config": {"embedder": EMBED_MODEL, "chunk_size": DEFAULT_SIZE, "overlap": DEFAULT_OVERLAP,
                   "chunks": table.count_rows(), "metric": "cosine", "answerable_questions": answerable},
        "recall": recall, "paper_recall": paper_recall, "per_question": results,
    }, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
