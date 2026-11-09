"""Score each retrieval variant on the labelled test set, against the same index.

Usage: uv run --active python scripts/eval_ablation.py
Writes results/ablation_retrieval.json (per-variant recall@k, latency, per-question ranks).
"""
import json
import statistics
import time
from pathlib import Path

from eval_retrieval import load_questions

from rag.hybrid import (
    DEFAULT_POOL,
    RERANK_MODEL,
    Bm25Index,
    bm25_search,
    hybrid_search,
    make_cross_encoder_scorer,
    rerank,
)
from rag.index import FastEmbedder, open_index, search
from rag.retrieval_eval import KS, recall_at_k, score_question

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ablation_retrieval.json"
RERANK_CANDIDATES = 20


def main() -> int:
    table, embedder = open_index(ROOT / "index" / "lancedb"), FastEmbedder()
    bm25 = Bm25Index(table.to_arrow().select(["arxiv_id", "chunk_index", "text"]).to_pylist())
    scorer = make_cross_encoder_scorer()

    dense = lambda q, k: search(table, embedder, q, k=k)
    sparse = lambda q, k: bm25_search(bm25, q, k=k)
    hybrid = lambda q, k: hybrid_search(q, dense=dense, sparse=sparse, k=k, pool=DEFAULT_POOL)
    reranked = lambda q, k: rerank(q, hybrid(q, RERANK_CANDIDATES), scorer=scorer, k=k)
    variants = {"dense": dense, "bm25": sparse, "hybrid_rrf": hybrid, "hybrid_rrf+rerank": reranked}

    questions = load_questions()
    report = {}
    for name, retrieve in variants.items():
        latencies, results = [], []
        for q in questions:
            started = time.monotonic()
            results.append(score_question(q, table, embedder, max_k=max(KS), retrieve=retrieve))
            latencies.append(time.monotonic() - started)
        report[name] = {
            "recall": {f"recall@{k}": round(recall_at_k(results, k), 3) for k in KS},
            "paper_recall": {f"paper_recall@{k}": round(recall_at_k(results, k, "paper_rank"), 3) for k in KS},
            "median_latency_s": round(statistics.median(latencies), 3),
            "per_question": results,
        }
        r = report[name]
        print(f"{name:<20} recall@1/3/5/10 = {[r['recall'][f'recall@{k}'] for k in KS]}  "
              f"paper@5 = {r['paper_recall']['paper_recall@5']}  median latency {r['median_latency_s']}s")
    OUT.write_text(json.dumps({"rerank_model": RERANK_MODEL, "rerank_candidates": RERANK_CANDIDATES,
                               "pool": DEFAULT_POOL, "variants": report}, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
