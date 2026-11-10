"""Score retrieval variants on the contextual index (chunks embedded / BM25-indexed with their notes).

Usage: PYTHONPATH=scripts uv run --active python scripts/eval_contextual.py
Writes results/ablation_contextual.json. Quote matching and reranking use the ORIGINAL chunk text.
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
OUT = ROOT / "results" / "ablation_contextual.json"
RERANK_CANDIDATES = 20


def main() -> int:
    table, embedder = open_index(ROOT / "index" / "lancedb_ctx"), FastEmbedder()
    rows = table.to_arrow().select(["arxiv_id", "chunk_index", "text", "contextual_text"]).to_pylist()
    bm25 = Bm25Index(rows, text_key="contextual_text")
    scorer = make_cross_encoder_scorer(RERANK_MODEL)

    dense = lambda q, k: search(table, embedder, q, k=k)
    sparse = lambda q, k: bm25_search(bm25, q, k=k)
    hybrid = lambda q, k: hybrid_search(q, dense=dense, sparse=sparse, k=k, pool=DEFAULT_POOL)
    variants = {
        "ctx_dense": dense, "ctx_bm25": sparse, "ctx_hybrid_rrf": hybrid,
        "ctx_hybrid_rrf+rerank[bge-reranker-base]":
            lambda q, k: rerank(q, hybrid(q, RERANK_CANDIDATES), scorer=scorer, k=k),
    }
    questions, report = load_questions(), {}
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
        print(f"{name:<44} recall@1/3/5/10 = {[r['recall'][f'recall@{k}'] for k in KS]}  "
              f"paper@5 = {r['paper_recall']['paper_recall@5']}  median {r['median_latency_s']}s", flush=True)
    OUT.write_text(json.dumps({"variants": report}, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
