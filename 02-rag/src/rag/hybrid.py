"""Keyword (BM25) search, rank fusion and cross-encoder reranking over the chunk table."""
import re
from collections.abc import Callable

from rank_bm25 import BM25Okapi

RRF_C = 60  # standard reciprocal-rank-fusion damping constant
DEFAULT_POOL = 50  # candidates taken from each retriever before fusing / reranking
RERANK_MODEL = "BAAI/bge-reranker-base"

Hit = dict
Retrieve = Callable[[str, int], list[Hit]]
Scorer = Callable[[str, list[str]], list[float]]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Bm25Index:
    def __init__(self, rows: list[Hit]):
        if not rows:
            raise ValueError("no chunks to index")
        self._rows = rows
        self._bm25 = BM25Okapi([tokenize(r["text"]) for r in rows])


def bm25_search(index: Bm25Index, query: str, k: int = 5) -> list[Hit]:
    scores = index._bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**index._rows[i], "score": float(scores[i])} for i in order]


def rrf_fuse(rankings: list[list[Hit]], k: int, c: int = RRF_C) -> list[Hit]:
    """Reciprocal rank fusion: a chunk's score is the sum of 1/(c+rank) over the lists it is in."""
    if not rankings:
        raise ValueError("need at least one ranking to fuse")
    fused: dict[tuple, float] = {}
    first_seen: dict[tuple, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit["arxiv_id"], hit["chunk_index"])
            fused[key] = fused.get(key, 0.0) + 1.0 / (c + rank)
            first_seen.setdefault(key, hit)
    best = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
    return [{**first_seen[key], "score": fused[key]} for key in best]


def hybrid_search(query: str, *, dense: Retrieve, sparse: Retrieve, k: int, pool: int = DEFAULT_POOL) -> list[Hit]:
    return rrf_fuse([dense(query, pool), sparse(query, pool)], k=k)


def rerank(query: str, hits: list[Hit], *, scorer: Scorer, k: int) -> list[Hit]:
    """Re-order candidates by a (query, passage) relevance scorer and keep the best k."""
    if not hits:
        return []
    scores = scorer(query, [h["text"] for h in hits])
    ranked = sorted(zip(hits, scores, strict=True), key=lambda p: p[1], reverse=True)[:k]
    return [{**h, "score": float(s)} for h, s in ranked]


def make_cross_encoder_scorer(model_name: str = RERANK_MODEL) -> Scorer:
    """Local ONNX cross-encoder via fastembed (downloads the model on first use)."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    model = TextCrossEncoder(model_name=model_name)
    return lambda query, texts: [float(s) for s in model.rerank(query, texts)]
