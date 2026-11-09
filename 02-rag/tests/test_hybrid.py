import pytest

from rag.hybrid import Bm25Index, bm25_search, hybrid_search, rerank, rrf_fuse, tokenize


def _hit(pid, idx, text="t", score=0.0):
    return {"arxiv_id": pid, "chunk_index": idx, "text": text, "score": score}


ROWS = [
    _hit("a", 0, "the zebra finch learns song from tutors"),
    _hit("a", 1, "rockets burn liquid hydrogen"),
    _hit("b", 0, "MMLU accuracy of 86.4 percent on the benchmark"),
]


def test_tokenize_lowercases_and_keeps_numbers():
    assert tokenize("MMLU: 86.4%, Self-Attention") == ["mmlu", "86", "4", "self", "attention"]


def test_bm25_ranks_exact_term_match_first():
    hits = bm25_search(Bm25Index(ROWS), "zebra finch", k=2)
    assert (hits[0]["arxiv_id"], hits[0]["chunk_index"]) == ("a", 0)
    assert len(hits) == 2


def test_bm25_no_overlap_still_returns_k_hits():
    assert len(bm25_search(Bm25Index(ROWS), "quantum", k=3)) == 3


def test_rrf_rewards_agreement_between_rankings():
    x, y, z = _hit("a", 0), _hit("a", 1), _hit("b", 0)
    fused = rrf_fuse([[x, y], [z, y]], k=3)
    assert (fused[0]["arxiv_id"], fused[0]["chunk_index"]) == ("a", 1)  # in both lists
    assert len(fused) == 3


def test_rrf_does_not_mutate_inputs():
    x = _hit("a", 0, score=0.9)
    rrf_fuse([[x]], k=1)
    assert x["score"] == 0.9


def test_rrf_rejects_empty_input():
    with pytest.raises(ValueError):
        rrf_fuse([], k=3)


def test_rerank_orders_by_scorer_and_truncates():
    hits = [_hit("a", 0, "low"), _hit("a", 1, "high"), _hit("b", 0, "mid")]
    ranked = rerank("q", hits, scorer=lambda q, texts: [{"low": 0.1, "high": 0.9, "mid": 0.5}[t] for t in texts], k=2)
    assert [h["text"] for h in ranked] == ["high", "mid"]
    assert ranked[0]["score"] == 0.9
    assert hits[0]["score"] == 0.0  # input untouched


def test_hybrid_search_combines_dense_and_bm25():
    dense = lambda q, k: [_hit("b", 0), _hit("a", 1)]
    bm25 = lambda q, k: [_hit("a", 0), _hit("b", 0)]
    fused = hybrid_search("q", dense=dense, sparse=bm25, k=3, pool=10)
    assert (fused[0]["arxiv_id"], fused[0]["chunk_index"]) == ("b", 0)
