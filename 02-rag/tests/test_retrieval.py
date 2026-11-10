import hashlib
import math
import re

import pytest

from rag.index import build_index, search
from rag.retrieval_eval import recall_at_k, score_question
from rag.testset import Evidence, Question

DIM = 64


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, no model download."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


TEXTS = {
    "1111.1111": "Cats are small mammals. " * 30 + "The zebra finch song is learned from tutors. " * 3,
    "2222.2222": "Rockets burn liquid hydrogen with oxygen. " * 30 + "Orbital insertion needs 9.4 km/s. " * 3,
}


@pytest.fixture
def table(tmp_path):
    return build_index(TEXTS, FakeEmbedder(), tmp_path / "db", size=300, overlap=100)


def test_search_returns_ranked_chunks_with_metadata(table):
    hits = search(table, FakeEmbedder(), "how do rockets use hydrogen and oxygen", k=3)
    assert len(hits) == 3
    assert hits[0]["arxiv_id"] == "2222.2222"
    assert {"arxiv_id", "chunk_index", "text", "score"} <= set(hits[0])


def test_rebuilding_replaces_the_table(tmp_path):
    build_index(TEXTS, FakeEmbedder(), tmp_path / "db", size=300, overlap=100)
    again = build_index({"1111.1111": TEXTS["1111.1111"]}, FakeEmbedder(), tmp_path / "db", size=300, overlap=100)
    assert {h["arxiv_id"] for h in search(again, FakeEmbedder(), "rockets", k=50)} == {"1111.1111"}


def _q(qid, qtype, evidence):
    return Question(id=qid, type=qtype, question="orbital insertion velocity", evidence=evidence)


def test_score_question_finds_rank_of_first_chunk_containing_each_quote(table):
    q = _q("a", "single", [Evidence(arxiv_id="2222.2222", quote="Orbital insertion needs 9.4 km/s.")])
    result = score_question(q, table, FakeEmbedder(), max_k=10)
    assert result["rank"] is not None and result["rank"] <= 10
    assert result["paper_rank"] is not None


def test_score_question_miss_when_quote_not_retrievable(table):
    q = _q("b", "single", [Evidence(arxiv_id="1111.1111", quote="quote that is nowhere in the paper")])
    assert score_question(q, table, FakeEmbedder(), max_k=10)["rank"] is None


def test_cross_question_rank_is_the_worse_of_its_two_quotes(table):
    ev = [
        Evidence(arxiv_id="2222.2222", quote="Orbital insertion needs 9.4 km/s."),
        Evidence(arxiv_id="1111.1111", quote="The zebra finch song is learned from tutors."),
    ]
    result = score_question(_q("c", "cross", ev), table, FakeEmbedder(), max_k=50)
    assert result["rank"] == max(result["evidence_ranks"])


def test_recall_at_k_ignores_unanswerable_and_counts_misses():
    results = [
        {"id": "1", "type": "single", "rank": 1},
        {"id": "2", "type": "single", "rank": 4},
        {"id": "3", "type": "cross", "rank": None},
        {"id": "4", "type": "unanswerable", "rank": None},
    ]
    assert recall_at_k(results, 1) == pytest.approx(1 / 3)
    assert recall_at_k(results, 5) == pytest.approx(2 / 3)


def test_score_question_accepts_a_custom_retriever(table):
    q = _q("c", "single", [Evidence(arxiv_id="1111.1111", quote="Cats are small mammals.")])
    fake = lambda query, k: [{"arxiv_id": "1111.1111", "chunk_index": 0, "text": "Cats are small mammals."}]
    assert score_question(q, table, FakeEmbedder(), retrieve=fake)["rank"] == 1


def test_build_index_embeds_context_but_keeps_original_text(tmp_path):
    contexts = {("1111.1111", 0): "A note about cats."}
    t = build_index({"1111.1111": TEXTS["1111.1111"]}, FakeEmbedder(), tmp_path / "db", size=300, overlap=100, contexts=contexts)
    row = next(r for r in t.to_arrow().to_pylist() if r["chunk_index"] == 0)
    assert row["text"].startswith("Cats are small")
    assert row["contextual_text"].startswith("A note about cats.\n\nCats are small")
    other = next(r for r in t.to_arrow().to_pylist() if r["chunk_index"] == 1)
    assert other["contextual_text"] == other["text"]  # no note recorded -> unchanged
