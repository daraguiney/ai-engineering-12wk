"""Score retrieval against the labelled test set: a hit means a retrieved chunk contains the quote."""
from collections.abc import Callable

from rag.index import Embedder, search
from rag.testset import Question, quote_in_text

KS = (1, 3, 5, 10)
Retrieve = Callable[[str, int], list[dict]]


def _first_rank(hits: list[dict], arxiv_id: str, quote: str | None) -> int | None:
    """1-based rank of the first hit from `arxiv_id` (containing `quote`, if given)."""
    for rank, hit in enumerate(hits, start=1):
        if hit["arxiv_id"] == arxiv_id and (quote is None or quote_in_text(quote, hit["text"])):
            return rank
    return None


def score_question(
    question: Question, table, embedder: Embedder, max_k: int = 10, retrieve: Retrieve | None = None
) -> dict:
    """Rank of the evidence in the top `max_k`. For two-paper questions, the worse of the two."""
    run = retrieve or (lambda query, k: search(table, embedder, query, k=k))
    hits = run(question.question, max_k)
    ev_ranks = [_first_rank(hits, e.arxiv_id, e.quote) for e in question.evidence]
    paper_ranks = [_first_rank(hits, e.arxiv_id, None) for e in question.evidence]

    def worst(ranks: list[int | None]) -> int | None:
        return None if not ranks or any(r is None for r in ranks) else max(ranks)

    return {
        "id": question.id,
        "type": question.type,
        "rank": worst(ev_ranks),
        "paper_rank": worst(paper_ranks),
        "evidence_ranks": ev_ranks,
        "top_papers": [h["arxiv_id"] for h in hits[:5]],
    }


def recall_at_k(results: list[dict], k: int, key: str = "rank") -> float:
    """Share of answerable questions whose evidence is in the top k (unanswerable excluded)."""
    scored = [r for r in results if r["type"] != "unanswerable"]
    if not scored:
        raise ValueError("no answerable questions to score")
    return sum(1 for r in scored if r[key] is not None and r[key] <= k) / len(scored)
