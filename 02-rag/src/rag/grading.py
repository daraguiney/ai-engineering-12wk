"""Blind human-grading sample for checking the answer judge, and the judge-vs-human comparison."""
import random

HUMAN_VERDICTS = ("correct", "partial", "incorrect")


def build_grading_sample(
    records_by_variant: dict[str, list[dict]],
    questions: list[dict],
    titles: dict[str, str],
    *,
    seed: int,
    n_correct: int = 8,
) -> tuple[list[dict], dict[str, dict]]:
    """Every judged-partial answer plus `n_correct` random judged-correct ones, shuffled.

    Returns (items for the grading page, key mapping item id -> hidden judge verdict).
    The items carry no judge verdict and no variant name, so grading is blind.
    """
    by_id = {q["id"]: q for q in questions}
    seen: set[tuple[str, str]] = set()
    pool: dict[str, list[tuple[dict, str, str]]] = {"correct": [], "partial": []}
    for variant, records in records_by_variant.items():
        for r in records:
            verdict = r.get("judge", {}).get("verdict")
            if verdict not in pool or (r["id"], r["answer"]) in seen:
                continue
            seen.add((r["id"], r["answer"]))
            pool[verdict].append((r, variant, verdict))
    rng = random.Random(seed)
    chosen = pool["partial"] + rng.sample(pool["correct"], min(n_correct, len(pool["correct"])))
    rng.shuffle(chosen)
    items, key = [], {}
    for n, (r, variant, verdict) in enumerate(chosen, start=1):
        q = by_id[r["id"]]
        item_id = f"g{n:02d}"
        items.append({
            "id": item_id, "type": q["type"], "question": q["question"], "reference": q["notes"],
            "evidence": [{"arxiv_id": e["arxiv_id"], "title": titles.get(e["arxiv_id"], ""), "quote": e["quote"]}
                         for e in q["evidence"]],
            "answer": r["answer"],
        })
        key[item_id] = {"question_id": q["id"], "judge": verdict, "variant": variant}
    return items, key


def agreement(key: dict[str, dict], verdicts: dict[str, dict]) -> dict:
    graded = [i for i in key if i in verdicts]
    if not graded:
        raise ValueError("no human grades to compare yet")
    disagreements = [
        {"id": i, "question_id": key[i]["question_id"], "judge": key[i]["judge"],
         "human": verdicts[i]["verdict"], "note": verdicts[i].get("note", "")}
        for i in graded if verdicts[i]["verdict"] != key[i]["judge"]
    ]
    return {"graded": len(graded), "agree": len(graded) - len(disagreements), "disagreements": disagreements}
