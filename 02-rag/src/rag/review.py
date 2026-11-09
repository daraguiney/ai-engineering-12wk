"""State for the local question-review app: questions in, verdicts out."""
import json
import os
from pathlib import Path

VERDICTS = ("ok", "fix", "drop")
MAX_NOTE_CHARS = 5000


def _read_json(path: Path, what: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} file {path} is not valid JSON: {exc}") from exc


def load_state(questions_path: Path, verdicts_path: Path) -> dict:
    questions = _read_json(questions_path, "questions")
    verdicts = _read_json(verdicts_path, "verdicts") if verdicts_path.is_file() else {}
    return {"questions": questions, "verdicts": verdicts}


def save_verdict(
    questions_path: Path, verdicts_path: Path, question_id: str, verdict: str, note: str
) -> dict:
    """Record one verdict, replacing any earlier one for that question. Returns all verdicts."""
    state = load_state(questions_path, verdicts_path)
    if question_id not in {q["id"] for q in state["questions"]}:
        raise ValueError(f"unknown question id: {question_id!r}")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    if not isinstance(note, str) or len(note) > MAX_NOTE_CHARS:
        raise ValueError(f"note must be text of at most {MAX_NOTE_CHARS} characters")
    updated = {**state["verdicts"], question_id: {"verdict": verdict, "note": note}}
    tmp = verdicts_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(updated, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, verdicts_path)  # atomic: a crash never leaves a half-written file
    return updated
