import json
from pathlib import Path

from extractor.schema import PaperExtraction


def load_record(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_record(
    directory: Path, arxiv_id: str, source_text: str, draft: PaperExtraction
) -> Path:
    """Write an unreviewed golden record. Never overwrites a reviewed one."""
    path = Path(directory) / f"{arxiv_id}.json"
    if path.exists() and load_record(path).get("reviewed"):
        raise FileExistsError(f"{path} is already reviewed; refusing to overwrite")
    record = {
        "arxiv_id": arxiv_id,
        "source_text": source_text,
        "expected": draft.model_dump(),
        "reviewed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
