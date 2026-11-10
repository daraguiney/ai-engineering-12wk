"""Contextual chunking: an LLM writes a short note placing each chunk in its paper.

The note is prepended to the chunk before embedding / BM25 indexing. The paper is sent as a
cached prefix so each extra chunk of the same paper only pays the cheap cache-read rate.
"""
import json
from pathlib import Path

from rag.chunking import Chunk

CONTEXT_MODEL = "claude-haiku-5-5"
MAX_OUTPUT_TOKENS = 200
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "context_v1.md"
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def build_request(document: str, chunk: str, model: str = CONTEXT_MODEL) -> dict:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").replace("{chunk}", chunk)
    return {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "thinking": {"type": "disabled"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"<document>\n{document}\n</document>",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }


def contextual_text(context: str, chunk_text: str) -> str:
    return f"{context}\n\n{chunk_text}"


def load_contexts(path: Path) -> dict[tuple[str, int], str]:
    if not path.is_file():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {(r["arxiv_id"], r["chunk_index"]): r["context"] for r in rows}


def _usage(response) -> dict[str, int]:
    return {k: getattr(response.usage, k, 0) or 0 for k in USAGE_KEYS}


def contextualize_paper(client, arxiv_id: str, document: str, chunks: list[Chunk], out_path: Path) -> dict[str, int]:
    """Write a context for every chunk not yet in `out_path`. Returns the token usage of this call.

    Sequential on purpose: the first request writes the cache that the rest of the paper reads.
    """
    done = load_contexts(out_path)
    total = dict.fromkeys(USAGE_KEYS, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        if (arxiv_id, chunk.index) in done:
            continue
        response = client.messages.create(**build_request(document, chunk.text))
        context = "".join(b.text for b in response.content if b.type == "text").strip()
        if not context:
            raise ValueError(f"empty context for {arxiv_id} chunk {chunk.index}")
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"arxiv_id": arxiv_id, "chunk_index": chunk.index, "context": context}) + "\n")
        total = {k: total[k] + v for k, v in _usage(response).items()}
    return total
