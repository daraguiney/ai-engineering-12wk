import instructor
from anthropic import Anthropic

from extractor.prompts import load_prompt
from extractor.schema import PaperExtraction

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_MAX_RETRIES = 3
MAX_OUTPUT_TOKENS = 2048


def make_client() -> instructor.Instructor:
    """Instructor-wrapped Anthropic client; reads ANTHROPIC_API_KEY from env."""
    # JSON mode: this model rejects the forced tool_choice used by TOOLS mode.
    return instructor.from_anthropic(Anthropic(), mode=instructor.Mode.ANTHROPIC_JSON)


def extract_paper(
    document: str,
    *,
    client,
    model: str = DEFAULT_MODEL,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> PaperExtraction:
    """Extract a PaperExtraction; Instructor re-asks on validation failure."""
    if not document.strip():
        raise ValueError("document must not be empty")
    prompt = load_prompt("extract_paper", prompt_version).replace("{document}", document)
    return client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=max_retries,
        response_model=PaperExtraction,
        messages=[{"role": "user", "content": prompt}],
    )
