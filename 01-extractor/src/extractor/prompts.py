from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str, version: str) -> str:
    """Load a versioned prompt template from prompts/<name>_<version>.md."""
    path = PROMPTS_DIR / f"{name}_{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt not found: {name} {version} ({path})")
    return path.read_text(encoding="utf-8")
