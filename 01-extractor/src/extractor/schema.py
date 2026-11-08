from pydantic import BaseModel, ConfigDict, Field, field_validator


class PaperExtraction(BaseModel):
    """Structured facts extracted from an arXiv paper."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Exact paper title")
    authors: list[str] = Field(min_length=1, description="Author names in order")
    problem: str = Field(description="The problem the paper addresses, one or two sentences")
    method: str = Field(description="The proposed approach, one or two sentences")
    datasets: list[str] = Field(default_factory=list, description="Datasets/benchmarks used")
    key_results: list[str] = Field(
        default_factory=list, description="Headline quantitative results, with numbers"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Limitations the authors state; empty if none"
    )

    @field_validator("title", "problem", "method")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()
