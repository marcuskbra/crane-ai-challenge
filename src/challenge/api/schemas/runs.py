"""Run execution request/response schemas."""

from pydantic import BaseModel, Field, field_validator


class RunCreate(BaseModel):
    """
    Request model for creating a new run.

    Attributes:
        prompt: Natural language task description

    """

    prompt: str = Field(
        ...,
        min_length=1,
        description="Natural language task to execute",
        examples=["calculate 2 + 3", "add todo Buy milk"],
    )

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only")
        return v.strip()
