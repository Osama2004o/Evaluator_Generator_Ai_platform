from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    is_acceptable: bool = Field(description="Whether the answer can be returned to the user")
    accuracy: int = Field(ge=0, le=10, description="Accuracy relative to retrieved evidence")
    relevance: int = Field(ge=0, le=10, description="Relevance to the question")
    completeness: int = Field(ge=0, le=10, description="Coverage of the requested information")
    consistency: int = Field(ge=0, le=10, description="Consistency with the retrieved evidence")
    grounding: int = Field(ge=0, le=10, description="Support for factual claims in the evidence")
    unsupported_claims: list[str] = Field(default_factory=list)
    feedback: str = Field(description="Specific, actionable feedback if rejected")
