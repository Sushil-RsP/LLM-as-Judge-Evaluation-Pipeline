
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class CriterionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: str
    score: int = Field(ge=1, le=5)
    rationale: str
    evidence: str


class PointwiseVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    scores: list[CriterionScore]
    overall_score: float = Field(ge=1, le=5)
    verdict: Literal["PASS", "FAIL", "REVIEW"]
    summary: str


class PairwiseVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: Literal["A", "B", "TIE"]
    confidence: int = Field(ge=1, le=5)
    rationale: str
    evidence_a: str
    evidence_b: str
