"""API request and response models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelId = Literal["before", "after"]
Tier = Literal["POSITIVE", "UNCERTAIN", "NEGATIVE"]
AuditStatus = Literal["VERIFIED", "FLAGGED", "DEFER"]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Metrics(ApiModel):
    topk_mass: float
    corner_ratio: float
    edge_ratio: float


class AnalyzeResponse(ApiModel):
    score: float
    tier: Tier
    metrics: Metrics
    overlay_png: str
    original_png: str
    evidence_hash: str


class ScoreResponse(ApiModel):
    score: float


class AuditRequest(ApiModel):
    evidence_hash: str = Field(min_length=1)
    # Optional local band (sliders). When POSITIVE, audit may run even if stored tier is not.
    call_tier: Tier | None = None


class AuditResponse(ApiModel):
    status: AuditStatus
    reason_lines: list[str]
    numbers_cited: list[str]
    defer_reason: str | None = None


class AskRequest(ApiModel):
    evidence_hash: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=300)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class AskResponse(ApiModel):
    answer: str | None = None
    error: str | None = None
