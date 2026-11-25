"""Pydantic v2 request, response, audit, and telemetry schemas for the inference gateway."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _require_non_blank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("Field must not be empty or whitespace-only.")
    return stripped


class EvidenceBreakdown(BaseModel):
    """Per-request grounding and conformal threshold evidence."""

    nli_entailment: float
    nli_contradiction: float
    nli_neutral: float
    cosine_sim: float
    entity_overlap: float
    length_ratio: float
    threshold_used: float | None = None


class QueryRequest(BaseModel):
    """Incoming inference request for the conformal-guarded router gateway."""

    query: str = Field(min_length=1)
    context: str | None = None
    alpha_override: float | None = Field(default=None, ge=0.01, le=0.30)
    lambda_cost_override: float | None = Field(default=None, ge=0.0, le=1.0)
    bypass_cache: bool = False

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        return _require_non_blank(value)

    @field_validator("context", mode="before")
    @classmethod
    def _normalize_context(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return value


class QueryResponse(BaseModel):
    """Gateway response returned by POST /v1/query."""

    request_id: str
    answer: str
    draft_answer: str | None = None
    model_used: str
    tier: Literal["tier1", "tier2"]
    provider: str
    escalated_from_tier1: bool = False
    cache_hit: bool = False
    fallback_used: bool = False
    estimated_cost_usd: float
    alpha_used: float
    lambda_cost_used: float
    calibrated_risk_score: float | None = None
    conformal_decision: Literal["PASS", "ABSTAIN", "UNSCORED"]
    evidence: EvidenceBreakdown | None = None
    latency_ms: float


class EvaluateRiskRequest(BaseModel):
    """Request payload for standalone conformal risk scoring."""

    query: str = Field(min_length=1)
    context: str = Field(min_length=1)
    candidate_answer: str = Field(min_length=1)
    alpha_override: float | None = Field(default=None, ge=0.01, le=0.30)

    @field_validator("query", "context", "candidate_answer")
    @classmethod
    def _validate_non_blank_fields(cls, value: str) -> str:
        return _require_non_blank(value)


class EvaluateRiskResponse(BaseModel):
    """Response payload for standalone conformal risk scoring."""

    calibrated_risk_score: float
    threshold_used: float
    alpha_used: float
    conformal_decision: Literal["PASS", "ABSTAIN"]
    evidence: EvidenceBreakdown


class AuditRecord(BaseModel):
    """Single logged gateway request record returned by audit endpoints."""

    model_config = ConfigDict(extra="ignore")

    request_id: str
    created_at: str
    query_text: str
    context_text: str | None = None
    answer_text: str
    draft_answer: str | None = None
    model_used: str
    tier: Literal["tier1", "tier2"]
    provider: str
    escalated_from_tier1: bool = False
    cache_hit: bool = False
    fallback_used: bool = False
    estimated_cost_usd: float
    frontier_cost_usd: float = 0.0
    alpha_used: float
    lambda_cost_used: float
    calibrated_risk_score: float | None = None
    conformal_decision: Literal["PASS", "ABSTAIN", "UNSCORED"]
    evidence: EvidenceBreakdown | None = None
    latency_ms: float


class AuditListResponse(BaseModel):
    """Paginated audit log response."""

    items: list[AuditRecord]
    total: int
    limit: int
    offset: int


class ParetoTelemetryResponse(BaseModel):
    """Aggregated live cost-quality-latency Pareto telemetry and offline curve."""

    total_requests: int
    cache_hit_rate: float
    abstention_rate: float
    avg_cost_usd: float
    estimated_frontier_cost_usd: float
    realized_cost_savings_pct: float
    p50_latency_ms: float
    p95_latency_ms: float
    tier1_share: float
    alpha_filter: float
    lambda_cost_filter: float
    live_points: list[dict[str, Any]]
    offline_curve: list[dict[str, Any]]


class BenchmarksSummaryResponse(BaseModel):
    """Offline Phase 1-3 benchmark artifact summary."""

    judge_audit: dict[str, Any]
    conformal: dict[str, Any]
    risk_metrics: dict[str, Any]
    pareto: dict[str, Any]


class HealthResponse(BaseModel):
    """Gateway readiness and dependency health status."""

    status: Literal["ok", "degraded"]
    artifacts: dict[str, bool]
    models: dict[str, bool]
    storage: dict[str, bool]
    providers: dict[str, bool]
