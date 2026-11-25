"""Tests for Phase 4 gateway Pydantic schemas, SQLite telemetry store, and FAISS semantic cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.gateway.cache import SemanticCache
from src.gateway.schemas import (
    AuditListResponse,
    AuditRecord,
    BenchmarksSummaryResponse,
    EvaluateRiskRequest,
    EvaluateRiskResponse,
    EvidenceBreakdown,
    HealthResponse,
    ParetoTelemetryResponse,
    QueryRequest,
    QueryResponse,
)
from src.gateway.storage import TelemetryStore


def _unit_vec(dim: int, idx: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def _make_record(
    request_id: str,
    *,
    query_text: str = "What is conformal prediction?",
    context_text: str | None = "Conformal prediction provides finite-sample coverage guarantees.",
    answer_text: str = "Conformal prediction offers finite-sample coverage guarantees.",
    draft_answer: str | None = None,
    model_used: str = "llama-3.1-8b-instant",
    tier: str = "tier1",
    provider: str = "groq",
    escalated_from_tier1: bool = False,
    cache_hit: bool = False,
    fallback_used: bool = False,
    estimated_cost_usd: float = 0.0004,
    frontier_cost_usd: float = 0.0020,
    alpha_used: float = 0.10,
    lambda_cost_used: float = 0.45,
    calibrated_risk_score: float | None = 0.12,
    conformal_decision: str = "PASS",
    evidence: dict | None = None,
    latency_ms: float = 42.0,
    created_at: str = "2026-11-25T10:00:00+00:00",
) -> dict:
    if evidence is None and conformal_decision != "UNSCORED":
        evidence = {
            "nli_entailment": 0.91,
            "nli_contradiction": 0.02,
            "nli_neutral": 0.07,
            "cosine_sim": 0.88,
            "entity_overlap": 0.80,
            "length_ratio": 0.95,
            "threshold_used": 0.35,
        }
    return {
        "request_id": request_id,
        "created_at": created_at,
        "query_text": query_text,
        "context_text": context_text,
        "answer_text": answer_text,
        "draft_answer": draft_answer,
        "model_used": model_used,
        "tier": tier,
        "provider": provider,
        "escalated_from_tier1": escalated_from_tier1,
        "cache_hit": cache_hit,
        "fallback_used": fallback_used,
        "estimated_cost_usd": estimated_cost_usd,
        "frontier_cost_usd": frontier_cost_usd,
        "alpha_used": alpha_used,
        "lambda_cost_used": lambda_cost_used,
        "calibrated_risk_score": calibrated_risk_score,
        "conformal_decision": conformal_decision,
        "evidence": evidence,
        "latency_ms": latency_ms,
    }


def test_query_request_and_evaluate_risk_validation() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="")
    with pytest.raises(ValidationError):
        QueryRequest(query="   \t\n  ")
    with pytest.raises(ValidationError):
        QueryRequest(query="valid", alpha_override=0.005)
    with pytest.raises(ValidationError):
        QueryRequest(query="valid", alpha_override=0.31)
    with pytest.raises(ValidationError):
        QueryRequest(query="valid", lambda_cost_override=-0.01)
    with pytest.raises(ValidationError):
        QueryRequest(query="valid", lambda_cost_override=1.01)

    req = QueryRequest(
        query="  Explain split conformal prediction  ",
        alpha_override=0.10,
        lambda_cost_override=0.50,
    )
    assert req.query == "Explain split conformal prediction"
    assert req.bypass_cache is False

    for bad_kwargs in (
        {"query": "  ", "context": "ctx", "candidate_answer": "ans"},
        {"query": "q", "context": "\n\t", "candidate_answer": "ans"},
        {"query": "q", "context": "ctx", "candidate_answer": "   "},
        {"query": "q", "context": "ctx", "candidate_answer": "ans", "alpha_override": 0.5},
    ):
        with pytest.raises(ValidationError):
            EvaluateRiskRequest(**bad_kwargs)


def test_response_and_telemetry_schemas_roundtrip() -> None:
    evidence = EvidenceBreakdown(
        nli_entailment=0.9,
        nli_contradiction=0.05,
        nli_neutral=0.05,
        cosine_sim=0.85,
        entity_overlap=0.75,
        length_ratio=1.0,
        threshold_used=0.32,
    )
    q_resp = QueryResponse(
        request_id="req-1",
        answer="Yes",
        model_used="llama-3.1-8b-instant",
        tier="tier1",
        provider="groq",
        estimated_cost_usd=0.0002,
        alpha_used=0.10,
        lambda_cost_used=0.45,
        calibrated_risk_score=0.11,
        conformal_decision="PASS",
        evidence=evidence,
        latency_ms=25.0,
    )
    assert q_resp.cache_hit is False

    eval_resp = EvaluateRiskResponse(
        calibrated_risk_score=0.11,
        threshold_used=0.32,
        alpha_used=0.10,
        conformal_decision="PASS",
        evidence=evidence,
    )
    assert eval_resp.conformal_decision == "PASS"

    audit_rec = AuditRecord(**_make_record("req-1"))
    audit_list = AuditListResponse(items=[audit_rec], total=1, limit=50, offset=0)
    assert audit_list.total == 1

    pareto = ParetoTelemetryResponse(
        total_requests=0,
        cache_hit_rate=0.0,
        abstention_rate=0.0,
        avg_cost_usd=0.0,
        estimated_frontier_cost_usd=0.0,
        realized_cost_savings_pct=0.0,
        p50_latency_ms=0.0,
        p95_latency_ms=0.0,
        tier1_share=0.0,
        alpha_filter=0.10,
        lambda_cost_filter=0.45,
        live_points=[],
        offline_curve=[],
    )
    assert pareto.total_requests == 0

    bench = BenchmarksSummaryResponse(
        judge_audit={"agree": 0.9},
        conformal={"alpha": 0.1},
        risk_metrics={"auroc": 0.9},
        pareto={"auc_qc": 0.75},
    )
    assert bench.pareto["auc_qc"] == 0.75

    health = HealthResponse(
        status="ok",
        artifacts={"calibrator": True},
        models={"nli": True},
        storage={"sqlite": True},
        providers={"groq": True},
    )
    assert health.status == "ok"


def test_semantic_cache_ignores_abstain_and_hits_pass_and_unscored(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry.db"
    index_path = tmp_path / "cache.faiss"
    store = TelemetryStore(db_path)
    store.init_schema()

    cache = SemanticCache(index_path=index_path, store=store, similarity_threshold=0.92, embedding_dim=16)

    abstain_rec = _make_record("req-abstain", conformal_decision="ABSTAIN", calibrated_risk_score=0.88)
    store.insert_request(abstain_rec)
    v_abstain = _unit_vec(16, 0)
    cache.put(v_abstain, abstain_rec)
    assert cache.lookup(v_abstain) is None

    pass_rec = _make_record("req-pass", conformal_decision="PASS")
    store.insert_request(pass_rec)
    v_pass = _unit_vec(16, 1) * 5.0  # unnormalized input should be L2-normalized
    cache.put(v_pass, pass_rec)

    # Cosine similarity = cos(theta) >= 0.92 should hit
    v_near = np.zeros(16, dtype=np.float32)
    v_near[1] = 0.95
    v_near[2] = float(np.sqrt(1.0 - 0.95**2))
    hit = cache.lookup(v_near)
    assert hit is not None
    assert hit["request_id"] == "req-pass"

    # Cosine similarity < 0.92 should miss
    v_far = np.zeros(16, dtype=np.float32)
    v_far[1] = 0.85
    v_far[2] = float(np.sqrt(1.0 - 0.85**2))
    assert cache.lookup(v_far) is None

    # UNSCORED entries may also be cached and hit
    unscored_rec = _make_record(
        "req-unscored",
        context_text=None,
        calibrated_risk_score=None,
        conformal_decision="UNSCORED",
        evidence=None,
    )
    store.insert_request(unscored_rec)
    v_unscored = _unit_vec(16, 3)
    cache.put(v_unscored, unscored_rec)
    hit_unscored = cache.lookup(v_unscored)
    assert hit_unscored is not None
    assert hit_unscored["request_id"] == "req-unscored"

    # Zero-norm and NaN embeddings handled safely without crashing
    assert cache.lookup(np.zeros(16, dtype=np.float32)) is None
    cache.put(np.zeros(16, dtype=np.float32), _make_record("req-zero"))

    with pytest.raises(ValueError):
        cache.lookup(np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError):
        cache.put(np.ones(8, dtype=np.float32), pass_rec)

    # Persistence across new SemanticCache instance
    reloaded = SemanticCache(
        index_path=index_path, store=store, similarity_threshold=0.92, embedding_dim=16
    )
    persisted_hit = reloaded.lookup(v_near)
    assert persisted_hit is not None
    assert persisted_hit["request_id"] == "req-pass"


def test_telemetry_store_query_audit_filters_and_pagination(tmp_path: Path) -> None:
    store = TelemetryStore(tmp_path / "telemetry.db")
    store.init_schema()

    records = [
        _make_record(
            "r1",
            query_text="Capital of France",
            tier="tier1",
            cache_hit=False,
            conformal_decision="PASS",
            created_at="2026-11-25T10:00:01+00:00",
        ),
        _make_record(
            "r2",
            query_text="Quantum gravity proof",
            tier="tier2",
            cache_hit=False,
            conformal_decision="ABSTAIN",
            created_at="2026-11-25T10:00:02+00:00",
        ),
        _make_record(
            "r3",
            query_text="Capital of Japan",
            tier="tier1",
            cache_hit=True,
            conformal_decision="PASS",
            created_at="2026-11-25T10:00:03+00:00",
        ),
        _make_record(
            "r4",
            query_text="Write a haiku",
            context_text=None,
            tier="tier2",
            cache_hit=False,
            conformal_decision="UNSCORED",
            calibrated_risk_score=None,
            evidence=None,
            created_at="2026-11-25T10:00:04+00:00",
        ),
    ]
    for rec in records:
        store.insert_request(rec)

    fetched = store.get_request("r1")
    assert fetched is not None
    assert fetched["evidence"]["nli_entailment"] == pytest.approx(0.91)
    assert store.get_request("missing") is None

    pass_items, pass_total = store.query_audit(decision="PASS")
    assert pass_total == 2
    assert [r["request_id"] for r in pass_items] == ["r3", "r1"]

    abstain_items, abstain_total = store.query_audit(decision="ABSTAIN")
    assert abstain_total == 1
    assert abstain_items[0]["request_id"] == "r2"

    unscored_items, unscored_total = store.query_audit(decision="UNSCORED")
    assert unscored_total == 1
    assert unscored_items[0]["request_id"] == "r4"

    tier1_cached, tier1_cached_total = store.query_audit(tier="tier1", cache_hit=True)
    assert tier1_cached_total == 1
    assert tier1_cached[0]["request_id"] == "r3"

    search_items, search_total = store.query_audit(q="capital", limit=1, offset=1)
    assert search_total == 2
    assert len(search_items) == 1
    assert search_items[0]["request_id"] == "r1"


def test_telemetry_store_aggregate_pareto_empty_and_populated(tmp_path: Path) -> None:
    store = TelemetryStore(tmp_path / "telemetry.db")
    store.init_schema()

    empty = store.aggregate_pareto(alpha=0.10, lambda_cost=0.45)
    assert empty["total_requests"] == 0
    assert empty["cache_hit_rate"] == 0.0
    assert empty["abstention_rate"] == 0.0
    assert empty["avg_cost_usd"] == 0.0
    assert empty["estimated_frontier_cost_usd"] == 0.0
    assert empty["realized_cost_savings_pct"] == 0.0
    assert empty["p50_latency_ms"] == 0.0
    assert empty["p95_latency_ms"] == 0.0
    assert empty["tier1_share"] == 0.0
    assert empty["alpha_filter"] == pytest.approx(0.10)
    assert empty["lambda_cost_filter"] == pytest.approx(0.45)
    assert empty["live_points"] == []
    assert empty["offline_curve"] == []

    # Populate 4 rows at alpha=0.10, lambda_cost=0.45 and 1 row at different settings
    store.insert_request(
        _make_record(
            "p1",
            tier="tier1",
            cache_hit=False,
            conformal_decision="PASS",
            estimated_cost_usd=0.001,
            frontier_cost_usd=0.004,
            latency_ms=10.0,
        )
    )
    store.insert_request(
        _make_record(
            "p2",
            tier="tier1",
            cache_hit=True,
            conformal_decision="PASS",
            estimated_cost_usd=0.0,
            frontier_cost_usd=0.004,
            latency_ms=20.0,
        )
    )
    store.insert_request(
        _make_record(
            "p3",
            tier="tier2",
            cache_hit=False,
            conformal_decision="ABSTAIN",
            estimated_cost_usd=0.004,
            frontier_cost_usd=0.004,
            latency_ms=30.0,
        )
    )
    store.insert_request(
        _make_record(
            "p4",
            tier="tier1",
            cache_hit=False,
            conformal_decision="PASS",
            estimated_cost_usd=0.001,
            frontier_cost_usd=0.004,
            latency_ms=40.0,
        )
    )
    store.insert_request(
        _make_record(
            "other",
            tier="tier2",
            alpha_used=0.05,
            lambda_cost_used=0.80,
            estimated_cost_usd=0.004,
            frontier_cost_usd=0.004,
            latency_ms=100.0,
        )
    )

    agg = store.aggregate_pareto(alpha=0.10, lambda_cost=0.45)
    assert agg["total_requests"] == 4
    assert agg["cache_hit_rate"] == pytest.approx(0.25)
    assert agg["abstention_rate"] == pytest.approx(0.25)
    assert agg["avg_cost_usd"] == pytest.approx(0.0015)
    assert agg["estimated_frontier_cost_usd"] == pytest.approx(0.004)
    assert agg["realized_cost_savings_pct"] == pytest.approx(62.5)
    assert agg["tier1_share"] == pytest.approx(0.75)
    assert agg["p50_latency_ms"] == pytest.approx(25.0)
    assert agg["p95_latency_ms"] == pytest.approx(38.5)
    assert len(agg["live_points"]) == 2
