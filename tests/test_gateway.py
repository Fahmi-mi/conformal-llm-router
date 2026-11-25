"""Tests for Phase 4 unified async providers and FastAPI gateway endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.conformal.calibrator import ConformalBundle
from src.conformal.risk_model import RISK_FEATURE_COLS
from src.eval.features import QUERY_META_COLUMNS, FeatureExtractor
from src.gateway.cache import SemanticCache
from src.gateway.main import create_app
from src.gateway.providers import GatewaySettings, ProviderRouter, UpstreamUnavailableError
from src.gateway.schemas import (
    AuditListResponse,
    BenchmarksSummaryResponse,
    EvaluateRiskResponse,
    HealthResponse,
    ParetoTelemetryResponse,
    QueryResponse,
)
from src.gateway.storage import TelemetryStore
from src.router.trainer import RouterBundle

ABSTAIN_MESSAGE = "Abstained: generated response exceeded the calibrated hallucination risk bound."


class _StubEmbedder:
    """Deterministic unit-norm 384-d embedder for offline tests."""

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            lower = text.lower()
            if "weather" in lower:
                out[i, 0] = 1.0
            elif "capital" in lower or "paris" in lower:
                out[i, 1] = 1.0
            elif "quantum" in lower:
                out[i, 2] = 1.0
            else:
                out[i, 3 + (abs(hash(lower)) % 10)] = 1.0
        return out


class _StubNLIModel:
    """Deterministic NLI predictor keyed on response text markers."""

    def __init__(self) -> None:
        self.config = type("Cfg", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()

    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> np.ndarray:
        rows = []
        for _ctx, resp in pairs:
            lower = resp.lower()
            if "hallucinated" in lower or "unsupported" in lower:
                rows.append([0.85, 0.05, 0.10])
            elif "uncertain" in lower:
                rows.append([0.25, 0.55, 0.20])
            else:
                rows.append([0.02, 0.94, 0.04])
        return np.asarray(rows, dtype=np.float64)


class _StubRiskEstimator:
    """Maps NLI contradiction directly to hallucination probability."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # RISK_FEATURE_COLS has nli_contradiction at index 2
        contra_idx = RISK_FEATURE_COLS.index("nli_contradiction")
        risk = np.clip(X[:, contra_idx], 0.0, 1.0)
        return np.column_stack([1.0 - risk, risk])


class _StubRouterEstimator:
    """Routes short queries to Tier 1 (prob=0.70) and long/complex queries to Tier 2 (prob=0.20)."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # X has no PCA (pca=None) -> columns match QUERY_META_COLUMNS; char_len is col 0
        char_lens = X[:, 0]
        probs_t1 = np.where(char_lens > 80.0, 0.20, 0.70)
        return np.column_stack([1.0 - probs_t1, probs_t1])


def _make_bundles() -> tuple[ConformalBundle, RouterBundle]:
    calibrator = ConformalBundle(
        estimator=_StubRiskEstimator(),
        feature_cols=list(RISK_FEATURE_COLS),
        alpha_thresholds={"0.05": 0.20, "0.08": 0.30, "0.10": 0.35, "0.20": 0.60},
        default_alpha=0.10,
        alpha_min=0.01,
        alpha_max=0.30,
        model_name="stub-risk",
    )
    router = RouterBundle(
        estimator=_StubRouterEstimator(),
        query_meta_cols=list(QUERY_META_COLUMNS),
        pca=None,
        default_lambda=0.45,
        embedding_dim=384,
    )
    return calibrator, router


def _build_test_client(
    tmp_path: Path,
    handler: Any,
    *,
    groq_key: str | None = "test-groq",
    gemini_key: str | None = "test-gemini",
    ollama_url: str | None = "http://localhost:11434",
    deepseek_key: str | None = "test-deepseek",
    with_artifacts: bool = True,
    write_benchmark_files: bool = False,
) -> tuple[TestClient, TelemetryStore, SemanticCache, GatewaySettings]:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if write_benchmark_files:
        (artifacts_dir / "pareto_sweep.json").write_text(
            json.dumps(
                {
                    "status": "ready",
                    "curve": [{"lambda_cost": 0.45, "avg_cost": 0.001, "avg_quality": 0.78}],
                }
            ),
            encoding="utf-8",
        )
        (artifacts_dir / "conformal_sweep.json").write_text(
            json.dumps({"status": "ready", "sweep": []}), encoding="utf-8"
        )
        (artifacts_dir / "risk_metrics.json").write_text(
            json.dumps({"status": "ready"}), encoding="utf-8"
        )

    settings = GatewaySettings(
        deepseek_api_key=deepseek_key,
        groq_api_key=groq_key,
        gemini_api_key=gemini_key,
        ollama_base_url=ollama_url,
        gateway_alpha_default=0.10,
        gateway_lambda_default=0.45,
        artifacts_dir=artifacts_dir,
        telemetry_db_path=tmp_path / "telemetry.db",
        faiss_index_path=tmp_path / "faiss.index",
        similarity_threshold=0.92,
    )
    store = TelemetryStore(settings.telemetry_db_path)
    store.init_schema()
    cache = SemanticCache(
        settings.faiss_index_path,
        store=store,
        similarity_threshold=settings.similarity_threshold,
    )
    extractor = FeatureExtractor(embedder=_StubEmbedder(), nli_model=_StubNLIModel())
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    provider_router = ProviderRouter(settings=settings, client=async_client)

    calibrator, router = _make_bundles() if with_artifacts else (None, None)
    app = create_app(
        settings=settings,
        store=store,
        cache=cache,
        extractor=extractor,
        calibrator=calibrator,
        router=router,
        provider_router=provider_router,
    )
    return TestClient(app), store, cache, settings


def test_query_without_context_returns_unscored_and_context_bypasses_unscored_cache(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Paris is the capital of France."}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client, store, _, _ = _build_test_client(tmp_path, handler)

    # 1. No context -> UNSCORED, calibrated_risk_score is None
    resp1 = client.post("/v1/query", json={"query": "What is the capital of France?"})
    assert resp1.status_code == 200
    body1 = QueryResponse.model_validate(resp1.json())
    assert body1.conformal_decision == "UNSCORED"
    assert body1.calibrated_risk_score is None
    assert body1.evidence is None
    assert body1.cache_hit is False
    assert len(calls) == 1

    # 2. Blank whitespace context -> also UNSCORED, hits cache for same query without context
    resp2 = client.post(
        "/v1/query", json={"query": "What is the capital of France?", "context": "   "}
    )
    assert resp2.status_code == 200
    body2 = QueryResponse.model_validate(resp2.json())
    assert body2.conformal_decision == "UNSCORED"
    assert body2.calibrated_risk_score is None
    assert body2.cache_hit is True
    assert body2.estimated_cost_usd == 0.0
    assert len(calls) == 1

    # 3. Same query WITH context -> must NOT reuse UNSCORED cache entry
    resp3 = client.post(
        "/v1/query",
        json={
            "query": "What is the capital of France?",
            "context": "France's capital city is Paris.",
        },
    )
    assert resp3.status_code == 200
    body3 = QueryResponse.model_validate(resp3.json())
    assert body3.cache_hit is False
    assert body3.conformal_decision == "PASS"
    assert body3.calibrated_risk_score is not None
    assert body3.evidence is not None
    assert len(calls) == 2

    items, total = store.query_audit(decision="UNSCORED")
    assert total == 2
    assert len(items) == 2


def test_tier1_abstention_escalates_once_to_tier2_and_handles_double_abstention(
    tmp_path: Path,
) -> None:
    scenario = {"mode": "pass_tier2"}
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if "groq.com" in url:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "This is a hallucinated Tier 1 claim."}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                },
            )
        if "deepseek.com" in url:
            text = (
                "Paris is the capital of France."
                if scenario["mode"] == "pass_tier2"
                else "This is an unsupported Tier 2 claim."
            )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": text}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                },
            )
        return httpx.Response(500, json={"error": "unexpected"})

    client, store, cache, _ = _build_test_client(tmp_path, handler)

    # Case A: Tier 1 exceeds alpha -> 1x escalate to Tier 2 -> Tier 2 passes
    resp_a = client.post(
        "/v1/query",
        json={
            "query": "What is the capital of France?",
            "context": "France's capital city is Paris.",
        },
    )
    assert resp_a.status_code == 200
    body_a = QueryResponse.model_validate(resp_a.json())
    assert body_a.escalated_from_tier1 is True
    assert body_a.tier == "tier2"
    assert body_a.provider == "deepseek"
    assert body_a.conformal_decision == "PASS"
    assert body_a.answer == "Paris is the capital of France."
    assert body_a.estimated_cost_usd > 0.0
    assert len(calls) == 2

    # Case B: Tier 1 exceeds alpha -> 1x escalate to Tier 2 -> Tier 2 ALSO exceeds alpha -> ABSTAIN
    scenario["mode"] = "fail_tier2"
    calls.clear()
    resp_b = client.post(
        "/v1/query",
        json={
            "query": "Explain quantum entanglement simply.",
            "context": "Quantum entanglement links particle states across distance.",
        },
    )
    assert resp_b.status_code == 200
    body_b = QueryResponse.model_validate(resp_b.json())
    assert body_b.escalated_from_tier1 is True
    assert body_b.tier == "tier2"
    assert body_b.conformal_decision == "ABSTAIN"
    assert body_b.answer == ABSTAIN_MESSAGE
    assert body_b.draft_answer == "This is an unsupported Tier 2 claim."
    assert len(calls) == 2

    # Persisted with draft_answer in SQLite, and NOT cached in FAISS
    stored = store.get_request(body_b.request_id)
    assert stored is not None
    assert stored["draft_answer"] == "This is an unsupported Tier 2 claim."
    assert stored["conformal_decision"] == "ABSTAIN"
    assert cache._index.ntotal == 1  # only Case A (PASS) was cached


def test_cache_rethresholds_stricter_alpha_and_lambda_tier_shift_bypasses_cache(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if "groq.com" in url:
            # First call returns "uncertain" (contradiction=0.25: PASS at alpha=0.10 [thr=0.35],
            # PASS at alpha=0.08 [thr=0.30], ABSTAIN at alpha=0.05 [thr=0.20])
            msg = "An uncertain answer." if len(calls) == 1 else "A grounded Tier 1 answer."
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": msg}}]},
            )
        if "deepseek.com" in url:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Grounded Tier 2 answer."}}]},
            )
        return httpx.Response(500)

    client, _, _, _ = _build_test_client(tmp_path, handler)
    payload = {
        "query": "What is the weather today?",
        "context": "Today the weather is sunny and 22C.",
    }

    # 1. Populate cache at default alpha=0.10 (risk=0.25 < thr=0.35 -> PASS, tier1)
    r1 = QueryResponse.model_validate(client.post("/v1/query", json=payload).json())
    assert r1.conformal_decision == "PASS"
    assert r1.calibrated_risk_score == pytest.approx(0.25)
    assert r1.tier == "tier1"
    assert r1.cache_hit is False
    assert len(calls) == 1

    # 2. Slightly stricter alpha_override=0.08 (thr=0.30 > 0.25) -> valid cache hit with updated threshold
    r2 = QueryResponse.model_validate(
        client.post("/v1/query", json={**payload, "alpha_override": 0.08}).json()
    )
    assert r2.cache_hit is True
    assert r2.conformal_decision == "PASS"
    assert r2.alpha_used == pytest.approx(0.08)
    assert r2.evidence is not None and r2.evidence.threshold_used == pytest.approx(0.30)
    assert len(calls) == 1

    # 3. Stricter alpha_override=0.05 (thr=0.20 <= 0.25) -> must NOT return stale PASS;
    # bypasses cache and runs live pipeline (here Tier 1 call #2 returns grounded answer -> PASS)
    r3 = QueryResponse.model_validate(
        client.post("/v1/query", json={**payload, "alpha_override": 0.05}).json()
    )
    assert r3.cache_hit is False
    assert r3.alpha_used == pytest.approx(0.05)
    assert len(calls) == 2

    # 4. lambda_cost_override=0.0 shifts predicted_tier from tier1 (0.70 < 1.0) to tier2 -> bypasses cache
    r4 = QueryResponse.model_validate(
        client.post("/v1/query", json={**payload, "lambda_cost_override": 0.0}).json()
    )
    assert r4.cache_hit is False
    assert r4.tier == "tier2"
    assert r4.provider == "deepseek"
    assert len(calls) == 3

    # 5. bypass_cache=True forces live call even when cached PASS exists
    r5 = QueryResponse.model_validate(
        client.post("/v1/query", json={**payload, "bypass_cache": True}).json()
    )
    assert r5.cache_hit is False
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_provider_429_fallback_chain_and_503_on_total_failure(tmp_path: Path) -> None:
    # 1. Groq 429 -> Gemini 200 (fallback_used=True)
    def groq_429_gemini_ok(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "groq.com" in url:
            return httpx.Response(429, json={"error": "rate_limit"})
        if "generativelanguage.googleapis.com" in url:
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "Gemini fallback answer."}]}}],
                    "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 8},
                },
            )
        return httpx.Response(500)

    settings = GatewaySettings(
        groq_api_key="g-key",
        gemini_api_key="gem-key",
        ollama_base_url="http://localhost:11434",
        deepseek_api_key="ds-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(groq_429_gemini_ok)) as client:
        pr = ProviderRouter(settings=settings, client=client)
        res = await pr.generate("tier1", "Hi", None)
        assert res.provider == "gemini"
        assert res.fallback_used is True
        assert res.answer == "Gemini fallback answer."

    # 2. Groq ReadTimeout + Gemini 503 -> Ollama 200 (fallback_used=True)
    def groq_gemini_fail_ollama_ok(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "groq.com" in url:
            raise httpx.ReadTimeout("timed out", request=request)
        if "generativelanguage.googleapis.com" in url:
            return httpx.Response(503)
        if "localhost:11434/api/chat" in url:
            return httpx.Response(
                200,
                json={
                    "message": {"content": "Ollama local fallback answer."},
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(groq_gemini_fail_ollama_ok)) as client:
        pr = ProviderRouter(settings=settings, client=client)
        res = await pr.generate("tier1", "Hi", None)
        assert res.provider == "ollama"
        assert res.fallback_used is True
        assert res.answer == "Ollama local fallback answer."

    # 3. Total upstream failure raises UpstreamUnavailableError and gateway returns clean 503 JSON
    def all_fail(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(all_fail)) as client:
        pr = ProviderRouter(settings=settings, client=client)
        with pytest.raises(UpstreamUnavailableError):
            await pr.generate("tier1", "Hi", None)

    api_client, _, _, _ = _build_test_client(tmp_path, all_fail)
    resp = api_client.post("/v1/query", json={"query": "Will this fail cleanly?"})
    assert resp.status_code == 503
    assert "detail" in resp.json()


def test_auxiliary_endpoints_empty_and_populated_states(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "groq.com" in str(request.url):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Paris is the capital of France."}}]},
            )
        return httpx.Response(500)

    # Degraded empty state (missing artifacts & missing benchmark files)
    degraded_client, _, _, _ = _build_test_client(
        tmp_path / "empty",
        handler,
        with_artifacts=False,
        write_benchmark_files=False,
    )
    h_empty = HealthResponse.model_validate(degraded_client.get("/health").json())
    assert h_empty.status == "degraded"
    assert h_empty.artifacts["calibrator"] is False
    assert h_empty.storage["sqlite"] is True

    p_empty = ParetoTelemetryResponse.model_validate(
        degraded_client.get("/v1/telemetry/pareto").json()
    )
    assert p_empty.total_requests == 0
    assert p_empty.offline_curve == []

    a_empty = AuditListResponse.model_validate(degraded_client.get("/v1/telemetry/audit").json())
    assert a_empty.total == 0
    assert a_empty.items == []

    b_empty = BenchmarksSummaryResponse.model_validate(
        degraded_client.get("/v1/benchmarks/summary").json()
    )
    assert b_empty.judge_audit == {"status": "tbd"}
    assert b_empty.conformal == {"status": "tbd"}

    # Populated state with artifacts & benchmark files
    client, _, _, _ = _build_test_client(
        tmp_path / "full",
        handler,
        with_artifacts=True,
        write_benchmark_files=True,
    )
    h_ok = HealthResponse.model_validate(client.get("/health").json())
    assert h_ok.status == "ok"
    assert h_ok.artifacts == {"calibrator": True, "router": True}
    assert h_ok.providers["groq"] is True

    b_ok = BenchmarksSummaryResponse.model_validate(client.get("/v1/benchmarks/summary").json())
    assert b_ok.judge_audit == {"status": "tbd"}  # judge_audit_summary.json intentionally absent
    assert b_ok.conformal["status"] == "ready"
    assert b_ok.risk_metrics["status"] == "ready"
    assert b_ok.pareto["status"] == "ready"

    # Evaluate-risk standalone endpoint
    er_pass = EvaluateRiskResponse.model_validate(
        client.post(
            "/v1/evaluate-risk",
            json={
                "query": "What is the capital of France?",
                "context": "France's capital city is Paris.",
                "candidate_answer": "Paris is the capital.",
            },
        ).json()
    )
    assert er_pass.conformal_decision == "PASS"

    er_abstain = EvaluateRiskResponse.model_validate(
        client.post(
            "/v1/evaluate-risk",
            json={
                "query": "What is the capital of France?",
                "context": "France's capital city is Paris.",
                "candidate_answer": "A hallucinated answer.",
                "alpha_override": 0.05,
            },
        ).json()
    )
    assert er_abstain.conformal_decision == "ABSTAIN"
    assert er_abstain.threshold_used == pytest.approx(0.20)

    # Populate telemetry via /v1/query and check pareto + audit filters
    q_resp = client.post(
        "/v1/query",
        json={
            "query": "What is the capital of France?",
            "context": "France's capital city is Paris.",
        },
    )
    assert q_resp.status_code == 200

    p_pop = ParetoTelemetryResponse.model_validate(
        client.get("/v1/telemetry/pareto?alpha=0.10&lambda_cost=0.45").json()
    )
    assert p_pop.total_requests == 1
    assert len(p_pop.live_points) == 1
    assert len(p_pop.offline_curve) == 1

    a_pop = AuditListResponse.model_validate(
        client.get("/v1/telemetry/audit?decision=PASS&tier=tier1&q=capital").json()
    )
    assert a_pop.total == 1
    assert a_pop.items[0].conformal_decision == "PASS"
