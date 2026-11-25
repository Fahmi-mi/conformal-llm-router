"""FastAPI application factory and inference gateway endpoints."""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.conformal.calibrator import (
    ConformalBundle,
    _decide,
    _resolve_threshold,
    predict_risk_and_decision,
)
from src.eval.features import FeatureExtractor
from src.gateway.cache import SemanticCache
from src.gateway.providers import (
    GatewaySettings,
    ProviderRouter,
    UpstreamUnavailableError,
)
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
from src.router.trainer import RouterBundle, predict_tier1_probability, select_tier

logger = logging.getLogger(__name__)

_ABSTAIN_FALLBACK_TEXT = "Abstained: generated response exceeded the calibrated hallucination risk bound."


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _load_pickle_artifact(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except (
        OSError,
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        ValueError,
        ImportError,
    ) as exc:  # pragma: no cover - defensive guard for corrupt artifacts
        logger.warning("Failed to load pickle artifact %s: %s", path, exc)
        return None


def _load_json_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "tbd"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"status": "tbd"}
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load JSON artifact %s: %s", path, exc)
        return {"status": "tbd"}


def _load_offline_pareto_curve(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load offline Pareto sweep %s: %s", path, exc)
        return []
    if isinstance(payload, dict):
        curve = payload.get("curve", [])
        return curve if isinstance(curve, list) else []
    return payload if isinstance(payload, list) else []


def _build_evidence(
    rag_df: pd.DataFrame,
    threshold: float,
) -> EvidenceBreakdown:
    row = rag_df.iloc[0]
    return EvidenceBreakdown(
        nli_entailment=float(row["nli_entailment"]),
        nli_contradiction=float(row["nli_contradiction"]),
        nli_neutral=float(row["nli_neutral"]),
        cosine_sim=float(row["cosine_sim"]),
        entity_overlap=float(row["entity_overlap"]),
        length_ratio=float(row["length_ratio"]),
        threshold_used=float(threshold),
    )


def _score_candidate(
    extractor: FeatureExtractor,
    calibrator: ConformalBundle,
    query: str,
    context: str,
    answer: str,
    alpha: float,
) -> tuple[float, Literal["PASS", "ABSTAIN"], EvidenceBreakdown]:
    rag_df = extractor.extract_rag_features([query], [context], [answer])
    risk_score, decision, threshold = predict_risk_and_decision(
        calibrator, rag_df, alpha=alpha
    )
    return float(risk_score), decision, _build_evidence(rag_df, threshold)


def _check_valid_cache_hit(
    cached: dict[str, Any] | None,
    *,
    context_text: str | None,
    predicted_tier: Literal["tier1", "tier2"],
    lambda_cost_override: float | None,
    alpha_used: float,
    calibrator: ConformalBundle,
) -> tuple[bool, EvidenceBreakdown | None]:
    if cached is None:
        return False, None

    cached_ctx = _normalize_optional_text(cached.get("context_text"))
    if context_text != cached_ctx:
        return False, None

    if context_text is not None and cached.get("conformal_decision") == "UNSCORED":
        return False, None

    if lambda_cost_override is not None and predicted_tier != cached.get("tier"):
        return False, None

    if context_text is None:
        return True, None

    cached_risk = cached.get("calibrated_risk_score")
    if cached_risk is None:
        return False, None

    _, threshold = _resolve_threshold(calibrator, alpha_used)
    if _decide(float(cached_risk), float(threshold)) == "ABSTAIN":
        return False, None

    raw_evidence = cached.get("evidence")
    evidence: EvidenceBreakdown | None = None
    if isinstance(raw_evidence, dict):
        evidence = EvidenceBreakdown(**{**raw_evidence, "threshold_used": float(threshold)})
    return True, evidence


@dataclass
class _GatewayRuntime:
    """Lazy runtime container so importing main or calling create_app performs zero disk I/O."""

    cfg: GatewaySettings
    store: TelemetryStore | None = None
    cache: SemanticCache | None = None
    extractor: FeatureExtractor | None = None
    calibrator: ConformalBundle | None = None
    router: RouterBundle | None = None
    provider_router: ProviderRouter | None = None
    initialized: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def ensure_ready(self) -> None:
        if self.initialized:
            return
        with self._lock:
            if self.initialized:
                return
            if self.store is None:
                self.store = TelemetryStore(self.cfg.telemetry_db_path)
            self.store.init_schema()
            if self.cache is None:
                self.cache = SemanticCache(
                    self.cfg.faiss_index_path,
                    store=self.store,
                    similarity_threshold=self.cfg.similarity_threshold,
                )
            if self.extractor is None:
                self.extractor = FeatureExtractor()
            if self.calibrator is None:
                self.calibrator = _load_pickle_artifact(self.cfg.artifacts_dir / "calibrator.pkl")
            if self.router is None:
                self.router = _load_pickle_artifact(self.cfg.artifacts_dir / "router.pkl")
            if self.provider_router is None:
                self.provider_router = ProviderRouter(settings=self.cfg)
            self.initialized = True


def create_app(
    settings: GatewaySettings | None = None,
    store: TelemetryStore | None = None,
    cache: SemanticCache | None = None,
    extractor: FeatureExtractor | None = None,
    calibrator: ConformalBundle | None = None,
    router: RouterBundle | None = None,
    provider_router: ProviderRouter | None = None,
) -> FastAPI:
    """Create the FastAPI gateway application with lazy lifespan initialization and DI."""
    cfg = settings or GatewaySettings()
    runtime = _GatewayRuntime(
        cfg=cfg,
        store=store,
        cache=cache,
        extractor=extractor,
        calibrator=calibrator,
        router=router,
        provider_router=provider_router,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(runtime.ensure_ready)
        yield

    app = FastAPI(
        title="Conformal LLM Router Gateway",
        version="0.1.0",
        description="Conformal hallucination control and Pareto-optimal LLM routing gateway.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(UpstreamUnavailableError)
    async def _handle_upstream_unavailable(
        _request: Request, exc: UpstreamUnavailableError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    async def _require_ready() -> _GatewayRuntime:
        if not runtime.initialized:
            await run_in_threadpool(runtime.ensure_ready)
        return runtime

    @app.post("/v1/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest) -> QueryResponse:
        rt = await _require_ready()
        assert rt.store is not None
        assert rt.cache is not None
        assert rt.extractor is not None
        assert rt.provider_router is not None
        if rt.calibrator is None or rt.router is None:
            raise HTTPException(
                status_code=503,
                detail="Gateway model artifacts (calibrator.pkl / router.pkl) are not loaded.",
            )

        t0 = time.perf_counter()
        context_text = _normalize_optional_text(req.context)
        alpha_used = float(
            req.alpha_override if req.alpha_override is not None else cfg.gateway_alpha_default
        )
        lambda_cost_used = float(
            req.lambda_cost_override
            if req.lambda_cost_override is not None
            else cfg.gateway_lambda_default
        )

        q_embs, q_meta_df = await run_in_threadpool(
            rt.extractor.extract_query_features, [req.query]
        )
        query_emb = q_embs[0]
        prob_tier1 = float(predict_tier1_probability(rt.router, q_embs, q_meta_df)[0])
        predicted_tier = select_tier(prob_tier1, lambda_cost_used)

        cached_record = None
        if not req.bypass_cache:
            cached_record = await run_in_threadpool(rt.cache.lookup, query_emb)

        is_hit, hit_evidence = _check_valid_cache_hit(
            cached_record,
            context_text=context_text,
            predicted_tier=predicted_tier,
            lambda_cost_override=req.lambda_cost_override,
            alpha_used=alpha_used,
            calibrator=rt.calibrator,
        )

        request_id = str(uuid4())
        if is_hit and cached_record is not None:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            record = {
                "request_id": request_id,
                "query_text": req.query,
                "context_text": context_text,
                "answer_text": cached_record["answer_text"],
                "draft_answer": cached_record.get("draft_answer"),
                "model_used": cached_record["model_used"],
                "tier": cached_record["tier"],
                "provider": cached_record["provider"],
                "escalated_from_tier1": False,
                "cache_hit": True,
                "fallback_used": False,
                "estimated_cost_usd": 0.0,
                "frontier_cost_usd": float(cached_record.get("frontier_cost_usd", 0.0)),
                "alpha_used": alpha_used,
                "lambda_cost_used": lambda_cost_used,
                "calibrated_risk_score": cached_record.get("calibrated_risk_score"),
                "conformal_decision": cached_record["conformal_decision"],
                "evidence": hit_evidence.model_dump() if hit_evidence is not None else None,
                "latency_ms": latency_ms,
            }
            await run_in_threadpool(rt.store.insert_request, record)
            return QueryResponse(
                request_id=request_id,
                answer=record["answer_text"],
                draft_answer=record["draft_answer"],
                model_used=record["model_used"],
                tier=record["tier"],
                provider=record["provider"],
                escalated_from_tier1=False,
                cache_hit=True,
                fallback_used=False,
                estimated_cost_usd=0.0,
                alpha_used=alpha_used,
                lambda_cost_used=lambda_cost_used,
                calibrated_risk_score=record["calibrated_risk_score"],
                conformal_decision=record["conformal_decision"],
                evidence=hit_evidence,
                latency_ms=latency_ms,
            )

        gen_res = await rt.provider_router.generate(predicted_tier, req.query, context_text)
        answer = gen_res.answer
        draft_answer: str | None = None
        model_used = gen_res.model_used
        tier_used: Literal["tier1", "tier2"] = gen_res.tier
        provider_used = gen_res.provider
        escalated_from_tier1 = False
        fallback_used = gen_res.fallback_used
        estimated_cost_usd = gen_res.estimated_cost_usd
        frontier_cost_usd = gen_res.frontier_cost_usd

        calibrated_risk_score: float | None = None
        conformal_decision: Literal["PASS", "ABSTAIN", "UNSCORED"] = "UNSCORED"
        evidence: EvidenceBreakdown | None = None

        if context_text is not None:
            calibrated_risk_score, conformal_decision, evidence = await run_in_threadpool(
                _score_candidate,
                rt.extractor,
                rt.calibrator,
                req.query,
                context_text,
                answer,
                alpha_used,
            )

            if conformal_decision == "ABSTAIN" and predicted_tier == "tier1":
                res_tier2 = await rt.provider_router.generate("tier2", req.query, context_text)
                escalated_from_tier1 = True
                tier_used = "tier2"
                model_used = res_tier2.model_used
                provider_used = res_tier2.provider
                fallback_used = fallback_used or res_tier2.fallback_used
                estimated_cost_usd += res_tier2.estimated_cost_usd
                frontier_cost_usd = res_tier2.frontier_cost_usd

                calibrated_risk_score, conformal_decision, evidence = await run_in_threadpool(
                    _score_candidate,
                    rt.extractor,
                    rt.calibrator,
                    req.query,
                    context_text,
                    res_tier2.answer,
                    alpha_used,
                )
                if conformal_decision == "PASS":
                    answer = res_tier2.answer
                else:
                    draft_answer = res_tier2.answer
                    answer = _ABSTAIN_FALLBACK_TEXT
            elif conformal_decision == "ABSTAIN":
                draft_answer = answer
                answer = _ABSTAIN_FALLBACK_TEXT

        latency_ms = (time.perf_counter() - t0) * 1000.0
        record = {
            "request_id": request_id,
            "query_text": req.query,
            "context_text": context_text,
            "answer_text": answer,
            "draft_answer": draft_answer,
            "model_used": model_used,
            "tier": tier_used,
            "provider": provider_used,
            "escalated_from_tier1": escalated_from_tier1,
            "cache_hit": False,
            "fallback_used": fallback_used,
            "estimated_cost_usd": estimated_cost_usd,
            "frontier_cost_usd": frontier_cost_usd,
            "alpha_used": alpha_used,
            "lambda_cost_used": lambda_cost_used,
            "calibrated_risk_score": calibrated_risk_score,
            "conformal_decision": conformal_decision,
            "evidence": evidence.model_dump() if evidence is not None else None,
            "latency_ms": latency_ms,
        }
        await run_in_threadpool(rt.store.insert_request, record)
        if conformal_decision != "ABSTAIN":
            await run_in_threadpool(rt.cache.put, query_emb, record)

        return QueryResponse(
            request_id=request_id,
            answer=answer,
            draft_answer=draft_answer,
            model_used=model_used,
            tier=tier_used,
            provider=provider_used,
            escalated_from_tier1=escalated_from_tier1,
            cache_hit=False,
            fallback_used=fallback_used,
            estimated_cost_usd=estimated_cost_usd,
            alpha_used=alpha_used,
            lambda_cost_used=lambda_cost_used,
            calibrated_risk_score=calibrated_risk_score,
            conformal_decision=conformal_decision,
            evidence=evidence,
            latency_ms=latency_ms,
        )

    @app.post("/v1/evaluate-risk", response_model=EvaluateRiskResponse)
    async def evaluate_risk_endpoint(req: EvaluateRiskRequest) -> EvaluateRiskResponse:
        rt = await _require_ready()
        assert rt.extractor is not None
        if rt.calibrator is None:
            raise HTTPException(
                status_code=503,
                detail="Conformal calibrator artifact (calibrator.pkl) is not loaded.",
            )
        alpha_used = float(
            req.alpha_override if req.alpha_override is not None else cfg.gateway_alpha_default
        )
        risk_score, decision, evidence = await run_in_threadpool(
            _score_candidate,
            rt.extractor,
            rt.calibrator,
            req.query,
            req.context,
            req.candidate_answer,
            alpha_used,
        )
        assert evidence.threshold_used is not None
        return EvaluateRiskResponse(
            calibrated_risk_score=risk_score,
            threshold_used=evidence.threshold_used,
            alpha_used=alpha_used,
            conformal_decision=decision,
            evidence=evidence,
        )

    @app.get("/v1/telemetry/pareto", response_model=ParetoTelemetryResponse)
    async def pareto_telemetry_endpoint(
        alpha: float | None = Query(default=None, ge=0.01, le=0.30),
        lambda_cost: float | None = Query(default=None, ge=0.0, le=1.0),
    ) -> ParetoTelemetryResponse:
        rt = await _require_ready()
        assert rt.store is not None
        alpha_val = float(alpha if alpha is not None else cfg.gateway_alpha_default)
        lambda_val = float(
            lambda_cost if lambda_cost is not None else cfg.gateway_lambda_default
        )
        summary = await run_in_threadpool(
            rt.store.aggregate_pareto, alpha_val, lambda_val
        )
        summary["offline_curve"] = _load_offline_pareto_curve(
            cfg.artifacts_dir / "pareto_sweep.json"
        )
        return ParetoTelemetryResponse(**summary)

    @app.get("/v1/telemetry/audit", response_model=AuditListResponse)
    async def audit_telemetry_endpoint(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        decision: Literal["PASS", "ABSTAIN", "UNSCORED"] | None = None,
        tier: Literal["tier1", "tier2"] | None = None,
        cache_hit: bool | None = None,
        q: str | None = None,
    ) -> AuditListResponse:
        rt = await _require_ready()
        assert rt.store is not None
        items, total = await run_in_threadpool(
            rt.store.query_audit,
            limit=limit,
            offset=offset,
            decision=decision,
            tier=tier,
            cache_hit=cache_hit,
            q=q,
        )
        return AuditListResponse(
            items=[AuditRecord(**item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    @app.get("/v1/benchmarks/summary", response_model=BenchmarksSummaryResponse)
    async def benchmarks_summary_endpoint() -> BenchmarksSummaryResponse:
        return BenchmarksSummaryResponse(
            judge_audit=_load_json_artifact(cfg.artifacts_dir / "judge_audit_summary.json"),
            conformal=_load_json_artifact(cfg.artifacts_dir / "conformal_sweep.json"),
            risk_metrics=_load_json_artifact(cfg.artifacts_dir / "risk_metrics.json"),
            pareto=_load_json_artifact(cfg.artifacts_dir / "pareto_sweep.json"),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_endpoint() -> HealthResponse:
        rt = await _require_ready()
        assert rt.store is not None
        assert rt.cache is not None
        assert rt.extractor is not None
        assert rt.provider_router is not None

        sqlite_ok = False
        try:
            with rt.store._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            sqlite_ok = True
        except (sqlite3.Error, OSError):
            sqlite_ok = False

        faiss_ok = getattr(rt.cache, "_index", None) is not None
        artifacts = {
            "calibrator": rt.calibrator is not None,
            "router": rt.router is not None,
        }
        models = {
            "embedder_loaded": getattr(rt.extractor, "_embedder", None) is not None,
            "nli_loaded": getattr(rt.extractor, "_nli_model", None) is not None,
        }
        storage = {"sqlite": sqlite_ok, "faiss": faiss_ok}
        providers = rt.provider_router.available_providers()
        is_ok = artifacts["calibrator"] and artifacts["router"] and sqlite_ok and faiss_ok

        return HealthResponse(
            status="ok" if is_ok else "degraded",
            artifacts=artifacts,
            models=models,
            storage=storage,
            providers=providers,
        )

    return app


app = create_app()
