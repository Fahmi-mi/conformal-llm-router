"""SQLite telemetry persistence, audit queries, and Pareto aggregation."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


def _serialize_evidence(record: dict[str, Any]) -> str | None:
    if "evidence_json" in record and record["evidence_json"] is not None:
        raw = record["evidence_json"]
        return raw if isinstance(raw, str) else json.dumps(raw)
    evidence = record.get("evidence")
    if evidence is None:
        return None
    if isinstance(evidence, BaseModel):
        return evidence.model_dump_json()
    return json.dumps(evidence)


def _compute_metrics(rows: list[sqlite3.Row]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "total_requests": 0,
            "cache_hit_rate": 0.0,
            "abstention_rate": 0.0,
            "avg_cost_usd": 0.0,
            "estimated_frontier_cost_usd": 0.0,
            "realized_cost_savings_pct": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "tier1_share": 0.0,
        }

    cache_hits = sum(int(r["cache_hit"]) for r in rows)
    abstains = sum(1 for r in rows if r["conformal_decision"] == "ABSTAIN")
    tier1_count = sum(1 for r in rows if r["tier"] == "tier1")
    avg_cost = sum(float(r["estimated_cost_usd"]) for r in rows) / total
    frontier_cost = sum(float(r["frontier_cost_usd"]) for r in rows) / total
    savings_pct = (
        ((frontier_cost - avg_cost) / frontier_cost) * 100.0 if frontier_cost > 0.0 else 0.0
    )
    latencies = np.array([float(r["latency_ms"]) for r in rows], dtype=float)
    return {
        "total_requests": total,
        "cache_hit_rate": cache_hits / total,
        "abstention_rate": abstains / total,
        "avg_cost_usd": avg_cost,
        "estimated_frontier_cost_usd": frontier_cost,
        "realized_cost_savings_pct": savings_pct,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "tier1_share": tier1_count / total,
    }


class TelemetryStore:
    """Thread-safe SQLite store for gateway request logs and FAISS cache mappings."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock, closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn

    def init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    context_text TEXT,
                    answer_text TEXT NOT NULL,
                    draft_answer TEXT,
                    model_used TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    escalated_from_tier1 INTEGER NOT NULL DEFAULT 0,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                    frontier_cost_usd REAL NOT NULL DEFAULT 0.0,
                    alpha_used REAL NOT NULL,
                    lambda_cost_used REAL NOT NULL,
                    calibrated_risk_score REAL,
                    conformal_decision TEXT NOT NULL,
                    evidence_json TEXT,
                    latency_ms REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    faiss_id INTEGER PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (request_id) REFERENCES requests(request_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_decision ON requests(conformal_decision)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_tier ON requests(tier)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_cache_hit ON requests(cache_hit)"
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        evidence_raw = row["evidence_json"]
        evidence = json.loads(evidence_raw) if evidence_raw else None
        return {
            "request_id": row["request_id"],
            "created_at": row["created_at"],
            "query_text": row["query_text"],
            "context_text": row["context_text"],
            "answer_text": row["answer_text"],
            "draft_answer": row["draft_answer"],
            "model_used": row["model_used"],
            "tier": row["tier"],
            "provider": row["provider"],
            "escalated_from_tier1": bool(row["escalated_from_tier1"]),
            "cache_hit": bool(row["cache_hit"]),
            "fallback_used": bool(row["fallback_used"]),
            "estimated_cost_usd": float(row["estimated_cost_usd"]),
            "frontier_cost_usd": float(row["frontier_cost_usd"]),
            "alpha_used": float(row["alpha_used"]),
            "lambda_cost_used": float(row["lambda_cost_used"]),
            "calibrated_risk_score": (
                float(row["calibrated_risk_score"])
                if row["calibrated_risk_score"] is not None
                else None
            ),
            "conformal_decision": row["conformal_decision"],
            "evidence": evidence,
            "evidence_json": evidence_raw,
            "latency_ms": float(row["latency_ms"]),
        }

    def insert_request(self, record: dict[str, Any]) -> None:
        created_at = record.get("created_at") or datetime.now(UTC).isoformat()
        evidence_json = _serialize_evidence(record)
        params = (
            record["request_id"],
            created_at,
            record["query_text"],
            record.get("context_text"),
            record["answer_text"],
            record.get("draft_answer"),
            record["model_used"],
            record["tier"],
            record["provider"],
            int(bool(record.get("escalated_from_tier1", False))),
            int(bool(record.get("cache_hit", False))),
            int(bool(record.get("fallback_used", False))),
            float(record.get("estimated_cost_usd", 0.0)),
            float(record.get("frontier_cost_usd", 0.0)),
            float(record["alpha_used"]),
            float(record["lambda_cost_used"]),
            (
                float(record["calibrated_risk_score"])
                if record.get("calibrated_risk_score") is not None
                else None
            ),
            record["conformal_decision"],
            evidence_json,
            float(record["latency_ms"]),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO requests (
                    request_id, created_at, query_text, context_text, answer_text,
                    draft_answer, model_used, tier, provider, escalated_from_tier1,
                    cache_hit, fallback_used, estimated_cost_usd, frontier_cost_usd,
                    alpha_used, lambda_cost_used, calibrated_risk_score,
                    conformal_decision, evidence_json, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def query_audit(
        self,
        limit: int = 50,
        offset: int = 0,
        decision: str | None = None,
        tier: str | None = None,
        cache_hit: bool | None = None,
        q: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        params: list[Any] = []

        if decision is not None:
            clauses.append("conformal_decision = ?")
            params.append(decision)
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        if cache_hit is not None:
            clauses.append("cache_hit = ?")
            params.append(int(cache_hit))
        if q is not None and q.strip():
            escaped_q = (
                q.strip()
                .lower()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            clauses.append("LOWER(query_text) LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped_q}%")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connect() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM requests {where_sql}",
                    params,
                ).fetchone()[0]
            )
            rows = conn.execute(
                f"SELECT * FROM requests {where_sql} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                [*params, max(0, int(limit)), max(0, int(offset))],
            ).fetchall()

        return [self._row_to_dict(row) for row in rows], total

    def aggregate_pareto(self, alpha: float, lambda_cost: float) -> dict[str, Any]:
        target_alpha = float(alpha)
        target_lambda = float(lambda_cost)
        with self._connect() as conn:
            all_rows = conn.execute(
                """
                SELECT
                    cache_hit,
                    conformal_decision,
                    tier,
                    estimated_cost_usd,
                    frontier_cost_usd,
                    latency_ms,
                    alpha_used,
                    lambda_cost_used
                FROM requests
                """
            ).fetchall()

        filtered_rows = [
            row
            for row in all_rows
            if abs(float(row["alpha_used"]) - target_alpha) < 1e-6
            and abs(float(row["lambda_cost_used"]) - target_lambda) < 1e-6
        ]
        summary = _compute_metrics(filtered_rows)
        grouped: dict[tuple[float, float], list[sqlite3.Row]] = {}
        for row in all_rows:
            key = (round(float(row["alpha_used"]), 6), round(float(row["lambda_cost_used"]), 6))
            grouped.setdefault(key, []).append(row)

        live_points: list[dict[str, Any]] = []
        for (grp_alpha, grp_lambda), grp_rows in sorted(
            grouped.items(), key=lambda item: (item[0][1], item[0][0])
        ):
            metrics = _compute_metrics(grp_rows)
            live_points.append(
                {
                    "alpha_used": grp_alpha,
                    "lambda_cost_used": grp_lambda,
                    **metrics,
                }
            )

        return {
            **summary,
            "alpha_filter": target_alpha,
            "lambda_cost_filter": target_lambda,
            "live_points": live_points,
            "offline_curve": [],
        }

    def register_cache_entry(self, faiss_id: int, request_id: str) -> None:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries (faiss_id, request_id, created_at)
                VALUES (?, ?, ?)
                """,
                (int(faiss_id), request_id, created_at),
            )

    def get_cached_request_by_faiss_id(self, faiss_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT r.*
                FROM cache_entries c
                JOIN requests r ON c.request_id = r.request_id
                WHERE c.faiss_id = ?
                """,
                (int(faiss_id),),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None
