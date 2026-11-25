"""Pareto frontier evaluation, OOD robustness sweep, and router artifact export."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.eval.features import QUERY_META_COLUMNS
from src.router.trainer import predict_tier1_probability, train_pareto_router

_REQUIRED_EVAL_COLS = ("tier1_quality", "tier2_quality", "tier1_cost", "tier2_cost")


def _validate_eval_inputs(
    probs_tier1: np.ndarray,
    df_eval: pd.DataFrame,
    lambdas: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    if df_eval.empty:
        raise ValueError("Evaluation DataFrame must not be empty.")
    missing = [col for col in _REQUIRED_EVAL_COLS if col not in df_eval.columns]
    if missing:
        raise ValueError(f"Missing required evaluation columns: {missing}")
    if not lambdas:
        raise ValueError("lambdas must not be empty.")

    probs = np.asarray(probs_tier1, dtype=float)
    if probs.ndim != 1 or len(probs) != len(df_eval):
        raise ValueError(
            f"Expected 1D probs_tier1 of length {len(df_eval)}, got shape {probs.shape}."
        )
    if not np.all(np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probs_tier1 must contain finite values in [0.0, 1.0].")

    clean_lambdas: list[float] = []
    for raw_lam in lambdas:
        lam = float(raw_lam)
        if not math.isfinite(lam) or lam < 0.0 or lam > 1.0:
            raise ValueError(f"Each lambda must be finite and in [0.0, 1.0], got {raw_lam}.")
        clean_lambdas.append(lam)

    arrays: list[np.ndarray] = []
    for col in _REQUIRED_EVAL_COLS:
        arr = df_eval[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Column '{col}' contains non-finite values.")
        arrays.append(arr)

    return probs, arrays[0], arrays[1], arrays[2], arrays[3], clean_lambdas


def _evaluate_routing_mask(
    route_tier1: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    frontier_cost: float,
) -> dict[str, float]:
    avg_quality = float(np.mean(np.where(route_tier1, q1, q2)))
    avg_cost = float(np.mean(np.where(route_tier1, c1, c2)))
    tier1_rate = float(np.mean(route_tier1))
    if frontier_cost > 0.0:
        normalized_cost = float(np.clip(avg_cost / frontier_cost, 0.0, 1.0))
        cost_savings = float((frontier_cost - avg_cost) / frontier_cost)
    else:
        normalized_cost = 0.0
        cost_savings = 0.0
    return {
        "avg_quality": avg_quality,
        "avg_cost": avg_cost,
        "normalized_cost": normalized_cost,
        "cost_savings": cost_savings,
        "tier1_rate": tier1_rate,
    }


def compute_pareto_frontier(
    probs_tier1: np.ndarray,
    df_eval: pd.DataFrame,
    lambdas: list[float],
    default_lambda: float = 0.45,
    random_state: int = 42,
) -> dict[str, Any]:
    """Sweep cost-weight lambdas over predicted Tier-1 probabilities and summarize Pareto metrics."""
    probs, q1, q2, c1, c2, clean_lambdas = _validate_eval_inputs(probs_tier1, df_eval, lambdas)
    default_lam = float(default_lambda)
    if not math.isfinite(default_lam) or default_lam < 0.0 or default_lam > 1.0:
        raise ValueError(f"default_lambda must be finite and in [0.0, 1.0], got {default_lambda}.")

    frontier_cost = float(np.mean(c2))
    frontier_quality = float(np.mean(q2))
    n_rows = len(df_eval)

    curve: list[dict[str, float]] = []
    for lam in clean_lambdas:
        route_tier1 = probs >= (1.0 - lam)
        metrics = _evaluate_routing_mask(route_tier1, q1, q2, c1, c2, frontier_cost)
        curve.append({"lambda_cost": lam, **metrics})

    if frontier_cost > 0.0:
        xs = np.array([pt["normalized_cost"] for pt in curve], dtype=float)
        ys = np.clip(np.array([pt["avg_quality"] for pt in curve], dtype=float), 0.0, 1.0)
        order = np.lexsort((ys, xs))
        auc_qc = float(np.clip(np.trapezoid(ys[order], xs[order]), 0.0, 1.0))
    else:
        auc_qc = 0.0

    qualifying = [pt for pt in curve if pt["avg_quality"] >= 0.95 * frontier_quality]
    cost_savings_at_95 = max(pt["cost_savings"] for pt in qualifying) if qualifying else None

    rng = np.random.default_rng(random_state)
    random_mask = rng.random(n_rows) < 0.5
    learned_mask = probs >= (1.0 - default_lam)

    baselines = {
        "always_frontier": _evaluate_routing_mask(
            np.zeros(n_rows, dtype=bool), q1, q2, c1, c2, frontier_cost
        ),
        "always_small": _evaluate_routing_mask(
            np.ones(n_rows, dtype=bool), q1, q2, c1, c2, frontier_cost
        ),
        "random_router": _evaluate_routing_mask(random_mask, q1, q2, c1, c2, frontier_cost),
        "learned_pareto_router": {
            "lambda_cost": default_lam,
            **_evaluate_routing_mask(learned_mask, q1, q2, c1, c2, frontier_cost),
        },
    }

    return {
        "curve": curve,
        "baselines": baselines,
        "auc_qc": auc_qc,
        "cost_savings_at_95pct_quality": cost_savings_at_95,
    }


def _extract_embeddings(df_split: pd.DataFrame) -> np.ndarray:
    if df_split.empty:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(df_split["query_embedding"].to_numpy()).astype(np.float32)


def run_router_pipeline(
    features_parquet: Path,
    config_path: Path,
    artifacts_dir: Path,
) -> tuple[Path, Path]:
    """Train the query-only Pareto router, evaluate in-domain/OOD sweeps, and export artifacts."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    random_state = int(cfg.get("random_state", 42))
    default_lambda = float(cfg.get("default_lambda_cost", 0.45))
    lambda_min = float(cfg.get("lambda_min", 0.0))
    lambda_max = float(cfg.get("lambda_max", 1.0))
    lambda_step = float(cfg.get("lambda_step", 0.05))
    n_pca = int(cfg.get("n_pca", 32))

    lambdas = [
        round(float(v), 4)
        for v in np.arange(lambda_min, lambda_max + lambda_step * 0.5, lambda_step)
        if v <= lambda_max + 1e-9
    ]

    df = pd.read_parquet(features_parquet)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    df_ood = df[df["split"] == "ood_test"].reset_index(drop=True)
    if df_train.empty or df_test.empty:
        raise ValueError("Features parquet must contain non-empty 'train' and 'test' splits.")

    emb_train = _extract_embeddings(df_train)
    emb_test = _extract_embeddings(df_test)

    bundle = train_pareto_router(
        df_train,
        emb_train,
        QUERY_META_COLUMNS,
        n_pca=n_pca,
        random_state=random_state,
    )
    bundle.default_lambda = default_lambda

    probs_test = predict_tier1_probability(bundle, emb_test, df_test)
    in_domain = compute_pareto_frontier(
        probs_test,
        df_test,
        lambdas,
        default_lambda=default_lambda,
        random_state=random_state,
    )

    if df_ood.empty:
        ood_available = False
        ood_curve: list[dict[str, float]] = []
        ood_baselines: dict[str, Any] = {}
        ood_auc_qc: float | None = None
        ood_cost_savings_at_95: float | None = None
        ood_tasks: list[str] = []
    else:
        emb_ood = _extract_embeddings(df_ood)
        probs_ood = predict_tier1_probability(bundle, emb_ood, df_ood)
        ood_res = compute_pareto_frontier(
            probs_ood,
            df_ood,
            lambdas,
            default_lambda=default_lambda,
            random_state=random_state,
        )
        ood_available = True
        ood_curve = ood_res["curve"]
        ood_baselines = ood_res["baselines"]
        ood_auc_qc = ood_res["auc_qc"]
        ood_cost_savings_at_95 = ood_res["cost_savings_at_95pct_quality"]
        ood_tasks = (
            sorted(str(t) for t in df_ood["eval_name"].unique())
            if "eval_name" in df_ood.columns
            else []
        )

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    router_path = artifacts_path / "router.pkl"
    with router_path.open("wb") as fh:
        pickle.dump(bundle, fh)

    sweep_path = artifacts_path / "pareto_sweep.json"
    sweep_payload = {
        "status": "ready",
        "n_train": len(df_train),
        "n_test": len(df_test),
        "n_ood": len(df_ood),
        "default_lambda": default_lambda,
        "lambdas": lambdas,
        "curve": in_domain["curve"],
        "baselines": in_domain["baselines"],
        "auc_qc": in_domain["auc_qc"],
        "cost_savings_at_95pct_quality": in_domain["cost_savings_at_95pct_quality"],
        "ood_available": ood_available,
        "ood_tasks": ood_tasks,
        "ood_curve": ood_curve,
        "ood_baselines": ood_baselines,
        "ood_auc_qc": ood_auc_qc,
        "ood_cost_savings_at_95pct_quality": ood_cost_savings_at_95,
    }
    sweep_path.write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")

    return router_path, sweep_path


if __name__ == "__main__":
    from src.router.pareto import run_router_pipeline as _cli_run_router_pipeline

    router_pkl, pareto_json = _cli_run_router_pipeline(
        Path("data/processed/routerbench_features.parquet"),
        Path("config/routerbench.yaml"),
        Path("artifacts"),
    )
    print(f"Exported: {router_pkl}, {pareto_json}")
