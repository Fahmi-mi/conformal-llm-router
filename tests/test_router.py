"""Tests for the query-only tier-mapped Pareto router trainer."""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.eval.features import QUERY_META_COLUMNS
from src.router.pareto import compute_pareto_frontier, run_router_pipeline
from src.router.trainer import (
    RouterBundle,
    predict_tier1_probability,
    select_tier,
    train_pareto_router,
)


def _make_synthetic_data(
    n_samples: int = 60,
    emb_dim: int = 384,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    embs = rng.normal(size=(n_samples, emb_dim)).astype(np.float32)
    meta = pd.DataFrame(
        {
            "char_len": rng.integers(10, 200, size=n_samples),
            "token_count": rng.integers(3, 50, size=n_samples),
        }
    )
    meta["tier1_sufficient"] = (meta["token_count"] < 25).astype(bool)
    return meta, embs


def test_router_training_and_boundary_tier_selection() -> None:
    meta, embs = _make_synthetic_data(n_samples=60, emb_dim=384, seed=42)
    bundle = train_pareto_router(meta, embs, ["char_len", "token_count"], n_pca=16, random_state=42)

    assert isinstance(bundle, RouterBundle)
    assert RouterBundle.__module__ == "src.router.trainer"
    assert bundle.embedding_dim == 384
    assert bundle.query_meta_cols == ["char_len", "token_count"]

    probs = predict_tier1_probability(bundle, embs[:5], meta.iloc[:5])
    assert probs.shape == (5,) and np.all((probs >= 0.0) & (probs <= 1.0))
    assert select_tier(0.5, lambda_cost=0.0) == "tier2"
    assert select_tier(0.5, lambda_cost=1.0) == "tier1"


def test_single_row_vs_batch_parity_and_no_feature_name_warnings() -> None:
    meta, embs = _make_synthetic_data(n_samples=50, emb_dim=32, seed=7)
    bundle = train_pareto_router(meta, embs, ["char_len", "token_count"], n_pca=8, random_state=7)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        batch_probs = predict_tier1_probability(bundle, embs[:5], meta.iloc[:5])
        single_2d_probs = np.array(
            [predict_tier1_probability(bundle, embs[i : i + 1], meta.iloc[i : i + 1])[0] for i in range(5)]
        )
        single_1d_probs = np.array(
            [predict_tier1_probability(bundle, embs[i], meta.iloc[i : i + 1])[0] for i in range(5)]
        )

    feature_name_warnings = [
        w for w in recorded if "valid feature names" in str(w.message).lower()
    ]
    assert not feature_name_warnings
    assert np.allclose(single_2d_probs, batch_probs)
    assert np.allclose(single_1d_probs, batch_probs)


def test_select_tier_thresholds_and_validation() -> None:
    assert select_tier(0.99, lambda_cost=0.0) == "tier2"
    assert select_tier(1.0, lambda_cost=0.0) == "tier1"
    assert select_tier(0.0, lambda_cost=1.0) == "tier1"
    assert select_tier(0.55, lambda_cost=0.45) == "tier1"
    assert select_tier(0.54, lambda_cost=0.45) == "tier2"

    for bad_lambda in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="lambda_cost"):
            select_tier(0.5, lambda_cost=bad_lambda)


def test_pca_clamping_and_single_class_edge_cases() -> None:
    rng = np.random.default_rng(123)
    embs = rng.normal(size=(10, 8)).astype(np.float32)

    df_all_true = pd.DataFrame(
        {
            "char_len": np.arange(10, 20),
            "token_count": np.arange(1, 11),
            "tier1_sufficient": np.ones(10, dtype=bool),
        }
    )
    bundle_true = train_pareto_router(df_all_true, embs, ["char_len", "token_count"], n_pca=32)
    assert bundle_true.pca is not None
    assert bundle_true.pca.n_components_ == 8
    probs_true = predict_tier1_probability(bundle_true, embs[:3], df_all_true.iloc[:3])
    assert np.allclose(probs_true, 1.0)

    df_all_false = df_all_true.copy()
    df_all_false["tier1_sufficient"] = False
    bundle_false = train_pareto_router(df_all_false, embs, ["char_len", "token_count"], n_pca=32)
    probs_false = predict_tier1_probability(bundle_false, embs[:3], df_all_false.iloc[:3])
    assert np.allclose(probs_false, 0.0)


def test_validation_errors_pickle_and_full_query_meta_columns() -> None:
    rng = np.random.default_rng(99)
    n = 40
    embs = rng.normal(size=(n, 384)).astype(np.float32)
    df = pd.DataFrame(
        {
            "char_len": rng.integers(10, 300, size=n),
            "token_count": rng.integers(2, 60, size=n),
            "sentence_count": rng.integers(1, 5, size=n),
            "question_mark_count": rng.integers(0, 3, size=n),
            "digit_ratio": rng.uniform(0.0, 0.2, size=n),
            "uppercase_ratio": rng.uniform(0.0, 0.3, size=n),
            "avg_word_len": rng.uniform(3.0, 8.0, size=n),
        }
    )
    df["tier1_sufficient"] = (df["token_count"] < 30).astype(bool)

    bundle = train_pareto_router(df, embs, QUERY_META_COLUMNS, n_pca=16, random_state=99)
    restored = pickle.loads(pickle.dumps(bundle))
    assert np.allclose(
        predict_tier1_probability(bundle, embs[:4], df.iloc[:4]),
        predict_tier1_probability(restored, embs[:4], df.iloc[:4]),
    )

    with pytest.raises(ValueError, match="embedding"):
        predict_tier1_probability(bundle, np.zeros((4, 128), dtype=np.float32), df.iloc[:4])

    with pytest.raises(ValueError, match="length|rows|mismatch"):
        predict_tier1_probability(bundle, embs[:3], df.iloc[:4])

    with pytest.raises(ValueError, match="missing|columns"):
        predict_tier1_probability(bundle, embs[:4], df.iloc[:4].drop(columns=["char_len"]))


def test_compute_pareto_frontier_endpoints_auc_and_baselines() -> None:
    df_eval = pd.DataFrame(
        {
            "tier1_quality": [0.95, 0.90, 0.20, 0.15],
            "tier2_quality": [0.95, 0.95, 0.90, 0.95],
            "tier1_cost": [0.0001, 0.0001, 0.0001, 0.0001],
            "tier2_cost": [0.0100, 0.0100, 0.0100, 0.0100],
        }
    )
    probs = np.array([0.9, 0.7, 0.3, 0.1], dtype=float)
    lambdas = [0.0, 0.2, 0.45, 0.8, 1.0]

    result = compute_pareto_frontier(probs, df_eval, lambdas)
    curve = result["curve"]
    assert len(curve) == len(lambdas)

    by_lam = {pt["lambda_cost"]: pt for pt in curve}
    baselines = result["baselines"]
    assert set(baselines.keys()) == {
        "always_frontier",
        "always_small",
        "random_router",
        "learned_pareto_router",
    }

    assert np.isclose(by_lam[0.0]["avg_cost"], baselines["always_frontier"]["avg_cost"])
    assert np.isclose(by_lam[0.0]["avg_quality"], baselines["always_frontier"]["avg_quality"])
    assert np.isclose(by_lam[1.0]["avg_cost"], baselines["always_small"]["avg_cost"])
    assert np.isclose(by_lam[1.0]["avg_quality"], baselines["always_small"]["avg_quality"])

    tier1_rates = [pt["tier1_rate"] for pt in curve]
    avg_costs = [pt["avg_cost"] for pt in curve]
    assert tier1_rates == sorted(tier1_rates)
    assert avg_costs == sorted(avg_costs, reverse=True)

    assert 0.0 <= result["auc_qc"] <= 1.0
    assert result["cost_savings_at_95pct_quality"] is not None
    assert np.isclose(result["cost_savings_at_95pct_quality"], by_lam[0.45]["cost_savings"])
    assert np.isclose(baselines["learned_pareto_router"]["avg_cost"], by_lam[0.45]["avg_cost"])
    assert np.isclose(
        baselines["learned_pareto_router"]["avg_quality"], by_lam[0.45]["avg_quality"]
    )

    no_95 = compute_pareto_frontier(probs, df_eval, [1.0])
    assert no_95["cost_savings_at_95pct_quality"] is None


def test_compute_pareto_frontier_zero_tier2_cost_safe() -> None:
    df_zero = pd.DataFrame(
        {
            "tier1_quality": [0.8, 0.4],
            "tier2_quality": [0.9, 0.9],
            "tier1_cost": [0.0, 0.0],
            "tier2_cost": [0.0, 0.0],
        }
    )
    probs = np.array([0.85, 0.2], dtype=float)
    result = compute_pareto_frontier(probs, df_zero, [0.0, 0.5, 1.0])

    assert result["auc_qc"] == 0.0
    assert all(pt["normalized_cost"] == 0.0 for pt in result["curve"])
    assert all(pt["cost_savings"] == 0.0 for pt in result["curve"])
    assert result["baselines"]["always_frontier"]["normalized_cost"] == 0.0
    assert result["baselines"]["always_small"]["cost_savings"] == 0.0


def _write_synthetic_routerbench_parquet(path: Path, include_ood: bool = True) -> None:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    splits = [("train", 30), ("test", 12)]
    if include_ood:
        splits.append(("ood_test", 8))

    idx = 0
    for split_name, count in splits:
        for _ in range(count):
            token_count = int(rng.integers(5, 60))
            tier1_suff = bool(token_count < 30)
            t1_q = 0.9 if tier1_suff else 0.2
            t2_q = 0.92
            rows.append(
                {
                    "sample_id": f"s_{idx}",
                    "eval_name": "mbpp" if split_name == "ood_test" else "mmlu",
                    "split": split_name,
                    "tier1_quality": t1_q,
                    "tier2_quality": t2_q,
                    "tier1_cost": 0.0001,
                    "tier2_cost": 0.005,
                    "tier1_sufficient": tier1_suff,
                    "char_len": token_count * 4,
                    "token_count": token_count,
                    "sentence_count": 2,
                    "question_mark_count": 1,
                    "digit_ratio": 0.05,
                    "uppercase_ratio": 0.1,
                    "avg_word_len": 4.5,
                    "query_embedding": rng.normal(size=16).astype(np.float32).tolist(),
                }
            )
            idx += 1
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_run_router_pipeline_exports_artifacts_and_handles_ood(tmp_path: Path) -> None:
    parquet_with_ood = tmp_path / "routerbench_with_ood.parquet"
    _write_synthetic_routerbench_parquet(parquet_with_ood, include_ood=True)

    cfg_path = tmp_path / "routerbench.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "random_state": 42,
                "default_lambda_cost": 0.45,
                "lambda_min": 0.0,
                "lambda_max": 1.0,
                "lambda_step": 0.25,
                "n_pca": 8,
            }
        ),
        encoding="utf-8",
    )

    artifacts_dir = tmp_path / "artifacts"
    router_path, sweep_path = run_router_pipeline(parquet_with_ood, cfg_path, artifacts_dir)
    assert router_path.exists()
    assert sweep_path.exists()

    with router_path.open("rb") as fh:
        bundle = pickle.load(fh)
    assert isinstance(bundle, RouterBundle)
    assert bundle.default_lambda == 0.45

    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ready"
    assert payload["default_lambda"] == 0.45
    assert payload["ood_available"] is True
    assert len(payload["curve"]) == 5
    assert len(payload["ood_curve"]) == 5
    assert set(payload["baselines"].keys()) == {
        "always_frontier",
        "always_small",
        "random_router",
        "learned_pareto_router",
    }
    assert 0.0 <= payload["auc_qc"] <= 1.0

    parquet_no_ood = tmp_path / "routerbench_no_ood.parquet"
    _write_synthetic_routerbench_parquet(parquet_no_ood, include_ood=False)
    artifacts_no_ood = tmp_path / "artifacts_no_ood"
    _, sweep_no_ood_path = run_router_pipeline(parquet_no_ood, cfg_path, artifacts_no_ood)
    payload_no_ood = json.loads(sweep_no_ood_path.read_text(encoding="utf-8"))
    assert payload_no_ood["ood_available"] is False
    assert payload_no_ood["ood_curve"] == []
