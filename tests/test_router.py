"""Tests for the query-only tier-mapped Pareto router trainer."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from src.eval.features import QUERY_META_COLUMNS
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
