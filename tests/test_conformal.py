import warnings

import numpy as np
import pandas as pd
import pytest

from src.conformal.risk_model import (
    RISK_FEATURE_COLS,
    predict_risk_proba,
    train_risk_classifiers,
)
from src.eval.features import RAG_FEATURE_COLUMNS


def _make_sample_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in RISK_FEATURE_COLS}
    data["has_hallucination"] = (data["nli_contradiction"] > 0).astype(int)
    return pd.DataFrame(data)


def test_train_risk_classifiers_returns_valid_probabilities():
    df = _make_sample_df(n=120, seed=42)
    models = train_risk_classifiers(df, random_state=42)
    assert set(models.keys()) == {"logistic", "lightgbm"}
    X = df[RISK_FEATURE_COLS].to_numpy()
    for name in ("logistic", "lightgbm"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            probs = predict_risk_proba(models[name], X)
        assert not caught, f"Unexpected warnings for {name}: {[str(w.message) for w in caught]}"
        assert probs.shape == (120,)
        assert np.all(np.isfinite(probs))
        assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_train_risk_classifiers_rejects_missing_columns():
    df = _make_sample_df(n=40).drop(columns=["cosine_sim"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        train_risk_classifiers(df)


def test_train_risk_classifiers_rejects_missing_target():
    df = _make_sample_df(n=40).drop(columns=["has_hallucination"])
    with pytest.raises(ValueError, match="has_hallucination"):
        train_risk_classifiers(df)


def test_train_risk_classifiers_rejects_non_finite_features():
    df = _make_sample_df(n=40)
    df.loc[0, "cosine_sim"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        train_risk_classifiers(df)


def test_risk_feature_cols_matches_eval_features():
    assert RISK_FEATURE_COLS == RAG_FEATURE_COLUMNS


def test_predict_risk_proba_rejects_non_finite_and_invalid_shape():
    df = _make_sample_df(n=60)
    models = train_risk_classifiers(df, random_state=42)
    X_bad = df[RISK_FEATURE_COLS].to_numpy(copy=True)
    X_bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        predict_risk_proba(models["logistic"], X_bad)

    with pytest.raises(ValueError, match="2D"):
        predict_risk_proba(models["lightgbm"], np.zeros(len(RISK_FEATURE_COLS)))
