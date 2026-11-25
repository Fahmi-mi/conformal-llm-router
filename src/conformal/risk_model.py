"""Base hallucination risk classifiers trained on RAG evaluation features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RISK_FEATURE_COLS: list[str] = [
    "cosine_sim",
    "nli_entailment",
    "nli_contradiction",
    "nli_neutral",
    "entity_overlap",
    "length_ratio",
    "response_token_count",
    "context_token_count",
]

_TARGET_COL = "has_hallucination"


def _validate_feature_array(X: Any) -> np.ndarray:
    """Convert features to a 2D finite float array."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError("Input feature array must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input features contain non-finite values (NaN or inf).")
    return arr


def _validate_binary_target(series: pd.Series) -> np.ndarray:
    """Validate and return binary integer labels {0, 1} with both classes present."""
    y_float = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(y_float)):
        raise ValueError(f"Target '{_TARGET_COL}' contains non-finite values.")
    if not np.all(np.isin(y_float, (0.0, 1.0))):
        raise ValueError(f"Target '{_TARGET_COL}' must contain only binary values {{0, 1}}.")
    y = y_float.astype(int)
    if np.unique(y).size < 2:
        raise ValueError(f"Target '{_TARGET_COL}' must contain both classes 0 and 1.")
    return y


def train_risk_classifiers(
    df_train: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train standardized LogisticRegression and LightGBM hallucination risk classifiers."""
    missing = [col for col in RISK_FEATURE_COLS if col not in df_train.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if _TARGET_COL not in df_train.columns:
        raise ValueError(f"Missing required target column: '{_TARGET_COL}'")

    X = _validate_feature_array(np.asarray(df_train[RISK_FEATURE_COLS], dtype=float))
    y = _validate_binary_target(df_train[_TARGET_COL])

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    logistic.fit(X, y)

    lightgbm = LGBMClassifier(random_state=random_state, verbose=-1)
    lightgbm.fit(X, y)

    return {"logistic": logistic, "lightgbm": lightgbm}


def predict_risk_proba(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Predict positive-class hallucination probabilities from a 2D feature array."""
    arr = _validate_feature_array(X)
    probs = np.asarray(estimator.predict_proba(arr), dtype=float)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Expected 2-column probability output, got shape {probs.shape}.")

    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        matches = np.where(np.asarray(classes) == 1)[0]
        pos_idx = int(matches[0]) if len(matches) else 1
    else:
        pos_idx = 1

    return np.asarray(probs[:, pos_idx], dtype=float)
