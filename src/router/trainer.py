"""Query-only tier-mapped Pareto router training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier

_TARGET_COL = "tier1_sufficient"
_DEFAULT_LAMBDA_COST = 0.45


@dataclass
class RouterBundle:
    """Serialized artifacts and metadata for query-only Tier-1/Tier-2 routing."""

    estimator: Any
    query_meta_cols: list[str]
    pca: Any | None
    default_lambda: float = _DEFAULT_LAMBDA_COST
    embedding_dim: int = 384


def _validate_embeddings_and_meta(
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    query_meta_cols: list[str],
    expected_dim: int | None = None,
    allow_1d_single: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate embedding and metadata alignment and return finite float64 arrays."""
    missing = [col for col in query_meta_cols if col not in meta_df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    emb_arr = np.asarray(embeddings, dtype=float)
    if allow_1d_single and emb_arr.ndim == 1 and len(meta_df) == 1:
        emb_arr = emb_arr.reshape(1, -1)
    if emb_arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {emb_arr.shape}.")
    if emb_arr.shape[1] == 0:
        raise ValueError("Embedding array must have at least one feature dimension.")
    if expected_dim is not None and emb_arr.shape[1] != expected_dim:
        raise ValueError(
            f"Expected embedding dimension {expected_dim}, got {emb_arr.shape[1]}."
        )
    if emb_arr.shape[0] != len(meta_df):
        raise ValueError(
            f"Row count mismatch between embeddings ({emb_arr.shape[0]}) "
            f"and metadata ({len(meta_df)})."
        )
    if not np.all(np.isfinite(emb_arr)):
        raise ValueError("Input embeddings contain non-finite values (NaN or inf).")

    if query_meta_cols:
        meta_arr = meta_df[query_meta_cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(meta_arr)):
            raise ValueError("Input metadata features contain non-finite values (NaN or inf).")
    else:
        meta_arr = np.empty((len(meta_df), 0), dtype=float)

    return emb_arr, meta_arr


def _validate_binary_target(series: pd.Series) -> np.ndarray:
    """Validate and convert binary target labels to integer {0, 1}."""
    if series.isna().any():
        raise ValueError(f"Target '{_TARGET_COL}' contains missing values.")
    y_float = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(y_float)) or not np.all(np.isin(y_float, (0.0, 1.0))):
        raise ValueError(f"Target '{_TARGET_COL}' must contain only binary values {{0, 1}}.")
    return y_float.astype(int)


def train_pareto_router(
    df_train: pd.DataFrame,
    embeddings_train: np.ndarray,
    query_meta_cols: list[str],
    n_pca: int = 32,
    random_state: int = 42,
) -> RouterBundle:
    """Train a PCA + LightGBM query-only classifier predicting Tier-1 sufficiency."""
    if n_pca < 0:
        raise ValueError(f"n_pca must be non-negative, got {n_pca}.")
    if _TARGET_COL not in df_train.columns:
        raise ValueError(f"Missing required target column: '{_TARGET_COL}'")

    emb_arr, meta_arr = _validate_embeddings_and_meta(
        embeddings_train,
        df_train,
        query_meta_cols,
    )
    n_samples, n_features = emb_arr.shape
    if n_samples == 0:
        raise ValueError("Training data must not be empty.")
    if n_pca == 0 and not query_meta_cols:
        raise ValueError("At least one feature (n_pca > 0 or query_meta_cols) is required.")

    y = _validate_binary_target(df_train[_TARGET_COL])

    pca: PCA | None = None
    if n_pca > 0:
        n_components = min(n_pca, n_samples, n_features)
        pca = PCA(n_components=n_components, random_state=random_state)
        emb_features = pca.fit_transform(emb_arr)
    else:
        emb_features = np.empty((n_samples, 0), dtype=float)

    X_train = np.hstack([emb_features, meta_arr])

    unique_classes = np.unique(y)
    if unique_classes.size < 2:
        estimator: Any = DummyClassifier(strategy="constant", constant=int(unique_classes[0]))
        estimator.fit(X_train, y)
    else:
        estimator = LGBMClassifier(random_state=random_state, verbose=-1)
        estimator.fit(X_train, y)

    return RouterBundle(
        estimator=estimator,
        query_meta_cols=list(query_meta_cols),
        pca=pca,
        default_lambda=_DEFAULT_LAMBDA_COST,
        embedding_dim=int(n_features),
    )


def predict_tier1_probability(
    bundle: RouterBundle,
    query_embeddings: np.ndarray,
    query_meta_df: pd.DataFrame,
) -> np.ndarray:
    """Predict Tier-1 sufficiency probabilities from query embeddings and metadata."""
    emb_arr, meta_arr = _validate_embeddings_and_meta(
        query_embeddings,
        query_meta_df,
        bundle.query_meta_cols,
        expected_dim=bundle.embedding_dim,
        allow_1d_single=True,
    )
    if len(query_meta_df) == 0:
        return np.empty(0, dtype=float)

    if bundle.pca is not None:
        emb_features = bundle.pca.transform(emb_arr)
    else:
        emb_features = np.empty((len(query_meta_df), 0), dtype=float)

    X = np.hstack([emb_features, meta_arr])
    probs = np.asarray(bundle.estimator.predict_proba(X), dtype=float)

    classes = getattr(bundle.estimator, "classes_", None)
    if classes is not None:
        matches = np.where(np.asarray(classes) == 1)[0]
        if len(matches) == 0:
            return np.zeros(probs.shape[0], dtype=float)
        return np.asarray(probs[:, int(matches[0])], dtype=float)

    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Expected 2-column probability output, got shape {probs.shape}.")
    return np.asarray(probs[:, 1], dtype=float)


def select_tier(prob_tier1: float, lambda_cost: float) -> Literal["tier1", "tier2"]:
    """Map predicted Tier-1 sufficiency probability and cost weight to a routing tier."""
    lam = float(lambda_cost)
    if not math.isfinite(lam) or lam < 0.0 or lam > 1.0:
        raise ValueError(f"lambda_cost must be finite and in [0.0, 1.0], got {lambda_cost}.")

    prob = float(prob_tier1)
    if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
        raise ValueError(f"prob_tier1 must be finite and in [0.0, 1.0], got {prob_tier1}.")

    return "tier1" if prob >= (1.0 - lam) else "tier2"
