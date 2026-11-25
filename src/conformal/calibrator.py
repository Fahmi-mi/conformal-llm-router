"""MAPIE split-conformal calibration and evaluation for hallucination risk gating."""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from mapie.classification import SplitConformalClassifier
from sklearn.metrics import brier_score_loss

from src.conformal.risk_model import (
    RISK_FEATURE_COLS,
    _validate_feature_array,
    predict_risk_proba,
    train_risk_classifiers,
)


@dataclass
class ConformalBundle:
    """Serialized conformal guardrail bundle with fitted classifier and alpha thresholds."""

    estimator: Any
    feature_cols: list[str]
    alpha_thresholds: dict[str, float]
    default_alpha: float = 0.10
    alpha_min: float = 0.01
    alpha_max: float = 0.30
    model_name: str = "lightgbm"


def _format_alpha_key(alpha: float) -> str:
    """Format alpha level cleanly as a deterministic string key."""
    rounded = round(float(alpha), 2)
    if abs(float(alpha) - rounded) <= 1e-9:
        return f"{rounded:.2f}"
    return f"{float(alpha):.6g}"


def _validate_binary_labels(y: Any, expected_len: int) -> np.ndarray:
    """Validate 1D binary integer labels {0, 1} matching expected sample count."""
    y_float = np.asarray(y, dtype=float)
    if y_float.ndim != 1 or len(y_float) != expected_len:
        raise ValueError(f"Expected 1D label array of length {expected_len}, got shape {y_float.shape}.")
    if not np.all(np.isfinite(y_float)) or not np.all(np.isin(y_float, (0.0, 1.0))):
        raise ValueError("Calibration/test labels must be finite binary values in {0, 1}.")
    return y_float.astype(int)


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute uniform-bin Expected Calibration Error (ECE) over [0, 1]."""
    probs_arr = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y_arr = np.asarray(y_true, dtype=float)
    if len(y_arr) == 0:
        raise ValueError("Cannot compute ECE on empty arrays.")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_arr)
    for idx in range(n_bins):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        mask = (probs_arr >= lo) & (probs_arr <= hi) if idx == 0 else (probs_arr > lo) & (probs_arr <= hi)
        count = int(np.sum(mask))
        if count > 0:
            ece += (count / n) * abs(float(np.mean(y_arr[mask])) - float(np.mean(probs_arr[mask])))
    return float(ece)


def _compute_split_metrics(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
    threshold: float,
    alpha: float | None,
) -> dict[str, Any]:
    """Compute false-accept rate, abstention rate, ECE, and Brier score for a risk cutoff."""
    pass_mask = risk_scores <= threshold
    false_accept_rate = float(np.mean((y_true == 1) & pass_mask))
    abstention_rate = float(np.mean(~pass_mask))
    ece = compute_ece(y_true, risk_scores)
    brier_score = float(brier_score_loss(y_true, np.clip(risk_scores, 0.0, 1.0)))
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "false_accept_rate": false_accept_rate,
        "abstention_rate": abstention_rate,
        "ece": ece,
        "brier_score": brier_score,
    }
    if alpha is not None:
        metrics["alpha"] = float(alpha)
    return metrics


def calibrate_conformal_thresholds(
    estimator: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    alphas: list[float],
) -> dict[str, float]:
    """Calibrate per-alpha hallucination probability cutoffs via MAPIE LAC split conformal."""
    if not alphas:
        raise ValueError("alphas must not be empty.")
    X_arr = _validate_feature_array(X_cal)
    y_arr = _validate_binary_labels(y_cal, len(X_arr))

    parsed_alphas = [float(a) for a in alphas]
    for a in parsed_alphas:
        if not np.isfinite(a) or a <= 0.0 or a >= 1.0:
            raise ValueError(f"Each alpha must lie strictly in (0, 1), got {a}.")

    n_cal = len(X_arr)
    mapie_alphas = [a for a in parsed_alphas if n_cal >= max(1.0 / a, 1.0 / (1.0 - a))]
    confidence_levels: float | list[float] = (
        [1.0 - a for a in mapie_alphas] if mapie_alphas else 0.5
    )

    scc = SplitConformalClassifier(
        estimator=estimator,
        confidence_level=confidence_levels,
        conformity_score="lac",
        prefit=True,
    )
    # Single-class calibration splits (e.g., all-negative y_cal) emit a benign label-count warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*conformalization dataset has less labels.*",
            category=UserWarning,
        )
        scc.conformalize(X_arr, y_arr)

    quantiles_by_alpha: dict[float, float] = {}
    if mapie_alphas:
        scc.predict_set(X_arr[:1])
        quantiles = np.asarray(scc._mapie_classifier.quantiles_, dtype=float).ravel()
        for a, q in zip(mapie_alphas, quantiles, strict=True):
            quantiles_by_alpha[a] = float(q)

    scores = np.asarray(scc.conformity_scores, dtype=float).ravel()
    thresholds: dict[str, float] = {}
    for a in parsed_alphas:
        if a in quantiles_by_alpha:
            q_hat = quantiles_by_alpha[a]
        else:
            # Finite-sample conservative conformal quantile fallback when n_cal < max(1/a, 1/(1-a)).
            level = float(np.ceil((n_cal + 1) * (1.0 - a)) / n_cal)
            q_hat = 1.0 if level >= 1.0 else float(np.quantile(scores, level, method="higher"))
        cutoff = float(np.clip(1.0 - q_hat, 0.0, 1.0))
        thresholds[_format_alpha_key(a)] = cutoff

    return thresholds


def _resolve_threshold(bundle: ConformalBundle, alpha: float | None) -> tuple[float, float]:
    """Validate alpha bounds and look up or linearly interpolate its calibrated threshold."""
    target_alpha = float(bundle.default_alpha if alpha is None else alpha)
    if (
        not np.isfinite(target_alpha)
        or target_alpha < bundle.alpha_min - 1e-12
        or target_alpha > bundle.alpha_max + 1e-12
    ):
        raise ValueError(
            f"alpha must be within [{bundle.alpha_min}, {bundle.alpha_max}], got {alpha}."
        )
    if not bundle.alpha_thresholds:
        raise ValueError("ConformalBundle.alpha_thresholds must not be empty.")

    pairs = sorted((float(k), float(v)) for k, v in bundle.alpha_thresholds.items())
    known_alphas = np.array([p[0] for p in pairs], dtype=float)
    known_thresholds = np.array([p[1] for p in pairs], dtype=float)

    diffs = np.abs(known_alphas - target_alpha)
    min_idx = int(np.argmin(diffs))
    if diffs[min_idx] <= 1e-9:
        return target_alpha, float(known_thresholds[min_idx])
    return target_alpha, float(np.interp(target_alpha, known_alphas, known_thresholds))


def evaluate_conformal_split(
    bundle: ConformalBundle,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Evaluate conformal guardrail metrics on a held-out feature/label split."""
    target_alpha, threshold = _resolve_threshold(bundle, alpha)
    X_arr = _validate_feature_array(X_test)
    y_arr = _validate_binary_labels(y_test, len(X_arr))
    risk_scores = predict_risk_proba(bundle.estimator, X_arr)
    metrics = _compute_split_metrics(y_arr, risk_scores, threshold, alpha=target_alpha)
    return {k: float(v) for k, v in metrics.items()}


def predict_risk_and_decision(
    bundle: ConformalBundle,
    features_df: pd.DataFrame,
    alpha: float | None = None,
) -> tuple[float, Literal["PASS", "ABSTAIN"], float]:
    """Score a single candidate row and emit PASS/ABSTAIN against the calibrated threshold."""
    _, threshold = _resolve_threshold(bundle, alpha)
    missing = [col for col in bundle.feature_cols if col not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if len(features_df) != 1:
        raise ValueError(f"Expected single-row features DataFrame, got {len(features_df)} rows.")

    X = np.asarray(features_df[bundle.feature_cols], dtype=float)
    risk_score = float(predict_risk_proba(bundle.estimator, X)[0])
    decision: Literal["PASS", "ABSTAIN"] = "PASS" if risk_score <= threshold else "ABSTAIN"
    return risk_score, decision, float(threshold)


def run_conformal_pipeline(
    features_parquet: Path,
    config_path: Path,
    artifacts_dir: Path,
) -> tuple[Path, Path, Path]:
    """Train base risk models, calibrate MAPIE thresholds, and export sweep/metrics artifacts."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    random_state = int(cfg.get("random_state", 42))
    default_alpha = float(cfg.get("default_alpha", 0.10))
    alpha_min = float(cfg.get("alpha_min", 0.01))
    alpha_max = float(cfg.get("alpha_max", 0.30))
    alpha_step = float(cfg.get("alpha_step", 0.01))

    alphas = [
        round(float(a), 4)
        for a in np.arange(alpha_min, alpha_max + alpha_step * 0.5, alpha_step)
        if a <= alpha_max + 1e-9
    ]

    df = pd.read_parquet(features_parquet)
    df_train = df[df["split"] == "train"]
    df_cal = df[df["split"] == "cal"]
    df_test = df[df["split"] == "test"]
    if df_train.empty or df_cal.empty or df_test.empty:
        raise ValueError("Features parquet must contain non-empty 'train', 'cal', and 'test' splits.")

    models = train_risk_classifiers(df_train, random_state=random_state)
    X_cal = df_cal[RISK_FEATURE_COLS].to_numpy(dtype=float)
    y_cal = _validate_binary_labels(df_cal["has_hallucination"], len(df_cal))
    X_test = df_test[RISK_FEATURE_COLS].to_numpy(dtype=float)
    y_test = _validate_binary_labels(df_test["has_hallucination"], len(df_test))

    bundles: dict[str, ConformalBundle] = {}
    sweeps: dict[str, list[dict[str, float]]] = {}
    for name in ("logistic", "lightgbm"):
        thresholds = calibrate_conformal_thresholds(models[name], X_cal, y_cal, alphas)
        bundle = ConformalBundle(
            estimator=models[name],
            feature_cols=list(RISK_FEATURE_COLS),
            alpha_thresholds=thresholds,
            default_alpha=default_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            model_name=name,
        )
        bundles[name] = bundle
        sweeps[name] = [evaluate_conformal_split(bundle, X_test, y_test, a) for a in alphas]

    primary_bundle = bundles["lightgbm"]
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    calibrator_path = artifacts_path / "calibrator.pkl"
    with calibrator_path.open("wb") as fh:
        pickle.dump(primary_bundle, fh)

    sweep_path = artifacts_path / "conformal_sweep.json"
    sweep_payload = {
        "status": "ready",
        "model": primary_bundle.model_name,
        "default_alpha": default_alpha,
        "n_train": len(df_train),
        "n_cal": len(df_cal),
        "n_test": len(df_test),
        "sweep": sweeps["lightgbm"],
        "logistic_sweep": sweeps["logistic"],
    }
    sweep_path.write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")

    heuristic_scores = np.clip(1.0 - df_test["cosine_sim"].to_numpy(dtype=float), 0.0, 1.0)
    heuristic_metrics = _compute_split_metrics(y_test, heuristic_scores, threshold=0.5, alpha=None)

    metrics_path = artifacts_path / "risk_metrics.json"
    risk_metrics_payload = {
        "status": "ready",
        "dataset": str(features_parquet),
        "split_sizes": {
            "train": len(df_train),
            "cal": len(df_cal),
            "test": len(df_test),
        },
        "uncalibrated_heuristic": {
            "guardrail_method": "Uncalibrated Heuristic Threshold",
            "target_alpha": None,
            **heuristic_metrics,
        },
        "logistic_mapie": {
            "0.10": {
                "guardrail_method": "Logistic + Split Conformal (MAPIE)",
                "target_alpha": 0.10,
                **evaluate_conformal_split(bundles["logistic"], X_test, y_test, alpha=0.10),
            },
        },
        "lightgbm_mapie": {
            "0.10": {
                "guardrail_method": "LightGBM + Split Conformal (MAPIE)",
                "target_alpha": 0.10,
                **evaluate_conformal_split(bundles["lightgbm"], X_test, y_test, alpha=0.10),
            },
            "0.05": {
                "guardrail_method": "LightGBM + Split Conformal (MAPIE)",
                "target_alpha": 0.05,
                **evaluate_conformal_split(bundles["lightgbm"], X_test, y_test, alpha=0.05),
            },
        },
    }
    metrics_path.write_text(json.dumps(risk_metrics_payload, indent=2), encoding="utf-8")

    return calibrator_path, sweep_path, metrics_path


if __name__ == "__main__":
    from src.conformal.calibrator import run_conformal_pipeline as _cli_run_pipeline

    cal_path, sweep_path, metrics_path = _cli_run_pipeline(
        Path("data/processed/ragtruth_features.parquet"),
        Path("config/ragtruth_conformal.yaml"),
        Path("artifacts"),
    )
    print(f"Exported: {cal_path}, {sweep_path}, {metrics_path}")
