import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.conformal.calibrator import (
    ConformalBundle,
    calibrate_conformal_thresholds,
    evaluate_conformal_split,
    predict_risk_and_decision,
    run_conformal_pipeline,
)
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


class _DummyBinaryEstimator:
    """Deterministic binary estimator returning exact risk probabilities from column 0."""

    classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DummyBinaryEstimator":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.clip(np.asarray(X, dtype=float)[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])


def _make_split_df(n_train: int = 400, n_cal: int = 500, n_test: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    total = n_train + n_cal + n_test
    data: dict[str, np.ndarray | list[str]] = {col: rng.normal(size=total) for col in RISK_FEATURE_COLS}
    cosine = np.clip(0.5 + 0.25 * np.asarray(data["cosine_sim"], dtype=float), 0.0, 1.0)
    data["cosine_sim"] = cosine
    logits = 2.0 * np.asarray(data["nli_contradiction"], dtype=float) - 2.5 * (cosine - 0.5)
    probs = 1.0 / (1.0 + np.exp(-logits))
    data["has_hallucination"] = rng.binomial(1, probs).astype(bool)
    data["split"] = ["train"] * n_train + ["cal"] * n_cal + ["test"] * n_test
    return pd.DataFrame(data)


def test_empirical_false_accept_rate_tracks_alpha():
    df = _make_split_df(n_train=400, n_cal=600, n_test=600, seed=7)
    df_train = df[df["split"] == "train"]
    df_cal = df[df["split"] == "cal"]
    df_test = df[df["split"] == "test"]

    models = train_risk_classifiers(df_train, random_state=42)
    X_cal = df_cal[RISK_FEATURE_COLS].to_numpy(dtype=float)
    y_cal = df_cal["has_hallucination"].astype(int).to_numpy()
    X_test = df_test[RISK_FEATURE_COLS].to_numpy(dtype=float)
    y_test = df_test["has_hallucination"].astype(int).to_numpy()

    alphas = [0.05, 0.10]
    thresholds = calibrate_conformal_thresholds(models["logistic"], X_cal, y_cal, alphas)
    assert set(thresholds.keys()) == {"0.05", "0.10"}
    assert 0.0 <= thresholds["0.05"] <= thresholds["0.10"] <= 1.0

    bundle = ConformalBundle(
        estimator=models["logistic"],
        feature_cols=list(RISK_FEATURE_COLS),
        alpha_thresholds=thresholds,
        default_alpha=0.10,
        alpha_min=0.01,
        alpha_max=0.30,
        model_name="logistic",
    )

    for alpha in alphas:
        metrics = evaluate_conformal_split(bundle, X_test, y_test, alpha=alpha)
        assert metrics["false_accept_rate"] <= alpha + 0.03
        assert 0.0 <= metrics["abstention_rate"] <= 1.0
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert 0.0 <= metrics["threshold"] <= 1.0


def test_predict_risk_and_decision_validation_and_boundary():
    bundle = ConformalBundle(
        estimator=_DummyBinaryEstimator(),
        feature_cols=list(RISK_FEATURE_COLS),
        alpha_thresholds={"0.05": 0.20, "0.10": 0.40},
        default_alpha=0.10,
        alpha_min=0.01,
        alpha_max=0.30,
        model_name="dummy",
    )

    row = pd.DataFrame([{col: 0.0 for col in RISK_FEATURE_COLS}])

    for bad_alpha in (0.0, -0.05, 0.31, 0.9, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            predict_risk_and_decision(bundle, row, alpha=bad_alpha)

    # Exact PASS at boundary risk == threshold
    row_at = row.copy()
    row_at["cosine_sim"] = 0.40
    risk, decision, thresh = predict_risk_and_decision(bundle, row_at, alpha=None)
    assert risk == pytest.approx(0.40)
    assert thresh == pytest.approx(0.40)
    assert decision == "PASS"

    # ABSTAIN just above threshold
    row_above = row.copy()
    row_above["cosine_sim"] = 0.4001
    risk_a, decision_a, thresh_a = predict_risk_and_decision(bundle, row_above, alpha=0.10)
    assert risk_a == pytest.approx(0.4001)
    assert thresh_a == pytest.approx(0.40)
    assert decision_a == "ABSTAIN"

    # Linear interpolation for uncalibrated in-range alpha=0.075 -> threshold=0.30
    row_interp = row.copy()
    row_interp["cosine_sim"] = 0.31
    risk_i, decision_i, thresh_i = predict_risk_and_decision(bundle, row_interp, alpha=0.075)
    assert thresh_i == pytest.approx(0.30)
    assert risk_i == pytest.approx(0.31)
    assert decision_i == "ABSTAIN"


@pytest.mark.parametrize(
    "y_cal",
    [
        np.zeros(60, dtype=int),
        np.ones(60, dtype=int),
        np.array([1] + [0] * 59, dtype=int),
    ],
)
def test_calibrate_handles_extreme_imbalance_and_small_n(y_cal: np.ndarray):
    rng = np.random.default_rng(123)
    X_cal = rng.uniform(0.0, 0.3, size=(len(y_cal), len(RISK_FEATURE_COLS)))
    est = _DummyBinaryEstimator()
    alphas = [0.01, 0.05, 0.10, 0.30]
    thresholds = calibrate_conformal_thresholds(est, X_cal, y_cal, alphas)
    assert len(thresholds) == len(alphas)
    for val in thresholds.values():
        assert np.isfinite(val)
        assert 0.0 <= val <= 1.0


def test_conformal_bundle_pickle_roundtrip():
    df = _make_split_df(n_train=120, n_cal=120, n_test=40, seed=9)
    models = train_risk_classifiers(df[df["split"] == "train"], random_state=42)
    df_cal = df[df["split"] == "cal"]
    thresholds = calibrate_conformal_thresholds(
        models["lightgbm"],
        df_cal[RISK_FEATURE_COLS].to_numpy(dtype=float),
        df_cal["has_hallucination"].astype(int).to_numpy(),
        [0.05, 0.10],
    )
    bundle = ConformalBundle(
        estimator=models["lightgbm"],
        feature_cols=list(RISK_FEATURE_COLS),
        alpha_thresholds=thresholds,
        default_alpha=0.10,
        alpha_min=0.01,
        alpha_max=0.30,
        model_name="lightgbm",
    )
    assert ConformalBundle.__module__ == "src.conformal.calibrator"
    loaded = pickle.loads(pickle.dumps(bundle))
    sample = df[df["split"] == "test"].iloc[[0]][RISK_FEATURE_COLS]
    assert predict_risk_and_decision(loaded, sample, alpha=0.10) == predict_risk_and_decision(
        bundle, sample, alpha=0.10
    )


def test_run_conformal_pipeline_exports_artifacts(tmp_path: Path):
    df = _make_split_df(n_train=150, n_cal=150, n_test=150, seed=11)
    parquet_path = tmp_path / "features.parquet"
    df.to_parquet(parquet_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "random_state: 42\n"
            "default_alpha: 0.10\n"
            "alpha_min: 0.01\n"
            "alpha_max: 0.30\n"
            "alpha_step: 0.01\n"
            "benchmark_alphas:\n"
            "  - 0.05\n"
            "  - 0.10\n"
        ),
        encoding="utf-8",
    )
    artifacts_dir = tmp_path / "artifacts"

    cal_path, sweep_path, metrics_path = run_conformal_pipeline(
        parquet_path, config_path, artifacts_dir
    )
    assert cal_path.exists()
    assert sweep_path.exists()
    assert metrics_path.exists()

    with cal_path.open("rb") as fh:
        bundle = pickle.load(fh)
    assert isinstance(bundle, ConformalBundle)
    assert type(bundle).__module__ == "src.conformal.calibrator"

    sweep_data = json.loads(sweep_path.read_text(encoding="utf-8"))
    assert sweep_data["status"] == "ready"
    assert len(sweep_data["sweep"]) == 30
    for point in sweep_data["sweep"]:
        for key in (
            "alpha",
            "threshold",
            "false_accept_rate",
            "abstention_rate",
            "ece",
            "brier_score",
        ):
            assert key in point
            assert np.isfinite(point[key])

    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_data["status"] == "ready"
    assert "uncalibrated_heuristic" in metrics_data
    assert "0.10" in metrics_data["logistic_mapie"]
    assert "0.10" in metrics_data["lightgbm_mapie"]
    assert "0.05" in metrics_data["lightgbm_mapie"]
