# Phase 2: Judge Bias Audit & Conformal Hallucination Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit LLM-as-a-Judge agreement and verbosity/position bias on `RAGTruth`, then train calibrated hallucination risk classifiers (`LogisticRegression`, `LightGBM`) and finite-sample `MAPIE` conformal guardrails producing `calibrator.pkl`.

**Architecture:** Branch `feat/judge-bias-audit` implements `src/eval/judge_audit.py` and `notebooks/01_judge_bias_audit.ipynb`. Branch `feat/conformal-guardrail` implements `src/conformal/risk_model.py`, `src/conformal/calibrator.py`, and `notebooks/02_conformal_guardrail.ipynb`, exporting `artifacts/calibrator.pkl`, `artifacts/conformal_sweep.json`, and `artifacts/risk_metrics.json`.

**Tech Stack:** Python 3.12, `uv`, `MAPIE`, `LightGBM`, `scikit-learn`, `scipy`, `httpx`, `pytest`

**Spec:** [2026-11-25-conformal-llm-router-design.md](../specs/2026-11-25-conformal-llm-router-design.md)

## Global Constraints

- Paid judge audit sampling (if raw `RAGTruth` lacks automated judge scores) is capped at `<= 300` paired examples via free-tier `Groq`/`Gemini` or low-cost `deepseek-chat` (`<$0.50`).
- Split Conformal calibration must strictly use the held-out `cal` split (never `train` or `test`) so exchangeability guarantees hold on `test`.
- `RISK_FEATURE_COLS` must match the exact columns produced by `src/eval/features.py`.
- Target sweep range: `alpha` in `[0.01, 0.30]` at step `0.01` (`default_alpha = 0.10`), also reporting benchmark rows at `alpha = 0.10` and `0.05`.

## Review Focus

- Calling `predict_risk_and_decision` when `alpha` is outside `[0.01, 0.30]` or missing from the precomputed sweep must interpolate or raise a clear `ValueError` rather than crashing with `KeyError`.
- `calibrate_conformal_thresholds` when a calibration split has all-negative or extreme class imbalance must handle boundary quantiles cleanly without returning `NaN`.
- `fit_bias_correction` on constant response lengths (`std == 0`) must avoid singular matrix errors during OLS/logistic regression adjustment.
- `run_judge_audit` must retry or gracefully fall back when a sample API call hits `429` without losing already-scored rows.
- Deserializing `artifacts/calibrator.pkl` across modules must not depend on `__main__` symbols (all pickled dataclasses live in `src/conformal/`).

---

### Task 1: Bias-Corrected LLM-as-a-Judge Audit (`feat/judge-bias-audit`)

**Files:**
- Create: `src/eval/judge_audit.py`
- Create: `notebooks/01_judge_bias_audit.ipynb`
- Test: `tests/test_judge_audit.py`

**Interfaces:**
- Consumes: `data/processed/ragtruth_features.parquet`, optional async `judge_fn`
- Produces:
  - `compute_judge_metrics(human_labels: Sequence[int], judge_scores: Sequence[float], task_types: Sequence[str]) -> dict[str, Any]`
  - `fit_bias_correction(judge_scores: np.ndarray, response_lengths: np.ndarray, positions: np.ndarray, human_labels: np.ndarray) -> dict[str, Any]`
  - `run_judge_audit(features_parquet: Path, out_json: Path, sample_size: int = 300, judge_fn: Callable[[str, str, str], Awaitable[float]] | None = None) -> dict[str, Any]`

- [ ] **Step 1: Create `feat/judge-bias-audit` branch and write failing judge audit tests**

Run: `git checkout -b feat/judge-bias-audit`
```python
# tests/test_judge_audit.py
import numpy as np
from src.eval.judge_audit import compute_judge_metrics, fit_bias_correction

def test_compute_judge_metrics_kappa_and_spearman():
    human = [0, 0, 1, 1, 0, 1]
    judge = [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]
    tasks = ["QA", "QA", "QA", "Summary", "Summary", "Summary"]
    res = compute_judge_metrics(human, judge, tasks)
    assert res["cohens_kappa"] > 0.8
    assert res["spearman_r"] > 0.8
    assert set(res["by_task"].keys()) == {"QA", "Summary"}

def test_fit_bias_correction_handles_constant_length_and_corrects_verbosity():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=80)
    lengths = rng.integers(20, 400, size=80)
    positions = rng.integers(0, 2, size=80)
    biased_scores = np.clip(0.2 + 0.5 * y + 0.001 * lengths + 0.1 * positions, 0, 1)
    res = fit_bias_correction(biased_scores, lengths, positions, y)
    assert "verbosity_coef" in res and "position_coef" in res
    assert res["corrected_spearman_r"] >= res["raw_spearman_r"] - 0.05
    # Constant length edge case
    res_const = fit_bias_correction(biased_scores, np.ones(80), positions, y)
    assert np.isfinite(res_const["verbosity_coef"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_judge_audit.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/eval/judge_audit.py` and `notebooks/01_judge_bias_audit.ipynb`**

Implement `compute_judge_metrics` (`cohen_kappa_score`, `spearmanr`, `confusion_matrix`), `fit_bias_correction` (`Ridge`/`LinearRegression` de-biasing residualizing length & position), and `run_judge_audit` (stratified `<=300` sampler with `429` backoff + checkpoint saving to `artifacts/judge_audit_summary.json`). Author `notebooks/01_judge_bias_audit.ipynb` explaining verbosity & position bias visually.

- [ ] **Step 4: Run tests and verify pass**

Run: `uv run pytest tests/test_judge_audit.py -v && uv run ruff check src/eval/judge_audit.py`
Expected: `2 passed`, zero lint warnings.

- [ ] **Step 5: Run audit notebook/script, commit, and merge `feat/judge-bias-audit` to `main`**

---

### Task 2: Base Hallucination Risk Classifiers (`feat/conformal-guardrail`)

**Files:**
- Create: `src/conformal/risk_model.py`
- Test: `tests/test_conformal.py` (Part 1)

**Interfaces:**
- Consumes: `data/processed/ragtruth_features.parquet` (`RISK_FEATURE_COLS`)
- Produces:
  - `RISK_FEATURE_COLS: list[str]`
  - `train_risk_classifiers(df_train: pd.DataFrame, random_state: int = 42) -> dict[str, Any]`
  - `predict_risk_proba(estimator: Any, X: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Create `feat/conformal-guardrail` branch and write failing risk model test**

Run: `git checkout -b feat/conformal-guardrail`
```python
# tests/test_conformal.py
import numpy as np
import pandas as pd
from src.conformal.risk_model import RISK_FEATURE_COLS, train_risk_classifiers, predict_risk_proba

def test_train_risk_classifiers_returns_valid_probabilities():
    rng = np.random.default_rng(42)
    data = {col: rng.normal(size=120) for col in RISK_FEATURE_COLS}
    data["has_hallucination"] = (data["nli_contradiction"] > 0).astype(int)
    df = pd.DataFrame(data)
    models = train_risk_classifiers(df, random_state=42)
    assert set(models.keys()) == {"logistic", "lightgbm"}
    probs = predict_risk_proba(models["lightgbm"], df[RISK_FEATURE_COLS].to_numpy())
    assert probs.shape == (120,) and np.all((probs >= 0.0) & (probs <= 1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conformal.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/conformal/risk_model.py`**

Build standardized `LogisticRegression` pipeline and `LightGBM` binary classifier predicting $P(\text{hallucination})$.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_conformal.py -v`
Expected: `1 passed`.

---

### Task 3: MAPIE Conformal Calibration & Sweep Artifacts (`feat/conformal-guardrail`)

**Files:**
- Create: `src/conformal/calibrator.py`
- Create: `notebooks/02_conformal_guardrail.ipynb`
- Modify: `tests/test_conformal.py`

**Interfaces:**
- Consumes: `src/conformal/risk_model.py`, `data/processed/ragtruth_features.parquet`, `config/ragtruth_conformal.yaml`
- Produces:
  - `@dataclass class ConformalBundle`
  - `calibrate_conformal_thresholds(estimator: Any, X_cal: np.ndarray, y_cal: np.ndarray, alphas: list[float]) -> dict[str, float]`
  - `evaluate_conformal_split(bundle: ConformalBundle, X_test: np.ndarray, y_test: np.ndarray, alpha: float) -> dict[str, float]`
  - `predict_risk_and_decision(bundle: ConformalBundle, features_df: pd.DataFrame, alpha: float | None = None) -> tuple[float, Literal["PASS", "ABSTAIN"], float]`
  - `run_conformal_pipeline(features_parquet: Path, config_path: Path, artifacts_dir: Path) -> tuple[Path, Path, Path]`

- [ ] **Step 1: Add failing conformal calibration and boundary tests to `tests/test_conformal.py`**

Add tests verifying:
1. Empirical false-accept rate on synthetic calibrated data tracks target $\alpha \in \{0.05, 0.10\}$ within finite-sample tolerance.
2. `predict_risk_and_decision` raises `ValueError` for out-of-range `alpha` (`0.0` or `0.9`) and returns exact `("PASS" | "ABSTAIN")` boundary behavior around `q_hat`.
3. `calibrator.pkl` round-trips cleanly via `pickle`/`joblib` and populates `conformal_sweep.json` + `risk_metrics.json`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conformal.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/conformal/calibrator.py` using `MAPIE` (`SplitConformalClassifier` / nonconformity score quantile on negative/hallucination class)**

Compute per-`alpha` acceptance threshold controlling $P(\text{hallucination}=1 \text{ and decision}=\text{PASS}) \le \alpha$, ECE, and Brier Score; add CLI runner (`uv run python -m src.conformal.calibrator`) and `notebooks/02_conformal_guardrail.ipynb`.

- [ ] **Step 4: Run tests and full Phase 2 calibration pipeline, then commit & merge**

Run: `uv run pytest tests/test_conformal.py -v && uv run python -m src.conformal.calibrator`
Expected: All tests PASS; `artifacts/calibrator.pkl`, `artifacts/conformal_sweep.json`, and `artifacts/risk_metrics.json` created.
Commit and merge `feat/conformal-guardrail` to `main`.
