# Phase 3: Tier-Mapped Learned Pareto Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a query-only Tier-Mapped Learned Pareto Router on `RouterBench`, evaluate cost-quality Pareto frontiers (including out-of-domain generalization), and export `artifacts/router.pkl` + `artifacts/pareto_sweep.json`.

**Architecture:** Branch `feat/pareto-router` builds `src/router/trainer.py`, `src/router/pareto.py`, `notebooks/03_pareto_router.ipynb`, and `tests/test_router.py`. Only query-side features (`all-MiniLM-L6-v2` embeddings + query metadata) enter the classifier so `router.pkl` transfers directly to live FastAPI prompts.

**Tech Stack:** Python 3.12, `uv`, `LightGBM`, `scikit-learn`, `numpy`, `pandas`, `pytest`

**Spec:** [2026-11-25-conformal-llm-router-design.md](../specs/2026-11-25-conformal-llm-router-design.md)

## Global Constraints

- **Tier-Mapped Router only:** `router.pkl` must use strictly query-side features (no `RouterBench` historical model-ID or ground-truth latency columns at inference time).
- Sweep parameter `lambda_cost` covers `[0.00, 1.00]` at step `0.05` (`default_lambda = 0.45`).
- Report baselines (`Always Frontier`, `Always Small`, `Random Router`, `Learned Pareto Router`) plus `AUC-QC`, cost savings at `>= 95%` frontier quality retention, and `ood_test` holdout performance.
- All pickled classes (`RouterBundle`) must reside in `src/router/trainer.py`.

## Review Focus

- `select_tier` at boundary thresholds `lambda_cost = 0.0` (prefer maximum quality / Tier 2) and `lambda_cost = 1.0` (prefer minimum cost / Tier 1) must behave deterministically without inverted logic.
- `compute_pareto_frontier` when `tier2_cost` equals `tier1_cost` or total frontier cost is `0` must avoid `ZeroDivisionError`.
- `train_pareto_router` with fewer queries than default PCA components must clamp PCA dimensions safely or skip PCA cleanly.
- Out-of-domain split evaluation must work gracefully even if `ood_test` has zero rows in a synthetic test fixture.
- `predict_tier1_probability` must validate matching embedding dimensionality and return bounded probabilities in `[0.0, 1.0]`.

---

### Task 1: Query-Only Tier-Mapped Router Trainer (`feat/pareto-router`)

**Files:**
- Create: `src/router/trainer.py`
- Test: `tests/test_router.py` (Part 1)

**Interfaces:**
- Consumes: `data/processed/routerbench_features.parquet`, `src/eval/features.py` query metadata columns
- Produces:
  - `@dataclass class RouterBundle` (`estimator: Any`, `query_meta_cols: list[str]`, `pca: Any | None`, `default_lambda: float`)
  - `train_pareto_router(df_train: pd.DataFrame, embeddings_train: np.ndarray, query_meta_cols: list[str], n_pca: int = 32, random_state: int = 42) -> RouterBundle`
  - `predict_tier1_probability(bundle: RouterBundle, query_embeddings: np.ndarray, query_meta_df: pd.DataFrame) -> np.ndarray`
  - `select_tier(prob_tier1: float, lambda_cost: float) -> Literal["tier1", "tier2"]`

- [ ] **Step 1: Create `feat/pareto-router` branch and write failing trainer tests**

Run: `git checkout -b feat/pareto-router`
```python
# tests/test_router.py
import numpy as np
import pandas as pd
from src.router.trainer import train_pareto_router, predict_tier1_probability, select_tier

def test_router_training_and_boundary_tier_selection():
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(60, 384)).astype(np.float32)
    meta = pd.DataFrame({"char_len": rng.integers(10, 200, size=60), "token_count": rng.integers(3, 50, size=60)})
    meta["tier1_sufficient"] = (meta["token_count"] < 25).astype(bool)
    bundle = train_pareto_router(meta, embs, ["char_len", "token_count"], n_pca=16, random_state=42)
    probs = predict_tier1_probability(bundle, embs[:5], meta.iloc[:5])
    assert probs.shape == (5,) and np.all((probs >= 0.0) & (probs <= 1.0))
    assert select_tier(0.5, lambda_cost=0.0) == "tier2"
    assert select_tier(0.5, lambda_cost=1.0) == "tier1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_router.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/router/trainer.py`**

Fit `PCA` (clamping `n_components = min(n_pca, n_samples, n_features)`) + `LightGBM` predicting `tier1_sufficient`, and implement `select_tier` mapping `lambda_cost` to decision threshold `1.0 - lambda_cost` (so higher cost sensitivity `lambda_cost -> 1.0` routes more queries to Tier 1).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_router.py -v`
Expected: `1 passed`.

---

### Task 2: Pareto Frontier Evaluation, OOD Robustness & Artifact Export (`feat/pareto-router`)

**Files:**
- Create: `src/router/pareto.py`
- Create: `notebooks/03_pareto_router.ipynb`
- Modify: `tests/test_router.py`

**Interfaces:**
- Consumes: `src/router/trainer.py`, `data/processed/routerbench_features.parquet`, `config/routerbench.yaml`
- Produces:
  - `compute_pareto_frontier(probs_tier1: np.ndarray, df_eval: pd.DataFrame, lambdas: list[float]) -> dict[str, Any]`
  - `run_router_pipeline(features_parquet: Path, config_path: Path, artifacts_dir: Path) -> tuple[Path, Path]`

- [ ] **Step 1: Add failing Pareto evaluation & zero-cost edge-case tests to `tests/test_router.py`**

Test `compute_pareto_frontier` on synthetic evaluation rows verifying:
1. Monotonic/expected cost vs quality endpoints (`lambda=0.0` matches `Always Frontier`, `lambda=1.0` matches `Always Small`), `auc_qc` in `[0, 1]`, and `baselines` dictionary keys.
2. Zero-cost `tier2_cost` rows do not trigger `ZeroDivisionError`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_router.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/router/pareto.py` and `notebooks/03_pareto_router.ipynb`**

Compute sweep curve, trapezoidal `AUC-QC`, cost savings at `>=95%` frontier quality, random/always-strong/always-weak baselines, and `ood_test` holdout evaluation; export `artifacts/router.pkl` and `artifacts/pareto_sweep.json`. Add CLI entrypoint (`uv run python -m src.router.pareto`).

- [ ] **Step 4: Run tests and full Phase 3 router pipeline, then commit & merge**

Run: `uv run pytest tests/test_router.py -v && uv run python -m src.router.pareto`
Expected: All tests PASS; `artifacts/router.pkl` and `artifacts/pareto_sweep.json` created.
Commit and merge `feat/pareto-router` to `main`.
