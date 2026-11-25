# Phase 1: Project Setup, Benchmark Ingestion & Shared Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize the Git repository with `uv` dependency management, ingest `RAGTruth` and `RouterBench` into deterministic splits, and build the shared CPU-friendly feature extractor used identically offline and online.

**Architecture:** Milestone 0 (`chore/project-setup`) creates the `uv` project, configs, and `.gitignore` preserving tracked `.impeccable/` artifacts. Milestone 1 (`feat/data-ingestion-features`) normalizes raw datasets into Parquet splits and computes MiniLM embeddings, NLI scores, entity overlap, and query-complexity signals via `src/eval/features.py`.

**Tech Stack:** Python 3.12, `uv`, `datasets`, `pyarrow`, `pandas`, `numpy`, `sentence-transformers`, `pytest`, `ruff`

**Spec:** [2026-11-25-conformal-llm-router-design.md](../specs/2026-11-25-conformal-llm-router-design.md)

## Global Constraints

- Python version floor `>=3.12,<3.13`; all dependency management via `uv` (`uv add`, `uv run`, `pyproject.toml`, `uv.lock`).
- Host GPU is RTX 3050 6 GB VRAM; offline feature extraction must run cleanly on CPU/6 GB VRAM with zero paid LLM API tokens.
- Preserve tracked `.impeccable/` files (`PRODUCT.md`, `.impeccable/config.json`, future `DESIGN.md`, `.impeccable/surfaces/`) in `.gitignore`; ignore `data/raw/`, `data/processed/`, `artifacts/`, `.env`, `.impeccable/mocks/`, `.impeccable/questions/`, `.impeccable/review/`, `.impeccable/config.local.json`.
- Never fabricate metrics; offline-to-online feature parity requires `src/eval/features.py` to accept both batch lists and single live inputs identically.

## Review Focus

- Empty or whitespace-only `context` / `response` strings passed to `extract_rag_features` must return finite `0.0` ratios/similarities instead of `NaN` or `ZeroDivisionError`.
- Hugging Face `RAGTruth` / `RouterBench` schema variants (nested JSON spans or missing optional columns) must raise a clear `ValueError` naming missing columns rather than silently mislabeling splits.
- Very long `context` passages (>512 tokens) must truncate safely inside embedding and NLI tokenizers without crashing batch extraction.
- Re-running `ingest_ragtruth` or `ingest_routerbench` must produce identical deterministic `train`/`cal`/`test`/`ood_test` splits for a fixed seed.
- `.gitignore` must exclude `data/` and `artifacts/` binaries while keeping `PRODUCT.md` and `.impeccable/config.json` tracked.

---

### Task 1: Repository, `uv` Toolchain & Config Setup (`chore/project-setup`)

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `pyproject.toml`
- Create: `config/ragtruth_conformal.yaml`
- Create: `config/routerbench.yaml`
- Create: `src/__init__.py`, `src/eval/__init__.py`, `src/conformal/__init__.py`, `src/router/__init__.py`, `src/gateway/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `pyproject.toml`, `uv.lock`, `config/ragtruth_conformal.yaml`, `config/routerbench.yaml`, `.env.example`

- [ ] **Step 1: Verify `mise`-managed `uv` is available (do not install a second `uv`) and initialize Git + `origin` remote**

Run: `/home/fahmi/.local/share/mise/shims/uv --version`
Expected: prints `uv ...` from `mise`.
Then run: `git init -b main && git remote add origin https://github.com/Fahmi-mi/conformal-llm-router.git`

- [ ] **Step 2: Create `chore/project-setup` branch, `.gitignore`, `.env.example`, and directory scaffolds**

Run: `git checkout -b chore/project-setup`
Write `.gitignore` ignoring `.venv/`, `__pycache__/`, `.pytest_cache/`, `data/raw/`, `data/processed/`, `data/telemetry.db*`, `artifacts/`, `.env`, `.impeccable/mocks/`, `.impeccable/questions/`, `.impeccable/review/`, `.impeccable/config.local.json`.
Write `.env.example` with `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_BASE_URL=http://localhost:11434`, `GATEWAY_ALPHA_DEFAULT=0.10`, `GATEWAY_LAMBDA_DEFAULT=0.45`.

- [ ] **Step 3: Initialize `uv` project and add baseline dependencies**

Write `pyproject.toml` (`requires-python = ">=3.12,<3.13"`), then run:
`uv add pyyaml pandas numpy pyarrow datasets scikit-learn lightgbm mapie sentence-transformers faiss-cpu fastapi uvicorn httpx pydantic pydantic-settings`
`uv add --dev pytest pytest-asyncio ruff ipykernel`

- [ ] **Step 4: Write `config/ragtruth_conformal.yaml`, `config/routerbench.yaml`, and failing config test**

```python
# tests/test_config.py
import yaml
from pathlib import Path

def test_configs_have_required_keys():
    rag = yaml.safe_load(Path("config/ragtruth_conformal.yaml").read_text())
    rtr = yaml.safe_load(Path("config/routerbench.yaml").read_text())
    assert rag["random_state"] == 42
    assert 0.0 < rag["default_alpha"] < 0.5
    assert rtr["ood_holdout_tasks"]
    assert 0.0 <= rtr["default_lambda_cost"] <= 1.0
```

- [ ] **Step 5: Verify config test & `uv run` environment, then commit and merge to `main`**

Run: `uv run pytest tests/test_config.py -v`
Expected: `1 passed`.
Commit on `chore/project-setup`, merge to `main`:
`git add .gitignore .env.example pyproject.toml uv.lock config/ src/ tests/ PRODUCT.md README.md .impeccable/config.json docs/`
`git commit -m "chore: initialize uv project, configs, and repo layout"`

---

### Task 2: Benchmark Dataset Ingestion & Deterministic Splitting (`feat/data-ingestion-features`)

**Files:**
- Create: `src/eval/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `config/ragtruth_conformal.yaml`, `config/routerbench.yaml`
- Produces:
  - `normalize_ragtruth_df(raw_df: pd.DataFrame, cal_ratio: float = 0.2, random_state: int = 42) -> pd.DataFrame`
  - `normalize_routerbench_df(raw_df: pd.DataFrame, ood_tasks: list[str], random_state: int = 42) -> pd.DataFrame`
  - `ingest_ragtruth(output_path: Path, cal_ratio: float = 0.2, random_state: int = 42) -> Path`
  - `ingest_routerbench(output_path: Path, ood_tasks: list[str], random_state: int = 42) -> Path`

- [ ] **Step 1: Checkout feature branch and write failing ingestion tests**

Run: `git checkout -b feat/data-ingestion-features`
```python
# tests/test_ingest.py
import pandas as pd
import pytest
from src.eval.ingest import normalize_ragtruth_df, normalize_routerbench_df

def test_normalize_ragtruth_deterministic_splits_and_schema():
    raw = pd.DataFrame({
        "id": [f"r{i}" for i in range(10)],
        "task_type": ["QA"] * 5 + ["Summary"] * 5,
        "split": ["train"] * 8 + ["test"] * 2,
        "prompt": ["q?"] * 10,
        "context": ["ctx"] * 10,
        "response": ["ans"] * 10,
        "labels": [[]] * 7 + [[{"label_type": "Evident Conflict"}]] * 3,
    })
    df1 = normalize_ragtruth_df(raw, cal_ratio=0.25, random_state=42)
    df2 = normalize_ragtruth_df(raw, cal_ratio=0.25, random_state=42)
    assert df1["split"].tolist() == df2["split"].tolist()
    assert set(df1["split"]) == {"train", "cal", "test"}
    assert df1["has_hallucination"].dtype == bool

def test_normalize_ragtruth_missing_column_raises():
    with pytest.raises(ValueError, match="missing required"):
        normalize_ragtruth_df(pd.DataFrame({"id": ["1"]}))

def test_normalize_routerbench_ood_holdout_and_tiers():
    raw = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "eval_name": ["mmlu", "gsm8k", "mbpp", "hellaswag"],
        "prompt": ["p1", "p2", "p3", "p4"],
        "mistralai/mistral-7b-chat": [0.9, 0.2, 0.8, 0.7],
        "gpt-4-1106-preview": [0.92, 0.9, 0.85, 0.75],
        "mistralai/mistral-7b-chat|total_cost": [0.001, 0.001, 0.001, 0.001],
        "gpt-4-1106-preview|total_cost": [0.03, 0.03, 0.03, 0.03],
    })
    df = normalize_routerbench_df(raw, ood_tasks=["mbpp"], random_state=42)
    assert df.loc[df["eval_name"] == "mbpp", "split"].iloc[0] == "ood_test"
    assert set(["tier1_quality", "tier2_quality", "tier1_cost", "tier2_cost", "tier1_sufficient"]).issubset(df.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.eval.ingest'`).

- [ ] **Step 3: Implement `src/eval/ingest.py` with schema validation, Hugging Face downloaders, and deterministic splitting**

Inspect actual `ParticleMedia/RAGTruth` (or `wandb/RAGTruth-processed`) and `withmartian/routerbench` column schemas on download, map to canonical columns, and implement `normalize_ragtruth_df`, `normalize_routerbench_df`, `ingest_ragtruth`, `ingest_routerbench`, and CLI entrypoint (`python -m src.eval.ingest`).

- [ ] **Step 4: Run ingestion tests and verify pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit Task 2**

Run: `git add src/eval/ingest.py tests/test_ingest.py && git commit -m "feat(eval): add RAGTruth and RouterBench ingestion and splitting"`

---

### Task 3: Shared CPU Feature Extractor (`src/eval/features.py`)

**Files:**
- Create: `src/eval/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: Parquet files produced by `src/eval/ingest.py`
- Produces:
  - `@dataclass class FeatureExtractorConfig` (`embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"`, `nli_model: str = "cross-encoder/nli-deberta-v3-small"`)
  - `class FeatureExtractor`:
    - `extract_query_features(queries: list[str]) -> tuple[np.ndarray, pd.DataFrame]`
    - `extract_rag_features(queries: list[str], contexts: list[str], responses: list[str]) -> pd.DataFrame`
  - `build_feature_tables(ragtruth_path: Path, routerbench_path: Path, out_dir: Path, extractor: FeatureExtractor | None = None) -> tuple[Path, Path]`

- [ ] **Step 1: Write failing feature extractor tests (including Review Focus edge cases)**

```python
# tests/test_features.py
import numpy as np
import pytest
from src.eval.features import FeatureExtractor

def test_extract_rag_features_handles_empty_strings_without_nan():
    ext = FeatureExtractor(embedder=..., nli_model=...) # lightweight fake stubs for fast unit tests
    df = ext.extract_rag_features(["q"], [""], [""])
    assert not df.isna().any().any()
    assert df["length_ratio"].iloc[0] == 0.0
    assert df["entity_overlap"].iloc[0] == 0.0

def test_extract_query_features_shape_and_l2_norm():
    ext = FeatureExtractor(embedder=..., nli_model=...)
    embs, meta = ext.extract_query_features(["Explain conformal prediction.", "2 + 2?"])
    assert embs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-4)
    assert set(["char_len", "token_count", "sentence_count", "question_mark_count"]).issubset(meta.columns)

def test_extract_rag_features_truncates_long_passages_safely():
    ext = FeatureExtractor(embedder=..., nli_model=...)
    long_ctx = "word " * 2000
    df = ext.extract_rag_features(["q"], [long_ctx], ["short answer"])
    assert np.isfinite(df["cosine_sim"].iloc[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.eval.features'`).

- [ ] **Step 3: Implement `src/eval/features.py` with injectable embedder/NLI models (lazy-loading real `sentence-transformers` defaults)**

Ensure `FeatureExtractor` handles empty strings (`0.0` safe division), truncates NLI pairs to `max_length=512`, computes entity overlap via lightweight capitalized/numeric token set overlap (zero heavy spaCy download needed), and includes CLI entrypoint (`uv run python -m src.eval.features`) writing `data/processed/ragtruth_features.parquet` and `data/processed/routerbench_features.parquet`.

- [ ] **Step 4: Run tests, generate real Parquet feature tables, and commit**

Run: `uv run pytest tests/test_features.py -v && uv run ruff check src/ tests/`
Expected: `3 passed`, zero ruff errors.
Run offline ingestion + feature build (`uv run python -m src.eval.ingest && uv run python -m src.eval.features`), verify Parquet row counts, commit & merge `feat/data-ingestion-features` to `main`.
