# Conformal Hallucination Control & Learned LLM Cost-Quality Router — System Design

**Date:** 2026-11-25  
**Status:** Ready for user approval (builder check resolved)  
**Spec Source:** [README.md](../../../README.md) · [PRODUCT.md](../../../PRODUCT.md) · Frontend Sketch: [assigned.html](../../../.impeccable/mocks/decision/assigned.html)

---

## 1. Intent & Why

This project serves two combined goals:
1. **Portfolio Showcase:** Demonstrate production-grade **Statistical AI Control & LLMOps**—solving uncalibrated hallucination risk, wasteful single-model routing, and biased automated judges with verifiable mathematics rather than ad-hoc prompt heuristics.
2. **Hands-On Learning Vehicle:** Structure the implementation so every statistical concept (judge bias regression, Split Conformal Prediction via `MAPIE`, and Pareto cost-quality routing) is first derived, visualized, and validated in Jupyter notebooks, then promoted into tested Python modules and an interactive SvelteKit console.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Low/zero-cost offline statistical pipeline:** Download and process `RAGTruth` (~18k responses) and `RouterBench` (>405k outcomes) into reproducible Parquet feature tables and frozen artifacts (`calibrator.pkl`, `router.pkl`) at zero paid tokens for calibration/routing (with at most a tiny stratified judge-audit sample of $\le 300$ pairs in Milestone 2a via free-tier `Groq`/`Gemini` or low-cost `deepseek-chat` $<\$0.50$ if raw `RAGTruth` lacks automated judge scores).
- **Offline-to-online feature parity:** Extract identical CPU-friendly query and context-response features (`all-MiniLM-L6-v2`, NLI entailment/contradiction, entity overlap, length signals) in both offline training and live FastAPI inference.
- **Tier-Mapped Learned Pareto Router:** Train a query-feature router on `RouterBench` (grouped into Tier 1 cheap/small vs. Tier 2 strong/frontier) and serve that exact learned model in the live gateway.
- **Cost-resilient live gateway:** Serve `FastAPI` with `DeepSeek` (`deepseek-chat`) as Tier 2 (and optional judge), free-tier `Groq` / `Gemini Flash-Lite` as Tier 1, optional lightweight `Ollama` fallback, `FAISS` + `SQLite` semantic caching/telemetry, and automatic `429` rate-limit fallback.
- **Interactive SvelteKit console:** Implement the confirmed *Archival Specimen Index* (`Operate` mode, `seed:26a4da6343180544`, `buildPath: code`) frontend with 6 workspaces, dual-mode $(\alpha, \lambda_{\text{cost}})$ simulation controls, and strict `Unrun / TBD` vs. `[Synthetic]` provenance honesty.

### Non-Goals
- Multi-user authentication, RBAC, or public cloud infrastructure provisioning (Terraform/Kubernetes).
- External PostgreSQL deployment for initial local usage (SQLite + local FAISS files are the chosen persistence layer).
- Running large 70B or multi-model local Ollama stacks on the host GPU.
- Online reinforcement learning or automatic retraining of `router.pkl` / `calibrator.pkl` from live traffic.

### Anti-Goals
- **Fabricated evidence:** Never render mock benchmark numbers without an explicit `[Synthetic]` badge or overwrite `TBD` tables before experiments actually execute.
- **Training-serving skew:** Never compute features one way in notebooks and a different way in `src/gateway/`.
- **UI mutation of server defaults:** Dragging $\alpha$ or $\lambda_{\text{cost}}$ sliders in the frontend must simulate curves and pass per-request overrides to the Playground, never silently mutate persistent gateway defaults.
- **Host toolchain lock-in:** Do not block SvelteKit development/running on local `node`/`npm` installation; provide Docker support alongside local scripts.

---

## 3. Environment & Technical Constraints

- **Host Environment:** Linux, `Python 3.12.3`, `/usr/bin/docker`, `273 GB` free disk space, `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (`6144 MiB` VRAM), `/usr/local/bin/ollama` (no local models pulled yet). `node`, `npm`, and `uv` are not currently installed on the host PATH; Python dependency management uses **`uv`** (`uv add` / `uv run` / `uv.lock`, installed in Milestone 0), and SvelteKit runs via Docker unless Node.js is installed locally.
- **Compute Budget:** Heavy local LLM generation on 6 GB VRAM is avoided. Embeddings (`all-MiniLM-L6-v2`), lightweight NLI cross-encoding, `LightGBM`, and `MAPIE` run comfortably on CPU or 6 GB GPU; live LLM calls use `DeepSeek` API credits and free-tier `Groq`/`Gemini` APIs.
- **Git & Branching Strategy:** Git is not yet initialized. Development follows a milestone-branch workflow off `main`:
  - `chore/project-setup` → `feat/data-ingestion-features` → `feat/judge-bias-audit` → `feat/conformal-guardrail` → `feat/pareto-router` → `feat/inference-gateway` → `feat/monitoring-dashboard` → `chore/docker-and-benchmarks`
  - Preserve tracked Impeccable artifacts (`PRODUCT.md`, `.impeccable/config.json`, future `DESIGN.md`, and `.impeccable/surfaces/`) in `.gitignore` while ignoring `.impeccable/mocks/`, `.impeccable/questions/`, `.impeccable/review/`, and `.impeccable/config.local.json`.

---

## 4. Approaches Considered & Chosen

| Approach | Summary | Trade-off | Status |
|---|---|---|---|
| **1. Milestone-by-Milestone Modular + Staged Artifacts** | Follow `README.md` phases sequentially: validate each statistical concept in a notebook, immediately extract clean code + unit tests into `src/` and export frozen `.pkl`/`.json` artifacts, then build FastAPI + SvelteKit on top of real artifacts. | Best alignment with learning goals and offline-to-online parity; UI arrives in the second half. | **Selected** |
| **2. App-First (Gateway & UI First)** | Build FastAPI + SvelteKit with heuristic/dummy routers first, then swap in `MAPIE` and `RouterBench` models later. | Early visual demo, but risks feature-schema rework once real dataset experiments run. | Rejected |
| **3. Contract-First Compose Boilerplate** | Scaffold full `docker-compose`, OpenAPI specs, and CI skeletons on day one before any ML/statistical validation. | High upfront DevOps boilerplate before touching core conformal/Pareto concepts. | Rejected |

---

## 5. System Architecture & Component Design

```text
Offline Pipeline (Zero Paid Tokens for Calibration/Routing; Optional <=300-Pair Judge Sample)
  RAGTruth ──▶ features.py ──▶ judge_audit.py ──▶ risk_model.py + calibrator.py (MAPIE)
                                                       │
                                                       ▼
                                          artifacts/calibrator.pkl + sweeps/*.json
  RouterBench ─▶ features.py ─▶ trainer.py (Tier-Mapped) + pareto.py
                                                       │
                                                       ▼
                                          artifacts/router.pkl + sweeps/*.json

Online Serving (FastAPI + SQLite/FAISS + SvelteKit)
  POST /v1/query ──▶ FAISS Semantic Cache (bypassed if bypass_cache or λ_cost regime changed)
                        ├─ Hit  ──▶ Re-check α_override on cached risk ─▶ Response + Telemetry
                        └─ Miss ──▶ features.py (query) ──▶ router.pkl (λ_cost)
                                      ├─ Tier 1 (Groq / Gemini Free) ─▶ [If context & risk > α: 1x Escalate to Tier 2]
                                      └─ Tier 2 (DeepSeek API)       ─┘
                                                                            │
                                                      ┌─ no context ────────┼─ with context ──▶ calibrator.pkl (α)
                                                      ▼                     ▼
                                                  UNSCORED         PASS (cache) / ABSTAIN (safe fallback; raw draft in DB)
```

### 5.1 Milestone 1: Benchmark Ingestion & Shared Feature Extractor (`src/eval/features.py`)
- **Datasets:** Download `RAGTruth` (`ParticleMedia/RAGTruth` / `wandb/RAGTruth-processed`) and `RouterBench` (`withmartian/routerbench`) into `data/raw/`.
- **Deterministic Splits:**
  - `RAGTruth`: preserve train/test split and carve a stratified calibration fold (`train` / `cal` / `test`) across task types (`QA`, `summarization`, `data-to-text`).
  - `RouterBench`: create both an in-domain `train` / `val` / `test` split and an **out-of-domain (OOD)** split holding out designated task categories.
- **Shared Feature Pipeline (`src/eval/features.py`):**
  - **Query-only features (used by Router offline & online):** `all-MiniLM-L6-v2` 384-dim embedding (or PCA/compressed + summary norms), token count, character length, sentence count, question/instruction markers, lexical complexity signals.
  - **Context-response features (used by Risk Scorer offline & online):** retrieval-to-answer cosine similarity (`all-MiniLM-L6-v2`), NLI entailment & contradiction probabilities (lightweight cross-encoder such as `cross-encoder/nli-deberta-v3-small`), named-entity overlap rate (lightweight regex/NER or spaCy-small), and answer-to-context length ratio.
- **Output:** `data/processed/ragtruth_features.parquet` and `data/processed/routerbench_features.parquet`.

### 5.2 Milestone 2a: Bias-Corrected LLM-as-a-Judge Audit (`src/eval/judge_audit.py`, `notebooks/01_judge_bias_audit.ipynb`)
- Inspect `RAGTruth` columns in Phase 1; evaluate automated judge scores against human span-level ground truth across task types using **Cohen's Kappa**, **Spearman correlation**, and confusion matrices.
- If raw `RAGTruth` lacks automated judge scores, evaluate a stratified sample of $\le 300$ paired examples (including length-expanded and order-swapped pairs for **verbosity bias** and **position bias**) using free-tier `Groq`/`Gemini` or low-cost `deepseek-chat` ($<\$0.50$), then fit length/position-controlled regression adjustment.
- Export `artifacts/judge_audit_summary.json` for `GET /v1/benchmarks/summary`.

### 5.3 Milestone 2b: Hallucination Risk & Conformal Guardrail (`src/conformal/`, `notebooks/02_conformal_guardrail.ipynb`)
- **Base Risk Models (`risk_model.py`):** Train `LogisticRegression` and `LightGBM` classifiers on `RAGTruth` train split to predict $P(\text{hallucination} \mid \text{query, context, response})$.
- **Conformal Calibration (`calibrator.py`):** Use `MAPIE` (Split Conformal Prediction / Conformal Risk Control) on the calibration split to compute threshold $\hat{\lambda}(\alpha)$ bounding empirical false-accept rate $\le \alpha$.
- **Evaluation:** Compare Uncalibrated Heuristic Threshold vs. `Logistic + MAPIE` ($\alpha=0.10$) vs. `LightGBM + MAPIE` ($\alpha \in \{0.10, 0.05\}$) on false-accept rate, abstention rate, ECE, and Brier Score.
- **Outputs:** `artifacts/calibrator.pkl` (base model + MAPIE state + threshold lookup) and `artifacts/conformal_sweep.json` (precomputed metrics across $\alpha \in [0.01, 0.30]$ at step `0.01`).

### 5.4 Milestone 3: Tier-Mapped Learned Pareto Router (`src/router/`, `notebooks/03_pareto_router.ipynb`)
- **Tier Mapping:** Partition `RouterBench` candidate models into **Tier 1 (Small/Cheap)** and **Tier 2 (Strong/Frontier)** based on benchmark cost/capability tiers, labeling each query with Tier 1 quality, Tier 2 quality, and cost delta.
- **Training (`trainer.py`):** Train a `LightGBM` (and logistic baseline) classifier using **query-only features** so `router.pkl` transfers directly to live user prompts without needing `RouterBench` model IDs at runtime.
- **Pareto Evaluation (`pareto.py`):** Sweep $\lambda_{\text{cost}} \in [0.00, 1.00]$ (step `0.05`) to compute cost reduction at $\ge 95\%$ frontier quality retention, **AUC-QC**, comparisons against `Always Frontier`, `Always Small`, and `Random Router`, and OOD task-holdout robustness.
- **Outputs:** `artifacts/router.pkl` and `artifacts/pareto_sweep.json` (curve points across $\lambda_{\text{cost}}$).

### 5.5 Milestone 4a: Online FastAPI Gateway (`src/gateway/`)
- **Persistence (`storage.py`, `cache.py`):**
  - `SQLite` (`data/telemetry.db`) table `requests`: `request_id`, `created_at`, `query_text`, `context_text`, `answer_text`, `draft_answer`, `model_used`, `tier`, `escalated_from_tier1`, `provider`, `cache_hit`, `fallback_used`, `estimated_cost_usd`, `alpha_used`, `lambda_cost_used`, `calibrated_risk_score`, `conformal_decision` (`PASS` | `ABSTAIN` | `UNSCORED`), `nli_score`, `cosine_sim`, `entity_overlap`, `latency_ms`.
  - `FAISS` semantic cache (`IndexFlatIP` on normalized `all-MiniLM-L6-v2` embeddings) backed by SQLite payload lookup.
  - **Cache & Override Policy *(Builder Call)*:** `ABSTAIN` responses are never inserted into the FAISS cache. Cached `PASS` / `UNSCORED` entries store `calibrated_risk_score`, `evidence`, `tier`, and `lambda_cost_used`. On a cache hit, if `alpha_override` is supplied on a contextualized entry, the gateway re-evaluates `calibrated_risk_score <= alpha_override` (falling back to escalation/abstention if stricter); if `lambda_cost_override` shifts the predicted routing tier or `bypass_cache: true` is passed, cache lookup is bypassed so repeated Playground tests reflect slider changes.
- **Conformal Routing & Abstention Policy:**
  - **Optional `context` (`UNSCORED`):** When `context` is omitted or empty on `POST /v1/query`, routing still runs via `router.pkl`, conformal scoring is cleanly skipped (`calibrated_risk_score = null`, `evidence = null`, `conformal_decision = "UNSCORED"`), and no fake context is synthesized.
  - **1x Tier 1 → Tier 2 Auto-Escalation on `ABSTAIN`:** When `context` is present and Tier 1's response exceeds $\alpha$, the gateway automatically escalates once to Tier 2 (`DeepSeek`), logging `escalated_from_tier1: true` and combined cost. If Tier 2 passes ($\le \alpha$), Tier 2's answer is returned (`PASS`). If Tier 2 also exceeds $\alpha$, the gateway returns a safe abstention fallback message (`conformal_decision: "ABSTAIN"`) while persisting the unverified `draft_answer` in SQLite for inspection in the Audit Log Inspector and Playground drawer.
- **Unified Async Providers (`providers.py`):**
  - User provisions `DEEPSEEK_API_KEY` (Tier 2 default `deepseek-chat`) plus free-tier `GROQ_API_KEY` / `GEMINI_API_KEY` (Tier 1), with optional lightweight `Ollama` fallback.
  - `httpx.AsyncClient` adapters with automatic fallback on HTTP `429` or timeout (e.g., Tier 1 `Groq -> Gemini -> Ollama`) and `fallback_used: true` logged in telemetry.
- **API Endpoints (`main.py`, `schemas.py`):**
  - `POST /v1/query`: `{query, context?, alpha_override?, lambda_cost_override?, bypass_cache?}` → `{request_id, answer, draft_answer?, model_used, tier, escalated_from_tier1, cache_hit, fallback_used, estimated_cost_usd, calibrated_risk_score, conformal_decision, evidence?, latency_ms}`.
  - `POST /v1/evaluate-risk`: `{query, context, candidate_answer, alpha_override?}` → standalone conformal guardrail scoring (`PASS` | `ABSTAIN`).
  - `GET /v1/telemetry/pareto`: accepts optional `alpha` & `lambda_cost` query params; returns realized cost savings, abstention rate, cache hit rate, latency `p50`/`p95`, and Pareto points.
  - `GET /v1/telemetry/audit`: supports `limit`, `offset`, `decision` (`PASS`|`ABSTAIN`|`UNSCORED`), `tier`, `cache_hit`, `q` search; returns paginated specimen records (including `draft_answer` and `evidence`) for the Audit Log Inspector.
  - `GET /v1/benchmarks/summary`: returns loaded offline JSON summaries (`judge_audit`, `conformal`, `pareto`) with explicit `status: "ready" | "tbd"` per section.
  - `GET /health`: artifact readiness (`calibrator.pkl`, `router.pkl`), embedding/NLI readiness, cache backend status, and configured provider status.

### 5.6 Milestone 4b: SvelteKit Console (`dashboard/`) & Containerization
- **Frontend Spec (per confirmed Design Brief):**
  - SvelteKit + Tailwind CSS, English UI, `Archival Specimen Index` visual system (`#F6F3EC` ground, `#1C2721` 248px left rail, `#FCFBF8` cards, `#236B52` `PASS`, `#B84A39` `ABSTAIN`, `#B46914` `[Synthetic]`/`FALLBACK`, `tabular-nums`, inline specimen evidence strip).
  - 6 views inside persistent shell: `Executive Overview` (landing), `Pareto Frontier Explorer`, `Conformal Risk Control`, `Audit Log Inspector`, `Live Query & Risk Playground`, and `Offline Benchmark Summary`.
  - Dual-mode $(\alpha, \lambda_{\text{cost}})$ simulation bar: updates curves/filters immediately and passes overrides to `/v1/query` & `/v1/evaluate-risk` without altering server defaults.
  - Provenance handling: shows explicit `Unrun / TBD` empty states until backend artifacts/logs exist, with an optional `Preview Sample Data` toggle clearly badged `[Synthetic]`.
- **Packaging (`Dockerfile`, `dashboard/Dockerfile`, `docker-compose.yml`):**
  - Multi-service `docker-compose.yml` running `gateway` (FastAPI on `:8000`, mounting `./artifacts`, `./data`, `./config`) and `dashboard` (SvelteKit Node/Vite on `:5173` or `:3000`) so the full stack works identically with or without host Node.js.

---

## 6. Left to the Builder

- Manage Python dependencies and virtual environment with **`uv`** (`pyproject.toml` + `uv.lock`, `uv add`, `uv run`) in Milestone 0 (`chore/project-setup`, installing `uv` on the host first) and reuse `uv` inside the gateway `Dockerfile`, pinning Python 3.12-compatible versions of `mapie`, `lightgbm`, `sentence-transformers`, `faiss-cpu`, `fastapi`, `httpx`, `datasets`, `pyarrow`, `pytest`, etc.
- Inspect actual `RAGTruth` and `RouterBench` raw schemas during Milestone 1 and implement normalization adapters + exact NLI/NER package choices.
- Select the SvelteKit charting implementation (e.g., LayerChart, Apache ECharts, or reactive SVG components) matching the crisp hairline geometry in [assigned.html](../../../.impeccable/mocks/decision/assigned.html).
- Define exact CORS origins, `.env.example` variable names (`DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_BASE_URL`), and unit/integration test mocks for `429` provider fallback.
- Before writing UI code in `dashboard/`, record the Impeccable surface direction contract (`.impeccable/surfaces/`) and run the Impeccable design checks (`detect`, `audit`, `DESIGN.md` documentation) at finish.
