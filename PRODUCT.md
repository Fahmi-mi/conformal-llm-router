# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

- **Frontend (`dashboard/`):** SvelteKit + Tailwind CSS, English UI, running locally via Docker or Node.js (Node/npm not yet installed on host; Docker is available).
- **Backend & ML (`src/`):** FastAPI (`Python 3.12`), managed with **`uv`** (`pyproject.toml` / `uv.lock`, `uv add`, `uv run`), Pydantic, SQLite + FAISS (semantic cache & telemetry), `MAPIE`, `LightGBM`, `scikit-learn`, `sentence-transformers`.
- **Execution Mode:** Milestone-by-milestone modular pipeline producing versioned offline artifacts (`calibrator.pkl`, `router.pkl`) consumed directly by the FastAPI gateway.

## Users

1. **Project Author (Primary Builder & Learner):** Uses notebooks, telemetry, and the web console to study, validate, and audit conformal prediction, judge-bias correction, and Pareto routing behavior.
2. **Portfolio Reviewers & Technical Evaluators (Recruiters / AI Engineers):** Inspect live system capabilities, interactive risk/cost trade-offs, and verified offline benchmark evidence as an engineering portfolio demonstration.

## Product Purpose

Deliver a production-oriented **Statistical AI Control & LLMOps** platform that solves two core LLM/RAG bottlenecks: uncalibrated hallucination risk and unnecessary frontier-model inference cost. Success means every milestone in [README.md](README.md) executes reproducibly end-to-end—from offline statistical calibration to live API routing and frontend inspection—while clearly demonstrating the underlying data science and engineering concepts.

## Positioning

Unlike standard RAG wrappers or heuristic cosine-threshold routers, this platform treats LLM orchestration as a measurable statistical risk-control and cost-optimization problem: auditing automated judges against human labels (`RAGTruth`), bounding empirical hallucination pass-through via finite-sample Split Conformal Prediction (`MAPIE`), and routing queries via a learned cost-quality Pareto classifier (`RouterBench`) with graceful free-tier/paid-credit provider fallback.

## Operating Context

Two connected workflows operate across the system:
1. **Offline Eval & Calibration Pipeline:** Downloads `RAGTruth` (~18k responses, MIT) and `RouterBench` (>405k outcomes), extracts CPU-friendly features into Parquet, audits LLM-as-a-Judge bias, calibrates `MAPIE` risk guardrails, and trains a **Tier-Mapped Learned Router** (grouping `RouterBench` models into Tier 1 small/cheap vs. Tier 2 strong/frontier using query-only features).
2. **Online Inference Gateway & Web Console:** `FastAPI` serves `POST /v1/query`, `POST /v1/evaluate-risk`, `GET /v1/telemetry/pareto`, and `GET /health`. The SvelteKit console exposes five functional areas:
   - Pareto Frontier Explorer ($\lambda_{\text{cost}}$ cost-quality trade-off)
   - Conformal Reliability & Risk Control View ($\alpha$ error-bound slider)
   - Audit Log Inspector (flagged/abstained responses & context overlap)
   - Live Query & Risk Playground (interactive testing against `/v1/query` and `/v1/evaluate-risk`)
   - Offline Benchmark Summary (`RAGTruth`, `RouterBench`, and Judge Audit results)

## Capabilities and Constraints

- **Provider Strategy:** Live gateway uses **DeepSeek API** credits (`deepseek-chat`) as Tier 2 (and optional automated judge) paired with free-tier APIs (**Groq** / **Gemini Flash-Lite**) as Tier 1, plus optional lightweight local **Ollama** fallback and automatic `429` rate-limit fallback.
- **Hardware Constraint:** Host machine has an NVIDIA RTX 3050 6GB Laptop GPU and 273 GB free disk space; heavy local LLMs (e.g., 70B or large multi-model Ollama stacks) must not be required for live serving.
- **Persistence & Auth:** Uses local SQLite and FAISS rather than external PostgreSQL; public internet deployment and multi-user authentication are currently out of scope for the initial local Docker target.
- **Open / Undecided Facts:** Exact Tier 1 default provider precedence (`Groq` vs. `Gemini Flash-Lite`) and whether Judge Audit uses pre-existing `RAGTruth` annotations alone or live DeepSeek judge sampling on a subset will be settled during Phase 1–2 data inspection.

## Brand Commitments

- **Working Title:** Conformal Hallucination Control & Learned LLM Cost-Quality Router.
- **Language:** English across all UI copy, telemetry labels, and portfolio documentation.
- **Assets:** No existing logo or visual assets; no custom brand palette or typography committed yet.

## Evidence on Hand

- [README.md](README.md): Complete project specification, proposed repository layout, architecture diagram, API endpoints, and milestone plan.
- **Missing / Unrun Evidence:** Offline datasets (`RAGTruth`, `RouterBench`), trained model artifacts (`calibrator.pkl`, `router.pkl`), and all experiment metrics (currently marked `TBD` in `README.md`) have not been generated yet and must not be fabricated in any UI or documentation.

## Product Principles

1. **Mathematical Guarantees Over Ad-Hoc Thresholds:** Every guardrail and routing decision exposes its calibrated confidence, risk bound ($\alpha$), and cost-quality trade-off ($\lambda_{\text{cost}}$).
2. **Offline-to-Online Feature Parity:** Features engineered in offline notebooks (`all-MiniLM-L6-v2`, overlap/NLI signals) map identically into the live FastAPI pipeline without training-serving skew.
3. **Cost-Resilient Execution:** Offline calibration costs zero paid tokens; live inference uses DeepSeek credits efficiently alongside free-tier routing and semantic caching.
4. **Inspectable Learning Surface:** Both intermediate statistical evidence (judge bias, calibration curves, Pareto frontiers) and live request telemetry remain transparent and inspectable.
5. **Strict Evidence Honesty:** Benchmark tables and UI summaries clearly distinguish verified experimental runs from unrun (`TBD`) states.
