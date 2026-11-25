# Phase 4: FastAPI Gateway, SvelteKit Console & Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI inference & conformal guardrail gateway (`src/gateway/`), the SvelteKit + Tailwind monitoring console (`dashboard/`), Docker Compose packaging, and populate verified benchmark tables in `README.md`.

**Architecture:** Branch `feat/inference-gateway` implements Pydantic contracts, SQLite telemetry, FAISS semantic cache, DeepSeek/Groq/Gemini/Ollama async client with 1x Tier 1->Tier 2 escalation on `ABSTAIN`, and all 6 FastAPI endpoints. Branch `feat/monitoring-dashboard` builds the 6-view SvelteKit console (`Archival Specimen Index`, `seed:26a4da6343180544`, `code-led`). Branch `chore/docker-and-benchmarks` finalizes Docker Compose, Impeccable finish checks, and `README.md` result tables.

**Tech Stack:** FastAPI, Pydantic v2, `httpx`, `SQLite`, `FAISS`, SvelteKit, Tailwind CSS, Docker Compose, `uv`, `pytest`

**Spec:** [2026-11-25-conformal-llm-router-design.md](../specs/2026-11-25-conformal-llm-router-design.md) · Frontend Sketch: [assigned.html](../../../.impeccable/mocks/decision/assigned.html)

## Global Constraints

- **Optional `context` (`UNSCORED`):** If `context` is omitted/empty on `POST /v1/query`, skip conformal scoring (`calibrated_risk_score = null`, `evidence = null`, `conformal_decision = "UNSCORED"`).
- **1x Tier 1 -> Tier 2 Escalation on `ABSTAIN`:** When `context` is present and Tier 1 exceeds `alpha`, escalate once to Tier 2 (`DeepSeek`). If Tier 2 also exceeds `alpha`, return safe fallback text (`conformal_decision = "ABSTAIN"`) and persist `draft_answer` in SQLite.
- **Cache & Override Rules:** Never cache `ABSTAIN` responses in FAISS; re-check `alpha_override` on contextualized cache hits; bypass cache when `lambda_cost_override` shifts the tier or `bypass_cache=true`.
- **UI Honesty & Mode:** `Operate` mode, English UI, `Archival Specimen Index` (`#F6F3EC`, `#1C2721`, `#FCFBF8`, `#236B52`, `#B84A39`, `#B46914`), explicit `Unrun / TBD` empty states, and `[Synthetic]` badge on sample preview mode.
- Load Impeccable `reference/craft-floor.md` immediately before any UI code edits, record `.impeccable/surfaces/` direction contract, and run `impeccable detect` + `document` (`DESIGN.md`) before closing `feat/monitoring-dashboard`.

## Review Focus

- Calling `POST /v1/query` when all upstream Tier 1 providers hit `429` or timeout must execute the fallback chain (`Groq -> Gemini -> Ollama`) and return a clean `503` JSON error (never an unhandled 500 traceback) if every provider fails.
- Re-running the identical prompt with a stricter `alpha_override` (e.g. `0.02` after `0.20`) must NOT blindly return the cached `PASS` if the cached risk exceeds `0.02`.
- `POST /v1/evaluate-risk` with empty `candidate_answer` or `context` must validate via Pydantic (`422`) or return finite safe scores without crashing.
- `GET /v1/benchmarks/summary` and `GET /v1/telemetry/audit` when `artifacts/*.json` are missing or `requests` table has `0` rows must return valid `200` JSON with `status: "tbd"` / `items: []`.
- SvelteKit UI when the FastAPI gateway is unreachable (`503`/network error) must render a clear degraded/offline banner rather than blank charts or uncaught promise rejections.

---

### Task 1: Pydantic Schemas, SQLite Telemetry & FAISS Cache (`feat/inference-gateway`)

**Files:**
- Create: `src/gateway/schemas.py`, `src/gateway/storage.py`, `src/gateway/cache.py`
- Test: `tests/test_gateway_storage.py`

**Interfaces:**
- Consumes: `numpy`, `faiss`, `sqlite3`, `pydantic`
- Produces:
  - `QueryRequest`, `QueryResponse`, `EvaluateRiskRequest`, `EvaluateRiskResponse`, `EvidenceBreakdown`, `ParetoTelemetryResponse`, `AuditListResponse`, `BenchmarksSummaryResponse`, `HealthResponse`
  - `TelemetryStore(db_path: Path)` (`init_schema()`, `insert_request(record: dict)`, `get_request(request_id: str)`, `query_audit(...)`, `aggregate_pareto(alpha: float, lambda_cost: float)`)
  - `SemanticCache(index_path: Path, store: TelemetryStore, similarity_threshold: float = 0.92)` (`lookup(embedding: np.ndarray) -> dict | None`, `put(embedding: np.ndarray, record: dict) -> None`)

- [ ] **Step 1: Create `feat/inference-gateway` branch and write failing storage/cache tests**

Run: `git checkout -b feat/inference-gateway`
Test: `ABSTAIN` entries are ignored by `SemanticCache.put`, `PASS` entries hit above threshold `0.92`, `TelemetryStore.query_audit` filters by `decision` (`PASS`/`ABSTAIN`/`UNSCORED`), and `aggregate_pareto` handles empty DB cleanly.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_gateway_storage.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `schemas.py`, `storage.py`, and `cache.py`**

- [ ] **Step 4: Run tests and verify pass**

Run: `uv run pytest tests/test_gateway_storage.py -v`
Expected: PASS.

---

### Task 2: Unified Async Providers & FastAPI Endpoints (`feat/inference-gateway`)

**Files:**
- Create: `src/gateway/providers.py`, `src/gateway/main.py`
- Test: `tests/test_gateway.py`

**Interfaces:**
- Consumes: `src/eval/features.py`, `src/conformal/calibrator.py`, `src/router/trainer.py`, `src/gateway/{schemas,storage,cache}.py`
- Produces:
  - `ProviderRouter` (`generate(tier: Literal["tier1", "tier2"], query: str, context: str | None) -> ProviderResult`)
  - `create_app(...) -> FastAPI` exposing `/v1/query`, `/v1/evaluate-risk`, `/v1/telemetry/pareto`, `/v1/telemetry/audit`, `/v1/benchmarks/summary`, `/health`

- [ ] **Step 1: Write failing API & provider fallback tests in `tests/test_gateway.py`**

Test using FastAPI `TestClient` + mocked `httpx` transport + temporary SQLite/FAISS:
1. `POST /v1/query` without `context` returns `conformal_decision == "UNSCORED"` and `calibrated_risk_score is None`.
2. `POST /v1/query` with `context` when Tier 1 exceeds `alpha` escalates 1x to Tier 2 (`escalated_from_tier1 == True`); if Tier 2 also exceeds `alpha`, returns safe abstention text while persisting `draft_answer`.
3. Cache hit with stricter `alpha_override` re-thresholds cached risk without returning stale `PASS`; `lambda_cost_override` shifting tier bypasses cache.
4. `429` from Groq falls back to Gemini/Ollama (`fallback_used == True`); total upstream failure returns clean `503`.
5. `/v1/evaluate-risk`, `/v1/telemetry/pareto`, `/v1/telemetry/audit`, `/v1/benchmarks/summary`, `/health` return valid typed contracts on empty & populated states.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_gateway.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/gateway/providers.py` and `src/gateway/main.py`**

- [ ] **Step 4: Run full backend test suite, commit, and merge `feat/inference-gateway` to `main`**

Run: `uv run pytest tests/ -v && uv run ruff check src/ tests/`
Expected: All tests PASS.

---

### Task 3: SvelteKit Console (`feat/monitoring-dashboard`)

**Files:**
- Create: `.impeccable/surfaces/dashboard.md` (direction contract before UI code)
- Create: `dashboard/package.json`, `dashboard/Dockerfile`, `dashboard/src/...` (6 routes, shared `SpecimenShell`, `ControlBar`, `EvidenceStrip`, `[Synthetic]` dataset fixtures, `api.ts` typed client)
- Create: `DESIGN.md` (at finish via Impeccable document flow)

**Interfaces:**
- Consumes: FastAPI endpoints (`/v1/query`, `/v1/evaluate-risk`, `/v1/telemetry/pareto`, `/v1/telemetry/audit`, `/v1/benchmarks/summary`, `/health`)
- Produces: Runnable SvelteKit + Tailwind app matching [assigned.html](../../../.impeccable/mocks/decision/assigned.html)

- [ ] **Step 1: Pre-UI Impeccable gates**

Create `feat/monitoring-dashboard` branch, read `/home/fahmi/.gemini/config/skills/impeccable/reference/craft-floor.md` and `new-work.md` section 5, write direction contract via `impeccable surface-brief write`, and verify `seed:26a4da6343180544` + `FINISH` line.

- [ ] **Step 2: Scaffold SvelteKit + Tailwind CSS (via `docker run --rm -v "$PWD/dashboard:/app" -w /app node:20-alpine ...` if host lacks `node`) and build shared shell + types**

Implement persistent `248px` moss-charcoal rail (`#1C2721`), `#F6F3EC` archival ground, dual-mode $(\alpha, \lambda_{\text{cost}})$ simulation bar, `Unrun / TBD` empty states, `[Synthetic]` toggle, and offline/503 banner.

- [ ] **Step 3: Implement the 6 workspaces + `EvidenceStrip` component**

Build routes: `/` (`Executive Overview`), `/pareto` (`Pareto Frontier Explorer`), `/conformal` (`Conformal Risk Control`), `/audit` (`Audit Log Inspector`), `/playground` (`Live Query & Risk Playground`), and `/benchmarks` (`Offline Benchmark Summary`).

- [ ] **Step 4: Build check, Impeccable mechanical detector, visual verification, and `DESIGN.md`**

Run SvelteKit build/check inside Docker Node container, run `/home/fahmi/.gemini/config/skills/impeccable/scripts/impeccable detect --json dashboard/src`, verify rendered desktop/mobile viewports, write `DESIGN.md`, and merge `feat/monitoring-dashboard` to `main`.

---

### Task 4: Docker Compose, End-to-End Verification & Benchmark Results (`chore/docker-and-benchmarks`)

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`
- Modify: `README.md` (replace `TBD` rows in Target Experiment Results with actual measured Phase 2–3 outputs)

- [ ] **Step 1: Create `chore/docker-and-benchmarks` branch and write `Dockerfile` (using `uv`) + `docker-compose.yml`**
- [ ] **Step 2: Build and smoke-test full stack (`docker compose up --build -d` -> curl `/health`, `/v1/benchmarks/summary`, and dashboard root)**
- [ ] **Step 3: Populate verified experiment tables in `README.md`, run final `uv run pytest`, commit, and merge to `main`**
