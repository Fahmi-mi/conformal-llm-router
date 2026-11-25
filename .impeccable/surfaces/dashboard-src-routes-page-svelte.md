---
version: 1
slug: "dashboard-src-routes-page-svelte"
primary_target: "dashboard/src/routes/+page.svelte"
related_targets: ["dashboard/src/routes/+layout.svelte"]
---

# Surface Brief: Conformal LLM Router Monitoring Console

- **Primary target:** `dashboard/src/routes/+page.svelte`
- **Related targets:** `dashboard/src/routes/+layout.svelte`, `dashboard/src/routes/pareto/+page.svelte`, `dashboard/src/routes/conformal/+page.svelte`, `dashboard/src/routes/audit/+page.svelte`, `dashboard/src/routes/playground/+page.svelte`, `dashboard/src/routes/benchmarks/+page.svelte`
- **Visitor mode:** `Operate`
- **Audience & job:** ML/AI engineers, technical portfolio evaluators, and the project author validating calibrated conformal hallucination risk bounds, learned Pareto cost-quality routing, semantic cache efficiency, live inference behavior, and offline benchmark provenance.
- **Routes (6 workspaces):**
  1. `/` — `Executive Overview` (gateway health, 4 telemetry KPIs, Pareto & Conformal summary curves, recent specimen audit log)
  2. `/pareto` — `Pareto Frontier Explorer` (interactive cost-quality SVG curve with In-Domain vs. OOD `mbpp` toggle, active `lambda_cost` operating point, `AUC-QC`, `cost_savings@95%`, baseline comparison table, live tier split)
  3. `/conformal` — `Conformal Risk Control` (interactive SVG curve across `alpha` in `[0.01, 0.30]`, active `alpha` cutoff rule, `y = alpha` bound line, ECE, Brier Score, Logistic vs. LightGBM vs. Heuristic comparison)
  4. `/audit` — `Audit Log Inspector` (filterable by `PASS`/`ABSTAIN`/`UNSCORED`, tier, `cache_hit`, and search query; locked tabular column widths; expandable specimen detail inspector for `query_text`, `context_text`, `answer_text`, `draft_answer`, and `EvidenceStrip`)
  5. `/playground` — `Live Query & Risk Playground` (interactive `POST /v1/query` with optional `context` and `bypass_cache`, plus `POST /v1/evaluate-risk`, with toggle to pass current `(alpha, lambda_cost)` overrides; shows decision badge, tier/escalation/fallback pills, cost, latency, `draft_answer` on abstention, and `EvidenceStrip`)
  6. `/benchmarks` — `Offline Benchmark Summary` (renders `README.md` benchmark tables from `/v1/benchmarks/summary` with explicit `Ready` vs. `TBD` provenance for Judge Audit, Conformal Calibration, and Cost-Quality Routing)
- **Dual-mode simulation controls:** Persistent header control bar adjusting target risk `alpha` (`0.01..0.30`) and cost weight `lambda_cost` (`0.00..1.00`) for live telemetry filtering and offline curve simulation without mutating FastAPI server defaults unless explicitly opted into in `/playground`.
- **Provenance & honest states:** Missing or unrun offline metrics display `TBD` (`[TBD]`) with their reported reason; `Preview Sample Data [Synthetic]` is an explicit global toggle labeled with `[Synthetic]` pills whenever active; gateway offline or `503` conditions surface an actionable degraded banner; loading states render structured skeletons.

## Direction contract

- **THESIS:** Archival Specimen Index — an analytical LLMOps workbench where every routing decision, calibration sweep, and conformal veto reads like a verifiable laboratory specimen, refusing generic slate-and-blue SaaS dashboards and dark neon observability clones.
- **OWN-WORLD:** Warm archival paper ground (`#F6F3EC`), deep moss-charcoal `248px` left rail (`#1C2721`), crisp specimen surfaces (`#FCFBF8`) bound by `1px solid #DCD6C8` hairlines, tabular-numeral UI sans-serif paired with monospace specimen IDs and calibrated thresholds, `#236B52` for `PASS`, `#B84A39` for `ABSTAIN`, `#B46914` for `[Synthetic]` and `FALLBACK`, `#5C6860` muted metadata ink, and the shared `EvidenceStrip` component quantifying NLI entailment, contradiction, cosine similarity, entity overlap, and conformal risk cutoff.
- **STORY:** A reviewer enters on the Executive Overview, immediately sees gateway readiness, provenance status, empirical false-accept rate against the active `alpha` bound, realized cost savings, and the dual Pareto/Conformal charts; they explore the cost-quality frontier and conformal guarantee curves, inspect individual specimens and abstained `draft_answer` records in the audit log, test custom prompts in the playground, and verify offline `README.md` benchmark tables with honest `Ready`/`TBD` labeling.
- **FIRST VIEWPORT:** Fixed `248px` `#1C2721` navigation rail on desktop (collapsing to a responsive top bar + drawer below `768px`) listing the 6 workspaces and live gateway provider/health readout; right workspace on `#F6F3EC` displaying the page header with provenance badge, dual-mode `(alpha, lambda_cost)` simulation bar, 4-column KPI strip, side-by-side Pareto and Conformal SVG plots, and the recent specimen audit table with locked column widths and inline `EvidenceStrip`.
- **FORM:** Archival Specimen Index (ordered list position 1, assigned direction), `seed:26a4da6343180544`.
- **FINISH:** "unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance"
