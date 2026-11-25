---
name: Conformal LLM Router Console
description: Archival Specimen Index design system for conformal hallucination control and cost-quality Pareto routing.
colors:
  archival-ground: "#f6f3ec"
  specimen-surface: "#fcfbf8"
  inset-stone: "#efece3"
  moss-rail: "#1c2721"
  moss-rail-active: "#2a3a31"
  specimen-ink: "#1c2721"
  muted-slate: "#5c6860"
  hairline-rule: "#dcd6c8"
  conformal-pass: "#236b52"
  conformal-pass-wash: "#e4f2ec"
  conformal-pass-ink: "#1b5642"
  conformal-abstain: "#b84a39"
  conformal-abstain-wash: "#fbe9e7"
  conformal-abstain-ink: "#9e3829"
  synthetic-amber: "#b46914"
  synthetic-wash: "#fdf6e9"
  synthetic-ink: "#8f520e"
  coral-indicator: "#d97757"
typography:
  headline:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "-0.02em"
  title:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 700
    lineHeight: 1.35
    letterSpacing: "0em"
  body:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0em"
  label:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    fontSize: "0.6875rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.06em"
  mono-data:
    fontFamily: "ui-monospace, 'SFMono-Regular', Menlo, Monaco, Consolas, monospace"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "0em"
rounded:
  xs: "1px"
  sm: "2px"
  md: "3px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.moss-rail}"
    textColor: "{colors.specimen-surface}"
    rounded: "{rounded.sm}"
    padding: "8px 16px"
  button-primary-hover:
    backgroundColor: "{colors.moss-rail-active}"
  button-secondary:
    backgroundColor: "{colors.specimen-surface}"
    textColor: "{colors.specimen-ink}"
    rounded: "{rounded.sm}"
    padding: "6px 12px"
  button-synthetic:
    backgroundColor: "{colors.synthetic-wash}"
    textColor: "{colors.synthetic-ink}"
    rounded: "{rounded.sm}"
    padding: "6px 12px"
  badge-pass:
    backgroundColor: "{colors.conformal-pass-wash}"
    textColor: "{colors.conformal-pass-ink}"
    rounded: "{rounded.sm}"
    padding: "2px 8px"
  badge-abstain:
    backgroundColor: "{colors.conformal-abstain-wash}"
    textColor: "{colors.conformal-abstain-ink}"
    rounded: "{rounded.sm}"
    padding: "2px 8px"
  specimen-card:
    backgroundColor: "{colors.specimen-surface}"
    textColor: "{colors.specimen-ink}"
    rounded: "{rounded.md}"
    padding: "16px"
  nav-item-active:
    backgroundColor: "{colors.moss-rail-active}"
    textColor: "{colors.specimen-surface}"
    rounded: "{rounded.md}"
    padding: "10px 12px"
---

# Design System: Conformal LLM Router Console

## Overview

**Creative North Star: "Archival Specimen Index"**

The Conformal LLM Router console treats every inference trace, conformal calibration curve, and cost-quality trade-off like an inspectable laboratory specimen. Instead of dark neon observability chrome or generic slate-and-blue SaaS boxes, the workspace pairs a deep moss-charcoal navigation ledger with warm archival rag paper surfaces, crisp hairline boundaries, and tabular numerical precision.

Operating clarity drives every surface decision. Sliders for target risk `alpha` and cost weight `lambda_cost` sit immediately above the data view without mutating server defaults, empirical metrics remain visibly bound to their provenance (`Ready`, `TBD`, or `[Synthetic]`), and every verification claim exposes the underlying NLI, cosine, entity, and threshold signals inside the `EvidenceStrip`.

**Key Characteristics:**
- High-contrast split shell pairing a persistent `248px` deep moss rail with a warm archival rag workbench.
- Flat, hairline-ruled specimen containers (`1px` stone borders, `2px`–`3px` corner radii) free of ambient drop shadows.
- Strict evidence provenance distinguishing live SQLite telemetry, offline benchmark artifacts, `[TBD]` unrun stages, and `[Synthetic]` sample previews.
- Tabular-numeral UI typography paired with monospace strictly for request identifiers, model strings, and calibrated statistical measurements.

## Colors

The palette contrasts grounded forest-green and vermilion decision signals against warm paper grounds and deep moss navigation chrome.

### Primary
- **Conformal Pass** (`conformal-pass` / `conformal-pass-wash` / `conformal-pass-ink`): Marks `PASS` decisions, satisfied conformal error bounds, Pareto frontier curves, and active primary slider tracks.
- **Moss Rail** (`moss-rail` / `moss-rail-active`): Grounds the persistent left navigation shell, primary execution buttons, and `Tier 2` frontier badges.

### Secondary
- **Conformal Abstain** (`conformal-abstain` / `conformal-abstain-wash` / `conformal-abstain-ink`): Signals `ABSTAIN` vetoes, active `alpha` cutoff lines, `ESCALATED` alerts, and unreachable gateway errors, using deepened ink (`conformal-abstain-ink`) inside tinted wash pills to preserve ≥4.5:1 AA contrast.
- **Synthetic Amber** (`synthetic-amber` / `synthetic-wash` / `synthetic-ink`): Reserved exclusively for `[Synthetic]` sample preview mode, `[TBD]` unrun benchmark notices, and provider `FALLBACK` pills.
- **Coral Indicator** (`coral-indicator`): Used solely as a `6px` circular indicator dot inside the active workspace navigation row.

### Neutral
- **Archival Ground** (`archival-ground`): Base canvas across all six workspace routes.
- **Specimen Surface** (`specimen-surface`): Elevated analytical panels, KPI tiles, and tables.
- **Inset Stone** (`inset-stone`): Table hover washes, empty-state bands, and uncalibrated metric strips.
- **Hairline Rule** (`hairline-rule`): `1px` panel and table structural dividers.
- **Specimen Ink & Muted Slate** (`specimen-ink`, `muted-slate`): High-contrast primary copy and secondary axis/column metadata.

### Named Rules
**The Provenance Color Rule.** Amber (`synthetic-amber`) is reserved strictly for `[Synthetic]`, `[TBD]`, and `FALLBACK` states so reviewers never mistake fixture previews or degraded runs for verified production metrics.
**The AA Wash Contrast Rule.** Semantic pill text over tinted status washes must achieve at least 4.5:1 contrast (`conformal-abstain-ink` on `conformal-abstain-wash`, `conformal-pass-ink` on `conformal-pass-wash`).

## Typography

**Display Font:** System UI Sans (`system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`)
**Body Font:** System UI Sans (`system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`)
**Label/Mono Font:** System Monospace (`ui-monospace, 'SFMono-Regular', Menlo, Monaco, Consolas, monospace`)

**Character:** Workhorse analytical sans-serif tuned for rapid Operate-mode scanning with enforced `font-variant-numeric: tabular-nums`, reserving monospace strictly for specimen IDs, model slugs, and statistical figures.

### Hierarchy
- **Headline** (`700`, `1.5rem` / `24px`, `1.25` line-height, `-0.02em` tracking): Page titles (`<h1>`) anchoring every workspace route.
- **Title** (`700`, `0.9375rem` / `15px`, `1.35` line-height): Panel and table section headings (`<h2>`).
- **Body** (`400`, `0.8125rem`–`0.875rem` / `13px`–`14px`, `1.5` line-height, `65ch`–`75ch` max measure for prose): Table cells, descriptions, and query/context text blocks.
- **Label** (`600`, `0.6875rem` / `11px`, `1.2` line-height, `0.06em` tracking, uppercase): Table column headers, KPI captions, and form field labels.
- **Mono Data** (`500`, `0.75rem` / `12px`, `1.4` line-height): Request UUIDs, `#SPEC-*` tags, calibrated risk scores, thresholds, ECE/Brier figures, and route indices.

### Named Rules
**The Tabular Measurement Rule.** Every numeric KPI, table column, slider readout, and SVG axis tick enforces `tabular-nums` so interactive `(alpha, lambda_cost)` scrubbing never induces layout jitter.
**The No-Kicker Rule.** Page and section headings speak directly without uppercase eyebrow labels placed above them.

## Layout

Desktop viewports (`>= 768px`) use a two-column CSS Grid (`248px` sticky full-height moss rail + `1fr` scrollable workbench with `min-w-0`). Below `768px`, the rail collapses into a sticky top header with an expandable navigation drawer. Workspace panels stack with a `16px` (`gap-4`) vertical rhythm, bounded horizontally by clean `overflow-x-auto` table wrappers with locked `table-layout: fixed` column widths on specimen ledgers.

## Elevation & Depth

Surfaces are flat and structurally separated solely by `1px` hairline borders (`hairline-rule`) and tonal contrast between `archival-ground`, `specimen-surface`, and `inset-stone`. Zero drop shadows or blur layers are applied.

### Named Rules
**The Flat Hairline Rule.** Depth comes from surface tint and `1px` stone rules, never ambient shadows or `>1px` accent borders on cards or alerts.

## Shapes

All cards, panels, buttons, inputs, and status badges share crisp architectural geometry with small radii (`2px` for controls and badges, `3px` for specimen panels and navigation items). Charts render crisp SVG polylines and dashed reference thresholds with circular `3px`–`6px` operating markers.

## Components

### Buttons
- **Shape:** Tight rectangular (`2px` radius).
- **Primary:** Solid `moss-rail` fill with `specimen-surface` text (`8px 16px` padding), shifting to `moss-rail-active` on hover.
- **Secondary:** `specimen-surface` fill with `1px` `hairline-rule` stroke and `specimen-ink` text (`6px 12px` padding), shifting to `inset-stone` on hover.
- **Synthetic Trigger:** `synthetic-wash` fill with `1px` `synthetic-amber` border and `synthetic-ink` text.

### Chips
- **Style:** `2px` rounded uppercase tags (`11px`, `600` weight, `1px` border) with semantic wash/ink pairings for `PASS`, `ABSTAIN`, `UNSCORED`, `SYNTHETIC`, `FALLBACK`, `ESCALATED`, `CACHE`, `READY`, and `TBD`.

### Cards / Containers
- **Corner Style:** `3px` radius (`rounded-[3px]`).
- **Background:** `specimen-surface` over `archival-ground`.
- **Shadow Strategy:** None (`box-shadow: none`).
- **Border:** `1px solid` `hairline-rule`.
- **Internal Padding:** `16px` (`p-4`) standard; `12px 16px` for dense filter/control bars.

### Inputs / Fields
- **Style:** `archival-ground` fill with `1px` `hairline-rule` border (`2px` radius), `specimen-ink` text, and `muted-slate` placeholder (`>= 4.5:1` contrast).
- **Focus:** `2px solid` `conformal-pass` outline with `2px` offset (`:focus-visible`).
- **Sliders:** `6px` track (`#e3ddd0`) with `14px` rectangular thumb transitioning from `specimen-ink` to `conformal-pass` on hover.

### Navigation
- **Style:** `248px` `moss-rail` sidebar on desktop (`>= 768px`) with 6 route links (`Executive Overview`, `Pareto Frontier`, `Conformal Risk`, `Audit Log Inspector`, `Live Playground`, `Offline Benchmarks`), consistent `15px` stroke-`1.75` inline SVG icons, and `01`–`06` catalog indices. Active links receive `moss-rail-active` fill and a `6px` `coral-indicator` dot.

### EvidenceStrip
Signature verification component rendering grounding evidence in two densities:
- **Compact mode:** Four `6px x 14px` vertical fill meters (`nli_entailment`, `nli_contradiction`, `cosine_sim`, `entity_overlap`) paired with compact monospace `E:` / `C:` readouts for table cells.
- **Detailed mode:** A hairline-divided 4-column metric grid paired with a conformal risk-vs-cutoff horizontal progress gauge marking `calibrated_risk_score` against the vertical `threshold_used` cutoff bar.

## Do's and Don'ts

### Do:
- **Do** keep `alpha` (`0.01`–`0.30`) and `lambda_cost` (`0.00`–`1.00`) controls strictly synchronized with SVG chart operating markers and API query parameters.
- **Do** mark every unrun offline artifact metric explicitly as `TBD` with its backend reason string, and attach `[Synthetic]` badges whenever sample fixtures are enabled.
- **Do** preserve `1px` hairline borders and flat tonal separation across all panels, empty states, and inspector panes.
- **Do** theme native browser surfaces (`::selection`, `caret-color`, `:focus-visible`, scrollbars, range inputs, and `tabular-nums`) from the archival palette.

### Don't:
- **Don't** place uppercase kickers or eyebrows above headings, or render nested bordered cards inside `.specimen-panel` containers.
- **Don't** use `>1px` colored left/right border stripes on navigation links, callouts, cards, or alerts.
- **Don't** substitute Unicode glyphs or emojis for the stroke-`1.75` SVG icon set, or use body/UI text smaller than `12px` (`11px` floor strictly for compact pills/labels).
- **Don't** fabricate live telemetry or hide `Gateway Unreachable` / `503` errors when FastAPI is offline or missing artifacts.
