<script lang="ts">
	import { findNearestConformalPoint, formatFixed, formatPct } from '#lib/analytics';
	import ConformalChart from '#lib/components/ConformalChart.svelte';
	import DecisionBadge from '#lib/components/DecisionBadge.svelte';
	import { consoleState } from '#lib/state.svelte';
	import type { RiskMetricEntry } from '#lib/types';

	let showLogistic = $state(true);
	let showAbstention = $state(true);

	const conformal = $derived(consoleState.benchmarks?.conformal);
	const riskMetrics = $derived(consoleState.benchmarks?.risk_metrics);
	const sweep = $derived(conformal?.sweep ?? []);
	const logisticSweep = $derived(conformal?.logistic_sweep ?? []);

	const activeLgbm = $derived(findNearestConformalPoint(sweep, consoleState.alpha));
	const activeLogistic = $derived(findNearestConformalPoint(logisticSweep, consoleState.alpha));

	const boundSatisfied = $derived(
		activeLgbm ? activeLgbm.false_accept_rate <= consoleState.alpha + 1e-6 : true
	);

	const comparisonRows = $derived.by(() => {
		if (!riskMetrics) return [] as RiskMetricEntry[];
		const rows: RiskMetricEntry[] = [];
		if (riskMetrics.uncalibrated_heuristic) rows.push(riskMetrics.uncalibrated_heuristic);
		if (riskMetrics.logistic_mapie?.['0.10']) rows.push(riskMetrics.logistic_mapie['0.10']);
		if (riskMetrics.logistic_mapie?.['0.05']) rows.push(riskMetrics.logistic_mapie['0.05']);
		if (riskMetrics.lightgbm_mapie?.['0.10']) rows.push(riskMetrics.lightgbm_mapie['0.10']);
		if (riskMetrics.lightgbm_mapie?.['0.05']) rows.push(riskMetrics.lightgbm_mapie['0.05']);
		return rows;
	});
</script>

<svelte:head>
	<title>Conformal Risk Control — Conformal LLM Router</title>
</svelte:head>

<div class="space-y-4">
	<div class="flex flex-wrap items-center justify-between gap-3 border-b border-[#DCD6C8] pb-3">
		<div>
			<h1 class="text-2xl font-bold tracking-[-0.02em] text-[#1C2721]">
				Conformal Risk Control (RAGTruth)
			</h1>
			<p class="text-xs text-[#5C6860] mt-0.5 max-w-[70ch]">
				Split conformal calibration (MAPIE LAC) across α ∈ [0.01, 0.30] on {conformal?.n_test?.toLocaleString() ?? '2,700'} held-out test specimens ({conformal?.n_cal?.toLocaleString() ?? '3,018'} calibration split).
			</p>
		</div>

		<div class="flex flex-wrap items-center gap-3">
			{#if consoleState.syntheticPreview}
				<DecisionBadge variant="SYNTHETIC" label="Synthetic" />
			{/if}
			<label class="inline-flex items-center gap-1.5 text-xs font-medium text-[#1C2721] cursor-pointer">
				<input
					type="checkbox"
					bind:checked={showLogistic}
					class="accent-[#236B52] rounded-[2px]"
				/>
				<span>Compare Logistic</span>
			</label>
			<label class="inline-flex items-center gap-1.5 text-xs font-medium text-[#1C2721] cursor-pointer">
				<input
					type="checkbox"
					bind:checked={showAbstention}
					class="accent-[#236B52] rounded-[2px]"
				/>
				<span>Show Abstention Curve</span>
			</label>
		</div>
	</div>

	<!-- Active alpha metrics -->
	<div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3.5">
		<div class="specimen-panel p-4">
			<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
				Calibrated Cutoff τ(α)
			</div>
			<div class="text-2xl font-bold font-mono-data text-[#1C2721] my-1">
				{activeLgbm ? formatFixed(activeLgbm.threshold, 4) : 'TBD'}
			</div>
			<div class="text-xs text-[#5C6860]">
				Accept if calibrated risk &lt; τ at α={consoleState.alpha.toFixed(2)}
			</div>
		</div>

		<div class="specimen-panel p-4">
			<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
				Empirical False-Accept
			</div>
			<div class="text-2xl font-bold font-mono-data {boundSatisfied ? 'text-[#236B52]' : 'text-[#B84A39]'} my-1">
				{activeLgbm ? formatPct(activeLgbm.false_accept_rate, 2) : 'TBD'}
			</div>
			<div class="text-xs {boundSatisfied ? 'text-[#236B52]' : 'text-[#B84A39]'}">
				{#if activeLgbm}
					{boundSatisfied ? 'Guarantee satisfied' : 'Exceeds target'} (FAR ≤ α={formatPct(consoleState.alpha, 1)})
				{:else}
					Awaiting conformal sweep [TBD]
				{/if}
			</div>
		</div>

		<div class="specimen-panel p-4">
			<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
				Abstention Rate
			</div>
			<div class="text-2xl font-bold font-mono-data text-[#B84A39] my-1">
				{activeLgbm ? formatPct(activeLgbm.abstention_rate, 1) : 'TBD'}
			</div>
			<div class="text-xs text-[#5C6860]">
				{#if activeLogistic}
					Logistic @ α={consoleState.alpha.toFixed(2)}: {formatPct(activeLogistic.abstention_rate, 1)}
				{:else}
					Held back or escalated to protect risk bound
				{/if}
			</div>
		</div>

		<div class="specimen-panel p-4">
			<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
				ECE &amp; Brier Score
			</div>
			<div class="text-2xl font-bold font-mono-data text-[#1C2721] my-1">
				{activeLgbm ? `${activeLgbm.ece.toFixed(3)} · ${activeLgbm.brier_score.toFixed(3)}` : 'TBD'}
			</div>
			<div class="text-xs text-[#5C6860]">
				{#if activeLogistic}
					Logistic: ECE {activeLogistic.ece.toFixed(3)} · Brier {activeLogistic.brier_score.toFixed(3)}
				{:else}
					Probability calibration quality on test split
				{/if}
			</div>
		</div>
	</div>

	<!-- Interactive Conformal Sweep SVG -->
	<section class="specimen-panel p-4 space-y-3" aria-label="Conformal alpha sweep plot">
		<div class="flex flex-wrap items-center justify-between gap-2">
			<h2 class="text-sm font-bold text-[#1C2721]">
				Empirical False-Accept Rate vs. Target Risk Bound α
			</h2>
			<div class="flex flex-wrap items-center gap-4 text-xs text-[#5C6860]">
				<span class="inline-flex items-center gap-1.5">
					<span class="w-3 h-0.5 bg-[#1C2721]"></span>
					LightGBM + MAPIE
				</span>
				{#if showLogistic}
					<span class="inline-flex items-center gap-1.5">
						<span class="w-3 h-0.5 border-t-2 border-dashed border-[#5C6860]"></span>
						Logistic + MAPIE
					</span>
				{/if}
				{#if showAbstention}
					<span class="inline-flex items-center gap-1.5">
						<span class="w-3 h-0.5 border-t-2 border-dotted border-[#236B52]"></span>
						Abstention Rate (Right Axis)
					</span>
				{/if}
				<span class="inline-flex items-center gap-1.5">
					<span class="w-0.5 h-3 bg-[#B84A39]"></span>
					Active α={consoleState.alpha.toFixed(2)}
				</span>
			</div>
		</div>

		<ConformalChart
			sweep={sweep}
			logisticSweep={logisticSweep}
			activeAlpha={consoleState.alpha}
			compact={false}
			{showLogistic}
			{showAbstention}
			onSelectAlpha={(a) => consoleState.setAlpha(a)}
		/>
	</section>

	<!-- Guardrail Method Comparison -->
	<section class="specimen-panel p-4 space-y-3">
		<div class="flex items-center justify-between">
			<h2 class="text-sm font-bold text-[#1C2721]">Guardrail Calibration Comparison</h2>
			<span class="text-xs font-mono-data text-[#5C6860]">
				Split sizes: {riskMetrics?.split_sizes?.train?.toLocaleString() ?? '—'} train / {riskMetrics?.split_sizes?.cal?.toLocaleString() ?? '—'} cal / {riskMetrics?.split_sizes?.test?.toLocaleString() ?? '—'} test
			</span>
		</div>

		{#if comparisonRows.length === 0}
			<p class="text-xs text-[#5C6860] py-4">Risk comparison metrics unavailable [TBD].</p>
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full min-w-[640px] border-collapse text-[13px]">
					<thead>
						<tr class="border-b border-[#DCD6C8] text-[11px] uppercase tracking-[0.06em] text-[#5C6860]">
							<th class="text-left py-2 px-2 font-semibold">Guardrail Method</th>
							<th class="text-right py-2 px-2 font-semibold">Target α</th>
							<th class="text-right py-2 px-2 font-semibold">Threshold τ</th>
							<th class="text-right py-2 px-2 font-semibold">False-Accept</th>
							<th class="text-right py-2 px-2 font-semibold">Abstention</th>
							<th class="text-right py-2 px-2 font-semibold">ECE</th>
							<th class="text-right py-2 px-2 font-semibold">Brier Score</th>
						</tr>
					</thead>
					<tbody>
						{#each comparisonRows as row}
							{@const targetAlpha = row.target_alpha ?? row.alpha ?? null}
							{@const violates = targetAlpha !== null && row.false_accept_rate > targetAlpha}
							<tr class="border-b border-[#ECE7DC]">
								<td class="py-2.5 px-2 font-medium text-[#1C2721]">{row.guardrail_method}</td>
								<td class="py-2.5 px-2 text-right font-mono-data">
									{targetAlpha !== null ? targetAlpha.toFixed(2) : 'N/A'}
								</td>
								<td class="py-2.5 px-2 text-right font-mono-data">{row.threshold.toFixed(4)}</td>
								<td class="py-2.5 px-2 text-right font-mono-data {violates ? 'text-[#B84A39] font-semibold' : 'text-[#236B52] font-semibold'}">
									{formatPct(row.false_accept_rate, 2)}
								</td>
								<td class="py-2.5 px-2 text-right font-mono-data">{formatPct(row.abstention_rate, 1)}</td>
								<td class="py-2.5 px-2 text-right font-mono-data">{row.ece.toFixed(4)}</td>
								<td class="py-2.5 px-2 text-right font-mono-data">{row.brier_score.toFixed(4)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</section>
</div>
