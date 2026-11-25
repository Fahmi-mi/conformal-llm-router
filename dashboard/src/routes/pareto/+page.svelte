<script lang="ts">
	import {
		findNearestParetoPoint,
		formatFixed,
		formatLatencyMs,
		formatPct,
		formatPercentValue,
		formatQualityRetention,
		formatUsd
	} from '#lib/analytics';
	import DecisionBadge from '#lib/components/DecisionBadge.svelte';
	import ParetoChart from '#lib/components/ParetoChart.svelte';
	import { consoleState } from '#lib/state.svelte';
	import type { ParetoBaseline } from '#lib/types';

	let domainMode = $state<'indomain' | 'ood'>('indomain');

	const pareto = $derived(consoleState.benchmarks?.pareto);
	const oodAvailable = $derived(Boolean(pareto?.ood_available && pareto?.ood_curve?.length));
	const effectiveMode = $derived(domainMode === 'ood' && oodAvailable ? 'ood' : 'indomain');

	const activeCurve = $derived(
		effectiveMode === 'ood'
			? (pareto?.ood_curve ?? [])
			: (consoleState.paretoTelemetry?.offline_curve?.length
					? consoleState.paretoTelemetry.offline_curve
					: (pareto?.curve ?? []))
	);

	const activeBaselines = $derived(
		(effectiveMode === 'ood' ? pareto?.ood_baselines : pareto?.baselines) ?? {}
	);

	const activeAucQc = $derived(effectiveMode === 'ood' ? pareto?.ood_auc_qc : pareto?.auc_qc);
	const activeSavings95 = $derived(
		effectiveMode === 'ood'
			? pareto?.ood_cost_savings_at_95pct_quality
			: pareto?.cost_savings_at_95pct_quality
	);

	const activePoint = $derived(findNearestParetoPoint(activeCurve, consoleState.lambdaCost));
	const frontierQuality = $derived(activeBaselines.always_frontier?.avg_quality);

	const baselineRows = $derived.by(() => {
		const order: Array<{ key: string; label: string }> = [
			{ key: 'always_frontier', label: 'Always Frontier Model' },
			{ key: 'always_small', label: 'Always Small Model' },
			{ key: 'random_router', label: 'Random Router' },
			{ key: 'learned_pareto_router', label: 'Learned Pareto Router' }
		];
		return order
			.map((entry) => ({
				...entry,
				data: activeBaselines[entry.key] as ParetoBaseline | undefined
			}))
			.filter((entry) => entry.data !== undefined);
	});

	const livePoints = $derived(consoleState.paretoTelemetry?.live_points ?? []);
	const activeLivePoint = $derived(
		livePoints.find(
			(pt) =>
				Math.abs(pt.alpha_used - consoleState.alpha) < 0.005 &&
				Math.abs(pt.lambda_cost_used - consoleState.lambdaCost) < 0.005
		) ?? null
	);
</script>

<svelte:head>
	<title>Pareto Frontier Explorer — Conformal LLM Router</title>
</svelte:head>

<div class="space-y-4">
	<div class="flex flex-wrap items-center justify-between gap-3 border-b border-[#DCD6C8] pb-3">
		<div>
			<h1 class="text-2xl font-bold tracking-[-0.02em] text-[#1C2721]">
				Pareto Frontier Explorer
			</h1>
			<p class="text-xs text-[#5C6860] mt-0.5 max-w-[70ch]">
				Inspect cost-quality trade-offs across routing cost weight λ_cost ∈ [0.00, 1.00]. Click any curve point to snap λ_cost.
			</p>
		</div>

		<div class="flex items-center gap-2">
			{#if consoleState.syntheticPreview}
				<DecisionBadge variant="SYNTHETIC" label="Synthetic" />
			{/if}
			<div class="inline-flex border border-[#DCD6C8] bg-[#FCFBF8] rounded-[2px] p-0.5" role="group" aria-label="Evaluation domain split">
				<button
					type="button"
					onclick={() => (domainMode = 'indomain')}
					class="px-3 py-1 text-xs font-semibold rounded-[2px] transition-colors cursor-pointer {effectiveMode === 'indomain'
						? 'bg-[#1C2721] text-[#FAF8F3]'
						: 'text-[#5C6860] hover:text-[#1C2721]'}"
				>
					In-Domain ({pareto?.n_test?.toLocaleString() ?? '7,214'})
				</button>
				<button
					type="button"
					disabled={!oodAvailable}
					onclick={() => (domainMode = 'ood')}
					class="px-3 py-1 text-xs font-semibold rounded-[2px] transition-colors cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed {effectiveMode === 'ood'
						? 'bg-[#1C2721] text-[#FAF8F3]'
						: 'text-[#5C6860] hover:text-[#1C2721]'}"
				>
					OOD ({pareto?.ood_tasks?.join(', ') || 'mbpp'}{pareto?.n_ood ? ` · ${pareto.n_ood}` : ''})
				</button>
			</div>
		</div>
	</div>

	<!-- 4 Key Pareto Metrics -->
	<div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3.5">
		<div class="specimen-panel p-4">
			<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">AUC-QC</div>
			<div class="text-2xl font-bold font-mono-data text-[#1C2721] my-1">
				{formatFixed(activeAucQc, 4)}
			</div>
			<div class="text-xs text-[#5C6860]">
				Area Under Quality-Cost Curve ({effectiveMode === 'ood' ? 'OOD holdout' : 'In-Domain test'})
			</div>
		</div>

		<div class="specimen-panel p-4">
			<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
				Cost Savings @ 95% Quality
			</div>
			<div class="text-2xl font-bold font-mono-data text-[#236B52] my-1">
				{formatPct(activeSavings95, 1)}
			</div>
			<div class="text-xs text-[#5C6860]">
				Max savings retaining ≥95% frontier quality
			</div>
		</div>

		<div class="specimen-panel p-4">
			<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
				Operating Point (λ = {consoleState.lambdaCost.toFixed(2)})
			</div>
			<div class="text-2xl font-bold font-mono-data text-[#1C2721] my-1">
				{activePoint ? formatPct(activePoint.avg_quality, 1) : 'TBD'}
			</div>
			<div class="text-xs text-[#236B52]">
				{#if activePoint}
					Retains {formatQualityRetention(activePoint.avg_quality, frontierQuality)} frontier · Tier-1 {formatPct(activePoint.tier1_rate, 1)}
				{:else}
					Awaiting curve data [TBD]
				{/if}
			</div>
		</div>

		<div class="specimen-panel p-4">
			<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
				Live Gateway Tier-1 Share
			</div>
			<div class="text-2xl font-bold font-mono-data text-[#1C2721] my-1">
				{consoleState.paretoTelemetry && consoleState.paretoTelemetry.total_requests > 0
					? formatPct(consoleState.paretoTelemetry.tier1_share, 1)
					: '0 reqs'}
			</div>
			<div class="text-xs text-[#5C6860]">
				{#if consoleState.paretoTelemetry && consoleState.paretoTelemetry.total_requests > 0}
					Realized savings {formatPercentValue(consoleState.paretoTelemetry.realized_cost_savings_pct, 1)} · p95 {formatLatencyMs(consoleState.paretoTelemetry.p95_latency_ms)}
				{:else}
					Filtered for α={consoleState.alpha.toFixed(2)}, λ={consoleState.lambdaCost.toFixed(2)}
				{/if}
			</div>
		</div>
	</div>

	<!-- Interactive Pareto SVG -->
	<section class="specimen-panel p-4 space-y-3" aria-label="Interactive Pareto frontier chart">
		<div class="flex flex-wrap items-center justify-between gap-2">
			<h2 class="text-sm font-bold text-[#1C2721]">
				Quality vs. Normalized Cost Frontier — {effectiveMode === 'ood' ? `OOD (${pareto?.ood_tasks?.join(', ') || 'mbpp'})` : 'RouterBench Test Split'}
			</h2>
			<div class="flex flex-wrap items-center gap-4 text-xs text-[#5C6860]">
				<span class="inline-flex items-center gap-1.5">
					<span class="w-3 h-0.5 bg-[#236B52]"></span>
					Learned Router Sweep
				</span>
				<span class="inline-flex items-center gap-1.5">
					<span class="w-2.5 h-2.5 rounded-full bg-[#B84A39]"></span>
					Active λ={consoleState.lambdaCost.toFixed(2)}
				</span>
				<span class="inline-flex items-center gap-1.5">
					<span class="w-2.5 h-2.5 border border-[#5C6860] bg-[#FCFBF8]"></span>
					Random Baseline
				</span>
			</div>
		</div>

		<ParetoChart
			curve={activeCurve}
			baselines={activeBaselines}
			livePoint={activeLivePoint}
			activeLambda={consoleState.lambdaCost}
			compact={false}
			datasetLabel={effectiveMode === 'ood' ? 'OOD mbpp' : 'RouterBench In-Domain'}
			onSelectLambda={(lam) => consoleState.setLambdaCost(lam)}
		/>
	</section>

	<!-- Baselines + Live Tier Split -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		<section class="specimen-panel p-4 space-y-3">
			<div class="flex items-center justify-between">
				<h2 class="text-sm font-bold text-[#1C2721]">Routing Strategy Baselines</h2>
				<span class="text-xs font-mono-data text-[#5C6860]">
					{effectiveMode === 'ood' ? 'OOD mbpp' : 'In-Domain'}
				</span>
			</div>

			{#if baselineRows.length === 0}
				<p class="text-xs text-[#5C6860] py-4">Baseline metrics unavailable [TBD].</p>
			{:else}
				<div class="overflow-x-auto">
					<table class="w-full min-w-[460px] border-collapse text-[13px]">
						<thead>
							<tr class="border-b border-[#DCD6C8] text-[11px] uppercase tracking-[0.06em] text-[#5C6860]">
								<th class="text-left py-2 px-2 font-semibold">Strategy</th>
								<th class="text-right py-2 px-2 font-semibold">Avg Quality</th>
								<th class="text-right py-2 px-2 font-semibold">Norm. Cost</th>
								<th class="text-right py-2 px-2 font-semibold">Savings</th>
								<th class="text-right py-2 px-2 font-semibold">Tier-1</th>
							</tr>
						</thead>
						<tbody>
							{#each baselineRows as row}
								<tr class="border-b border-[#ECE7DC]">
									<td class="py-2 px-2 font-medium text-[#1C2721]">
										{row.label}
										{#if row.data?.lambda_cost !== undefined}
											<span class="font-mono-data text-[11px] text-[#5C6860]">(λ={row.data.lambda_cost.toFixed(2)})</span>
										{/if}
									</td>
									<td class="py-2 px-2 text-right font-mono-data">{formatPct(row.data?.avg_quality, 1)}</td>
									<td class="py-2 px-2 text-right font-mono-data">{formatPct(row.data?.normalized_cost, 1)}</td>
									<td class="py-2 px-2 text-right font-mono-data text-[#236B52]">{formatPct(row.data?.cost_savings, 1)}</td>
									<td class="py-2 px-2 text-right font-mono-data">{formatPct(row.data?.tier1_rate, 1)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</section>

		<section class="specimen-panel p-4 space-y-3">
			<div class="flex items-center justify-between">
				<h2 class="text-sm font-bold text-[#1C2721]">Live Telemetry Cohorts by (α, λ_cost)</h2>
				<span class="text-xs font-mono-data text-[#5C6860]">
					{livePoints.length} cohort{livePoints.length === 1 ? '' : 's'}
				</span>
			</div>

			{#if livePoints.length === 0}
				<div class="border-t border-[#ECE7DC] bg-[#F6F3EC] p-5 text-xs text-[#5C6860]">
					No live SQLite telemetry cohorts recorded yet. Send requests in <a href="/playground" class="font-semibold text-[#236B52] hover:underline">Live Playground</a> to populate real Tier-1 vs. Tier-2 realized cost splits.
				</div>
			{:else}
				<div class="overflow-x-auto">
					<table class="w-full min-w-[460px] border-collapse text-[13px]">
						<thead>
							<tr class="border-b border-[#DCD6C8] text-[11px] uppercase tracking-[0.06em] text-[#5C6860]">
								<th class="text-left py-2 px-2 font-semibold">Cohort (α, λ)</th>
								<th class="text-right py-2 px-2 font-semibold">Reqs</th>
								<th class="text-right py-2 px-2 font-semibold">Tier-1</th>
								<th class="text-right py-2 px-2 font-semibold">Avg Cost</th>
								<th class="text-right py-2 px-2 font-semibold">Savings</th>
								<th class="text-right py-2 px-2 font-semibold">p95</th>
							</tr>
						</thead>
						<tbody>
							{#each livePoints as pt}
								{@const selected =
									Math.abs(pt.alpha_used - consoleState.alpha) < 0.005 &&
									Math.abs(pt.lambda_cost_used - consoleState.lambdaCost) < 0.005}
								<tr class="border-b border-[#ECE7DC] {selected ? 'bg-[#E4F2EC]/50 font-semibold' : ''}">
									<td class="py-2 px-2 font-mono-data text-xs">
										({pt.alpha_used.toFixed(2)}, {pt.lambda_cost_used.toFixed(2)})
									</td>
									<td class="py-2 px-2 text-right font-mono-data">{pt.total_requests}</td>
									<td class="py-2 px-2 text-right font-mono-data">{formatPct(pt.tier1_share, 0)}</td>
									<td class="py-2 px-2 text-right font-mono-data">{formatUsd(pt.avg_cost_usd)}</td>
									<td class="py-2 px-2 text-right font-mono-data text-[#236B52]">
										{formatPercentValue(pt.realized_cost_savings_pct, 1)}
									</td>
									<td class="py-2 px-2 text-right font-mono-data">{formatLatencyMs(pt.p95_latency_ms)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</section>
	</div>
</div>
