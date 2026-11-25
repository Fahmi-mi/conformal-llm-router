<script lang="ts">
	import {
		findNearestConformalPoint,
		findNearestParetoPoint,
		formatLatencyMs,
		formatPct,
		formatPercentValue,
		formatSpecimenId,
		isArtifactReady
	} from '#lib/analytics';
	import ConformalChart from '#lib/components/ConformalChart.svelte';
	import DecisionBadge from '#lib/components/DecisionBadge.svelte';
	import EvidenceStrip from '#lib/components/EvidenceStrip.svelte';
	import Icon from '#lib/components/Icon.svelte';
	import ParetoChart from '#lib/components/ParetoChart.svelte';
	import { consoleState } from '#lib/state.svelte';

	let expandedId = $state<string | null>(null);

	const conformalSweep = $derived(consoleState.benchmarks?.conformal?.sweep ?? []);
	const paretoCurve = $derived(
		consoleState.paretoTelemetry?.offline_curve?.length
			? consoleState.paretoTelemetry.offline_curve
			: (consoleState.benchmarks?.pareto?.curve ?? [])
	);
	const paretoBaselines = $derived(consoleState.benchmarks?.pareto?.baselines ?? {});

	const activeConformal = $derived(findNearestConformalPoint(conformalSweep, consoleState.alpha));
	const activePareto = $derived(findNearestParetoPoint(paretoCurve, consoleState.lambdaCost));

	const liveRequests = $derived(consoleState.paretoTelemetry?.total_requests ?? 0);
	const hasLiveTraffic = $derived(liveRequests > 0);
	const activeLivePoint = $derived(
		consoleState.paretoTelemetry?.live_points?.find(
			(pt) =>
				Math.abs(pt.alpha_used - consoleState.alpha) < 0.005 &&
				Math.abs(pt.lambda_cost_used - consoleState.lambdaCost) < 0.005
		) ?? null
	);

	const recentItems = $derived(consoleState.recentAudit?.items ?? []);
	const farWithinBound = $derived(
		activeConformal ? activeConformal.false_accept_rate <= consoleState.alpha + 1e-6 : true
	);

	function toggleExpand(id: string) {
		expandedId = expandedId === id ? null : id;
	}
</script>

<svelte:head>
	<title>Executive Overview — Conformal LLM Router</title>
</svelte:head>

<div class="space-y-4">
	<!-- Page topbar -->
	<div class="flex flex-wrap items-center justify-between gap-3 border-b border-[#DCD6C8] pb-3">
		<div>
			<h1 class="text-2xl font-bold tracking-[-0.02em] text-[#1C2721]">
				Executive Overview &amp; Live Risk Ledger
			</h1>
		</div>
		<div class="flex flex-wrap items-center gap-2">
			{#if consoleState.syntheticPreview}
				<DecisionBadge variant="SYNTHETIC" label="Sample Preview [Synthetic]" />
			{:else if consoleState.health}
				<DecisionBadge
					variant={consoleState.health.status === 'ok' ? 'PASS' : 'FALLBACK'}
					label={consoleState.health.status === 'ok' ? 'Gateway Nominal' : 'Gateway Degraded'}
				/>
			{:else}
				<DecisionBadge variant="ABSTAIN" label="Gateway Offline" />
			{/if}

			{#if consoleState.health}
				<span class="font-mono-data text-[11px] text-[#5C6860] border border-[#DCD6C8] bg-[#FCFBF8] px-2 py-0.5 rounded-[2px]">
					Artifacts: {consoleState.health.artifacts.calibrator && consoleState.health.artifacts.router ? 'Loaded' : 'Missing'} ·
					Store: {consoleState.health.storage.sqlite && consoleState.health.storage.faiss ? 'SQLite+FAISS' : 'Degraded'}
				</span>
			{/if}
		</div>
	</div>

	<!-- 4-KPI Strip -->
	{#if consoleState.loading && !consoleState.benchmarks && !consoleState.paretoTelemetry}
		<div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3.5" aria-label="Loading telemetry KPIs">
			{#each [0, 1, 2, 3] as idx}
				<div class="specimen-panel p-4 animate-pulse space-y-2.5" data-skeleton={idx}>
					<div class="h-3 w-28 bg-[#E3DDD0] rounded-[2px]"></div>
					<div class="h-7 w-24 bg-[#E3DDD0] rounded-[2px]"></div>
					<div class="h-3 w-36 bg-[#EFECE3] rounded-[2px]"></div>
				</div>
			{/each}
		</div>
	{:else}
		<section aria-label="Key telemetry indicators" class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3.5">
			<div class="specimen-panel p-4">
				<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
					Empirical False-Accept
				</div>
				<div class="text-[28px] font-bold leading-tight my-1.5 font-mono-data text-[#1C2721]">
					{activeConformal ? formatPct(activeConformal.false_accept_rate, 2) : 'TBD'}
				</div>
				<div class="text-xs {farWithinBound ? 'text-[#236B52]' : 'text-[#B84A39]'}">
					{#if activeConformal}
						{farWithinBound ? 'Within' : 'Above'} α ≤ {(consoleState.alpha * 100).toFixed(1)}% bound · RAGTruth test
					{:else}
						Calibration sweep unrun [TBD]
					{/if}
				</div>
			</div>

			<div class="specimen-panel p-4">
				<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
					Abstention Rate
				</div>
				<div class="text-[28px] font-bold leading-tight my-1.5 font-mono-data text-[#1C2721]">
					{#if hasLiveTraffic && consoleState.paretoTelemetry}
						{formatPct(consoleState.paretoTelemetry.abstention_rate, 1)}
					{:else if activeConformal}
						{formatPct(activeConformal.abstention_rate, 1)}
					{:else}
						TBD
					{/if}
				</div>
				<div class="text-xs text-[#B84A39]">
					{#if hasLiveTraffic}
						Vetoed by MAPIE gate · {liveRequests} {consoleState.syntheticPreview ? '[Synthetic]' : 'live'} reqs
					{:else if activeConformal}
						Offline MAPIE sweep @ α={consoleState.alpha.toFixed(2)}
					{:else}
						Awaiting conformal sweep [TBD]
					{/if}
				</div>
			</div>

			<div class="specimen-panel p-4">
				<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
					Realized Cost Savings
				</div>
				<div class="text-[28px] font-bold leading-tight my-1.5 font-mono-data text-[#1C2721]">
					{#if hasLiveTraffic && consoleState.paretoTelemetry}
						{formatPercentValue(consoleState.paretoTelemetry.realized_cost_savings_pct, 1)}
					{:else if activePareto}
						{formatPct(activePareto.cost_savings, 1)}
					{:else}
						TBD
					{/if}
				</div>
				<div class="text-xs text-[#236B52]">
					{#if hasLiveTraffic && consoleState.paretoTelemetry}
						vs. Always-Tier-2 · Tier-1 share {formatPct(consoleState.paretoTelemetry.tier1_share, 0)}
					{:else if activePareto}
						Offline RouterBench @ λ={consoleState.lambdaCost.toFixed(2)}
					{:else}
						Awaiting Pareto sweep [TBD]
					{/if}
				</div>
			</div>

			<div class="specimen-panel p-4">
				<div class="text-[11px] uppercase tracking-[0.08em] text-[#5C6860] font-semibold">
					Cache Hit &amp; p95 Latency
				</div>
				<div class="text-[28px] font-bold leading-tight my-1.5 font-mono-data text-[#1C2721]">
					{#if hasLiveTraffic && consoleState.paretoTelemetry}
						{formatPct(consoleState.paretoTelemetry.cache_hit_rate, 0)} · {formatLatencyMs(consoleState.paretoTelemetry.p95_latency_ms)}
					{:else}
						0 reqs
					{/if}
				</div>
				<div class="text-xs text-[#5C6860]">
					{#if hasLiveTraffic}
						FAISS + SQLite telemetry ({consoleState.syntheticPreview ? 'Synthetic fixture' : 'Live SQLite'})
					{:else}
						No live traffic yet — test in Live Playground
					{/if}
				</div>
			</div>
		</section>
	{/if}

	<!-- Dual summary charts -->
	<section aria-label="Pareto and conformal calibration summaries" class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		<div class="specimen-panel p-4 flex flex-col justify-between gap-2">
			<div class="flex items-center justify-between gap-2">
				<h2 class="text-[14px] font-bold text-[#1C2721]">
					Pareto Cost-Quality Frontier (RouterBench)
				</h2>
				<div class="flex items-center gap-3">
					<span class="font-mono-data text-xs text-[#5C6860]">λ = {consoleState.lambdaCost.toFixed(2)}</span>
					<a
						href="/pareto"
						class="inline-flex items-center gap-0.5 text-xs font-semibold text-[#236B52] hover:underline"
					>
						<span>Explore</span>
						<Icon name="chevron-right" size={14} />
					</a>
				</div>
			</div>
			<ParetoChart
				curve={paretoCurve}
				baselines={paretoBaselines}
				livePoint={activeLivePoint}
				activeLambda={consoleState.lambdaCost}
				compact={true}
			/>
		</div>

		<div class="specimen-panel p-4 flex flex-col justify-between gap-2">
			<div class="flex items-center justify-between gap-2">
				<h2 class="text-[14px] font-bold text-[#1C2721]">
					Conformal Risk Calibration (RAGTruth)
				</h2>
				<div class="flex items-center gap-3">
					<span class="font-mono-data text-xs text-[#B84A39]">α = {consoleState.alpha.toFixed(2)}</span>
					<a
						href="/conformal"
						class="inline-flex items-center gap-0.5 text-xs font-semibold text-[#236B52] hover:underline"
					>
						<span>Inspect</span>
						<Icon name="chevron-right" size={14} />
					</a>
				</div>
			</div>
			<ConformalChart
				sweep={conformalSweep}
				logisticSweep={consoleState.benchmarks?.conformal?.logistic_sweep ?? []}
				activeAlpha={consoleState.alpha}
				compact={true}
				showLogistic={false}
			/>
		</div>
	</section>

	<!-- Recent Specimen Audit Log -->
	<section aria-label="Recent specimen audit log" class="specimen-panel p-4">
		<div class="flex flex-wrap items-center justify-between gap-2 mb-3">
			<div class="flex items-center gap-2.5">
				<h2 class="text-[14px] font-bold text-[#1C2721]">Recent Specimen Audit Log</h2>
				{#if consoleState.syntheticPreview}
					<DecisionBadge variant="SYNTHETIC" label="Synthetic" />
				{/if}
			</div>
			<div class="flex items-center gap-4">
				<span class="hidden sm:inline text-xs text-[#5C6860]">
					Evidence Strip: Entailment · Contradiction · Cosine · Entity
				</span>
				<a
					href="/audit"
					class="inline-flex items-center gap-0.5 text-xs font-semibold text-[#236B52] hover:underline"
				>
					<span>Full Inspector</span>
					<Icon name="chevron-right" size={14} />
				</a>
			</div>
		</div>

		{#if consoleState.loading && !consoleState.recentAudit}
			<div class="space-y-2 animate-pulse py-3">
				<div class="h-8 bg-[#EFECE3] rounded-[2px]"></div>
				<div class="h-8 bg-[#EFECE3] rounded-[2px]"></div>
				<div class="h-8 bg-[#EFECE3] rounded-[2px]"></div>
			</div>
		{:else if recentItems.length === 0}
			<div class="border-t border-[#ECE7DC] bg-[#F6F3EC] p-6 text-center space-y-2">
				<p class="text-sm font-semibold text-[#1C2721]">No live specimen records logged yet</p>
				<p class="text-xs text-[#5C6860] max-w-[60ch] mx-auto">
					Every request sent to <code class="font-mono-data">POST /v1/query</code> is persisted to SQLite with its
					routed tier, calibrated risk score, and NLI/cosine grounding strip.
				</p>
				<div class="pt-1 flex flex-wrap justify-center gap-2">
					<a
						href="/playground"
						class="px-3 py-1.5 text-xs font-semibold bg-[#1C2721] text-[#FAF8F3] rounded-[2px] hover:bg-[#2A3A31]"
					>
						Open Live Playground
					</a>
					<button
						type="button"
						onclick={() => consoleState.setSyntheticPreview(true)}
						class="px-3 py-1.5 text-xs font-semibold border border-[#B46914] bg-[#FDF6E9] text-[#8F520E] rounded-[2px] hover:bg-[#FAECCB] cursor-pointer"
					>
						Preview Sample Data [Synthetic]
					</button>
				</div>
			</div>
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full min-w-[780px] table-fixed border-collapse text-[13px]">
					<colgroup>
						<col class="w-[116px]" />
						<col class="w-[260px]" />
						<col class="w-[155px]" />
						<col class="w-[88px]" />
						<col class="w-[175px]" />
						<col class="w-[96px]" />
					</colgroup>
					<thead>
						<tr class="border-b border-[#DCD6C8] text-[11px] uppercase tracking-[0.06em] text-[#5C6860]">
							<th class="text-left py-2 px-2 font-semibold">Specimen ID</th>
							<th class="text-left py-2 px-2 font-semibold">Query Summary</th>
							<th class="text-left py-2 px-2 font-semibold">Routed Tier</th>
							<th class="text-left py-2 px-2 font-semibold">Risk Score</th>
							<th class="text-left py-2 px-2 font-semibold">Evidence Strip</th>
							<th class="text-left py-2 px-2 font-semibold">Decision</th>
						</tr>
					</thead>
					<tbody>
						{#each recentItems as item (item.request_id)}
							<tr
								class="border-b border-[#ECE7DC] hover:bg-[#F6F3EC] transition-colors cursor-pointer"
								onclick={() => toggleExpand(item.request_id)}
							>
								<td class="py-2.5 px-2 font-mono-data text-xs font-semibold text-[#1C2721]">
									{formatSpecimenId(item.request_id)}
								</td>
								<td class="py-2.5 px-2 truncate text-[#1C2721]" title={item.query_text}>
									{item.query_text}
								</td>
								<td class="py-2.5 px-2 text-xs text-[#1C2721]">
									<span class="font-semibold uppercase">{item.tier}</span>
									<span class="text-[#5C6860]"> · {item.provider}</span>
									{#if item.cache_hit}
										<span class="ml-1 text-[10px] font-semibold text-[#1B5642]">CACHE</span>
									{/if}
								</td>
								<td class="py-2.5 px-2 font-mono-data text-xs">
									{item.calibrated_risk_score !== null ? item.calibrated_risk_score.toFixed(3) : '—'}
								</td>
								<td class="py-2.5 px-2">
									<EvidenceStrip evidence={item.evidence} riskScore={item.calibrated_risk_score} mode="compact" />
								</td>
								<td class="py-2.5 px-2">
									<DecisionBadge variant={item.conformal_decision} />
								</td>
							</tr>
							{#if expandedId === item.request_id}
								<tr class="bg-[#F6F3EC] border-b border-[#DCD6C8]">
									<td colspan="6" class="p-4 space-y-3">
										<div class="flex flex-wrap items-center justify-between gap-2 text-xs text-[#5C6860]">
											<span class="font-mono-data">
												ID: {item.request_id} · Model: {item.model_used} · α={item.alpha_used.toFixed(2)} · λ={item.lambda_cost_used.toFixed(2)}
											</span>
											<a href="/audit" class="font-semibold text-[#236B52] hover:underline">
												Open in Audit Inspector
											</a>
										</div>
										<div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
											<div>
												<div class="font-semibold text-[#5C6860] uppercase text-[10.5px] mb-1">Delivered Answer</div>
												<p class="text-[#1C2721] bg-[#FCFBF8] border border-[#DCD6C8] p-2.5 rounded-[2px] leading-relaxed">
													{item.answer_text}
												</p>
											</div>
											{#if item.draft_answer}
												<div>
													<div class="font-semibold text-[#B84A39] uppercase text-[10.5px] mb-1">
														Blocked Draft Answer (Abstained)
													</div>
													<p class="text-[#1C2721] bg-[#FBE9E7] border border-[#E8BEB8] p-2.5 rounded-[2px] leading-relaxed">
														{item.draft_answer}
													</p>
												</div>
											{/if}
										</div>
										<EvidenceStrip evidence={item.evidence} riskScore={item.calibrated_risk_score} mode="detailed" />
									</td>
								</tr>
							{/if}
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</section>

	<!-- Artifact provenance strip -->
	{#if consoleState.benchmarks}
		<div class="flex flex-wrap items-center gap-3 text-xs text-[#5C6860] pt-1">
			<span>Artifact Status:</span>
			<span class="inline-flex items-center gap-1.5">
				Conformal
				<DecisionBadge variant={isArtifactReady(consoleState.benchmarks.conformal.status) ? 'READY' : 'TBD'} />
			</span>
			<span class="inline-flex items-center gap-1.5">
				Pareto Router
				<DecisionBadge variant={isArtifactReady(consoleState.benchmarks.pareto.status) ? 'READY' : 'TBD'} />
			</span>
			<span class="inline-flex items-center gap-1.5">
				Judge Audit
				<DecisionBadge variant={isArtifactReady(consoleState.benchmarks.judge_audit.status) ? 'READY' : 'TBD'} />
			</span>
		</div>
	{/if}
</div>
