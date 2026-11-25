<script lang="ts">
	import { formatFixed, formatPct, isArtifactReady } from '#lib/analytics';
	import DecisionBadge from '#lib/components/DecisionBadge.svelte';
	import { consoleState } from '#lib/state.svelte';
	import type { ParetoBaseline, RiskMetricEntry } from '#lib/types';

	const summary = $derived(consoleState.benchmarks);
	const judge = $derived(summary?.judge_audit);
	const risk = $derived(summary?.risk_metrics);
	const pareto = $derived(summary?.pareto);

	const judgeReady = $derived(isArtifactReady(judge?.status) && Boolean(judge?.metrics));
	const conformalReady = $derived(isArtifactReady(risk?.status));
	const paretoReady = $derived(isArtifactReady(pareto?.status));

	const conformalRows = $derived.by(() => {
		const fallbackRows = [
			{ label: 'Uncalibrated Heuristic Threshold', alphaLabel: 'N/A', entry: risk?.uncalibrated_heuristic },
			{ label: 'Logistic + Split Conformal (MAPIE)', alphaLabel: '0.10', entry: risk?.logistic_mapie?.['0.10'] },
			{ label: 'LightGBM + Split Conformal (MAPIE)', alphaLabel: '0.10', entry: risk?.lightgbm_mapie?.['0.10'] },
			{ label: 'LightGBM + Split Conformal (MAPIE)', alphaLabel: '0.05', entry: risk?.lightgbm_mapie?.['0.05'] },
			{ label: 'Logistic + Split Conformal (MAPIE)', alphaLabel: '0.05', entry: risk?.logistic_mapie?.['0.05'] }
		];
		return fallbackRows as Array<{ label: string; alphaLabel: string; entry?: RiskMetricEntry }>;
	});

	const paretoRows = $derived.by(() => {
		const baselines = pareto?.baselines ?? {};
		const defs = [
			{ key: 'always_frontier', label: 'Always Frontier Model' },
			{ key: 'always_small', label: 'Always Small Model' },
			{ key: 'random_router', label: 'Random Router' },
			{ key: 'learned_pareto_router', label: 'Learned Pareto Router' }
		];
		return defs.map((d) => ({
			...d,
			data: baselines[d.key] as ParetoBaseline | undefined
		}));
	});
</script>

<svelte:head>
	<title>Offline Benchmark Summary — Conformal LLM Router</title>
</svelte:head>

<div class="space-y-5">
	<div class="flex flex-wrap items-center justify-between gap-3 border-b border-[#DCD6C8] pb-3">
		<div>
			<h1 class="text-2xl font-bold tracking-[-0.02em] text-[#1C2721]">
				Offline Benchmark Summary
			</h1>
			<p class="text-xs text-[#5C6860] mt-0.5 max-w-[74ch]">
				Verified offline pipeline results served from <code class="font-mono-data">/v1/benchmarks/summary</code>. Unrun artifacts remain explicitly marked <code class="font-mono-data">TBD</code>.
			</p>
		</div>

		<div class="flex flex-wrap items-center gap-2 text-xs">
			{#if consoleState.syntheticPreview}
				<DecisionBadge variant="SYNTHETIC" label="Synthetic" />
			{/if}
			<span class="inline-flex items-center gap-1.5 border border-[#DCD6C8] bg-[#FCFBF8] px-2.5 py-1 rounded-[2px]">
				Conformal <DecisionBadge variant={conformalReady ? 'READY' : 'TBD'} />
			</span>
			<span class="inline-flex items-center gap-1.5 border border-[#DCD6C8] bg-[#FCFBF8] px-2.5 py-1 rounded-[2px]">
				Pareto <DecisionBadge variant={paretoReady ? 'READY' : 'TBD'} />
			</span>
			<span class="inline-flex items-center gap-1.5 border border-[#DCD6C8] bg-[#FCFBF8] px-2.5 py-1 rounded-[2px]">
				Judge Audit <DecisionBadge variant={judgeReady ? 'READY' : 'TBD'} />
			</span>
		</div>
	</div>

	{#if consoleState.loading && !summary}
		<div class="space-y-4" aria-label="Loading offline benchmarks">
			{#each [0, 1, 2] as idx}
				<div class="specimen-panel p-4 animate-pulse space-y-3" data-skeleton={idx}>
					<div class="h-4 w-56 bg-[#E3DDD0] rounded-[2px]"></div>
					<div class="h-8 bg-[#EFECE3] rounded-[2px]"></div>
					<div class="h-8 bg-[#EFECE3] rounded-[2px]"></div>
				</div>
			{/each}
		</div>
	{:else}
		{#if !summary}
			<div role="status" class="border border-[#DFB97D] bg-[#FDF6E9] text-[#8F520E] px-4 py-2.5 rounded-[3px] text-xs">
				Offline benchmark summary could not be loaded from <code class="font-mono-data">/v1/benchmarks/summary</code>. Connect FastAPI or enable Sample Preview [Synthetic]; fields remain <code class="font-mono-data">TBD</code>.
			</div>
		{/if}

		<!-- Table 1: Conformal Hallucination Control -->
	<section class="specimen-panel p-4 space-y-3" aria-label="Conformal hallucination control table">
		<div class="flex flex-wrap items-center justify-between gap-2">
			<h2 class="text-sm font-bold text-[#1C2721]">
				1. Conformal Hallucination Control (RAGTruth Test Split)
			</h2>
			<div class="flex items-center gap-2">
				{#if risk?.split_sizes}
					<span class="font-mono-data text-xs text-[#5C6860]">
						n_train={risk.split_sizes.train.toLocaleString()} · n_cal={risk.split_sizes.cal.toLocaleString()} · n_test={risk.split_sizes.test.toLocaleString()}
					</span>
				{/if}
				<DecisionBadge variant={conformalReady ? 'READY' : 'TBD'} />
			</div>
		</div>

		<div class="overflow-x-auto">
			<table class="w-full min-w-[640px] border-collapse text-[13px]">
				<thead>
					<tr class="border-b border-[#DCD6C8] text-[11px] uppercase tracking-[0.06em] text-[#5C6860]">
						<th class="text-left py-2 px-2 font-semibold">Guardrail Method</th>
						<th class="text-right py-2 px-2 font-semibold">Target Risk α</th>
						<th class="text-right py-2 px-2 font-semibold">Empirical False-Accept Rate</th>
						<th class="text-right py-2 px-2 font-semibold">Abstention Rate</th>
						<th class="text-right py-2 px-2 font-semibold">ECE</th>
						<th class="text-right py-2 px-2 font-semibold">Brier Score</th>
					</tr>
				</thead>
				<tbody>
					{#each conformalRows as row}
						{@const e = row.entry}
						{@const targetAlpha = e ? (e.target_alpha ?? e.alpha ?? null) : null}
						{@const farColorClass = !e
							? 'text-[#8F520E]'
							: targetAlpha === null
								? 'text-[#1C2721]'
								: e.false_accept_rate <= targetAlpha
									? 'text-[#236B52]'
									: 'text-[#B84A39]'}
						<tr class="border-b border-[#ECE7DC]">
							<td class="py-2.5 px-2 font-medium text-[#1C2721]">{row.label}</td>
							<td class="py-2.5 px-2 text-right font-mono-data">{row.alphaLabel}</td>
							<td class="py-2.5 px-2 text-right font-mono-data font-semibold {farColorClass}">
								{e ? formatPct(e.false_accept_rate, 2) : 'TBD'}
							</td>
							<td class="py-2.5 px-2 text-right font-mono-data">
								{e ? formatPct(e.abstention_rate, 1) : 'TBD'}
							</td>
							<td class="py-2.5 px-2 text-right font-mono-data">
								{e ? formatFixed(e.ece, 4) : 'TBD'}
							</td>
							<td class="py-2.5 px-2 text-right font-mono-data">
								{e ? formatFixed(e.brier_score, 4) : 'TBD'}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</section>

	<!-- Table 2: Cost-Quality Routing -->
	<section class="specimen-panel p-4 space-y-3" aria-label="Cost-quality routing table">
		<div class="flex flex-wrap items-center justify-between gap-2">
			<h2 class="text-sm font-bold text-[#1C2721]">
				2. Cost-Quality Routing (RouterBench)
			</h2>
			<div class="flex items-center gap-2">
				{#if paretoReady && pareto}
					<span class="font-mono-data text-xs text-[#5C6860]">
						AUC-QC={formatFixed(pareto.auc_qc, 4)} · Savings@95%={formatPct(pareto.cost_savings_at_95pct_quality, 1)}{#if pareto.ood_available} · OOD ({pareto.ood_tasks?.join(', ') || 'mbpp'}) AUC-QC={formatFixed(pareto.ood_auc_qc, 4)}{/if}
					</span>
				{/if}
				<DecisionBadge variant={paretoReady ? 'READY' : 'TBD'} />
			</div>
		</div>

		<div class="overflow-x-auto">
			<table class="w-full min-w-[600px] border-collapse text-[13px]">
				<thead>
					<tr class="border-b border-[#DCD6C8] text-[11px] uppercase tracking-[0.06em] text-[#5C6860]">
						<th class="text-left py-2 px-2 font-semibold">Routing Strategy</th>
						<th class="text-right py-2 px-2 font-semibold">Avg Quality Score</th>
						<th class="text-right py-2 px-2 font-semibold">Avg Cost vs. Frontier (%)</th>
						<th class="text-right py-2 px-2 font-semibold">Cost Savings (%)</th>
						<th class="text-right py-2 px-2 font-semibold">Tier-1 Share</th>
					</tr>
				</thead>
				<tbody>
					{#each paretoRows as row}
						{@const b = row.data}
						<tr class="border-b border-[#ECE7DC]">
							<td class="py-2.5 px-2 font-medium text-[#1C2721]">
								{row.label}
								{#if b?.lambda_cost !== undefined}
									<span class="font-mono-data text-xs text-[#5C6860]">(λ={b.lambda_cost.toFixed(2)})</span>
								{/if}
							</td>
							<td class="py-2.5 px-2 text-right font-mono-data">
								{b ? formatPct(b.avg_quality, 1) : 'TBD'}
							</td>
							<td class="py-2.5 px-2 text-right font-mono-data">
								{b ? formatPct(b.normalized_cost, 1) : row.key === 'always_frontier' ? '100%' : 'TBD'}
							</td>
							<td class="py-2.5 px-2 text-right font-mono-data font-semibold {b ? 'text-[#236B52]' : 'text-[#8F520E]'}">
								{b ? formatPct(b.cost_savings, 1) : row.key === 'always_frontier' ? '0%' : 'TBD'}
							</td>
							<td class="py-2.5 px-2 text-right font-mono-data">
								{b ? formatPct(b.tier1_rate, 1) : 'TBD'}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</section>

	<!-- Table 3: Judge Audit -->
	<section class="specimen-panel p-4 space-y-3" aria-label="LLM-as-a-Judge bias audit">
		<div class="flex flex-wrap items-center justify-between gap-2">
			<h2 class="text-sm font-bold text-[#1C2721]">
				3. Bias-Corrected LLM-as-a-Judge Audit (RAGTruth)
			</h2>
			<DecisionBadge variant={judgeReady ? 'READY' : 'TBD'} />
		</div>

		{#if !judgeReady}
			<div class="border-t border-[#DFB97D] bg-[#FDF6E9] p-4 text-xs text-[#8F520E] space-y-1.5">
				<div class="font-semibold">
					Judge Bias Audit Unrun [TBD] — Requested Sample Size: {judge?.requested_sample_size ?? 300}
				</div>
				<p class="text-[#72420B]">
					{judge?.reason || 'No DEEPSEEK_API_KEY or GROQ_API_KEY configured in environment or .env.'}
				</p>
			</div>

			<div class="overflow-x-auto">
				<table class="w-full min-w-[560px] border-collapse text-[13px]">
					<thead>
						<tr class="border-b border-[#DCD6C8] text-[11px] uppercase tracking-[0.06em] text-[#5C6860]">
							<th class="text-left py-2 px-2 font-semibold">Agreement &amp; Bias Metric</th>
							<th class="text-right py-2 px-2 font-semibold">Raw Judge</th>
							<th class="text-right py-2 px-2 font-semibold">Bias-Corrected (Ridge)</th>
						</tr>
					</thead>
					<tbody>
						<tr class="border-b border-[#ECE7DC]">
							<td class="py-2 px-2 font-medium">Cohen's Kappa</td>
							<td class="py-2 px-2 text-right font-mono-data text-[#8F520E]">TBD</td>
							<td class="py-2 px-2 text-right font-mono-data text-[#8F520E]">TBD</td>
						</tr>
						<tr class="border-b border-[#ECE7DC]">
							<td class="py-2 px-2 font-medium">Spearman Correlation (r)</td>
							<td class="py-2 px-2 text-right font-mono-data text-[#8F520E]">TBD</td>
							<td class="py-2 px-2 text-right font-mono-data text-[#8F520E]">TBD</td>
						</tr>
						<tr class="border-b border-[#ECE7DC]">
							<td class="py-2 px-2 font-medium">Verbosity &amp; Position Coefs</td>
							<td class="py-2 px-2 text-right font-mono-data text-[#8F520E]">TBD</td>
							<td class="py-2 px-2 text-right font-mono-data text-[#8F520E]">TBD</td>
						</tr>
					</tbody>
				</table>
			</div>
		{:else if judge?.metrics}
			{@const rawM = judge.metrics}
			{@const corrM = judge.corrected_metrics}
			{@const bias = judge.bias_correction}
			<div class="text-xs text-[#5C6860] font-mono-data">
				Provider: {judge.provider ?? '—'} · Sample size: {judge.sample_size ?? rawM.n_samples} · Paired evals: {judge.paired_evaluations ?? '—'}
			</div>
			<div class="overflow-x-auto">
				<table class="w-full min-w-[560px] border-collapse text-[13px]">
					<thead>
						<tr class="border-b border-[#DCD6C8] text-[11px] uppercase tracking-[0.06em] text-[#5C6860]">
							<th class="text-left py-2 px-2 font-semibold">Agreement &amp; Bias Metric</th>
							<th class="text-right py-2 px-2 font-semibold">Raw Judge</th>
							<th class="text-right py-2 px-2 font-semibold">Bias-Corrected (Ridge)</th>
						</tr>
					</thead>
					<tbody>
						<tr class="border-b border-[#ECE7DC]">
							<td class="py-2 px-2 font-medium">Cohen's Kappa</td>
							<td class="py-2 px-2 text-right font-mono-data">{rawM.cohens_kappa.toFixed(4)}</td>
							<td class="py-2 px-2 text-right font-mono-data text-[#236B52] font-semibold">
								{corrM ? corrM.cohens_kappa.toFixed(4) : 'TBD'}
							</td>
						</tr>
						<tr class="border-b border-[#ECE7DC]">
							<td class="py-2 px-2 font-medium">Spearman Correlation (r)</td>
							<td class="py-2 px-2 text-right font-mono-data">{rawM.spearman_r.toFixed(4)}</td>
							<td class="py-2 px-2 text-right font-mono-data text-[#236B52] font-semibold">
								{corrM ? corrM.spearman_r.toFixed(4) : 'TBD'}
							</td>
						</tr>
						<tr class="border-b border-[#ECE7DC]">
							<td class="py-2 px-2 font-medium">Binary Agreement Accuracy</td>
							<td class="py-2 px-2 text-right font-mono-data">{formatPct(rawM.accuracy, 1)}</td>
							<td class="py-2 px-2 text-right font-mono-data">{corrM ? formatPct(corrM.accuracy, 1) : 'TBD'}</td>
						</tr>
						{#if bias}
							<tr class="border-b border-[#ECE7DC]">
								<td class="py-2 px-2 font-medium">Verbosity / Position Coefficients</td>
								<td class="py-2 px-2 text-right font-mono-data" colspan="2">
									verbosity={bias.verbosity_coef.toFixed(5)} · position={bias.position_coef.toFixed(5)}
								</td>
							</tr>
						{/if}
					</tbody>
				</table>
			</div>
		{/if}
	</section>
	{/if}
</div>
