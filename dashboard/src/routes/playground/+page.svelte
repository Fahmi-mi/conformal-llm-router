<script lang="ts">
	import { formatLatencyMs, formatUsd } from '#lib/analytics';
	import { GatewayApiError } from '#lib/api';
	import DecisionBadge from '#lib/components/DecisionBadge.svelte';
	import EvidenceStrip from '#lib/components/EvidenceStrip.svelte';
	import Icon from '#lib/components/Icon.svelte';
	import { consoleState } from '#lib/state.svelte';
	import { simulateSyntheticEvaluateRisk, simulateSyntheticQuery } from '#lib/synthetic';
	import type { EvaluateRiskResponse, QueryResponse } from '#lib/types';

	let activeTab = $state<'query' | 'risk'>('query');

	// Tab A: POST /v1/query
	let queryInput = $state(
		'What was the weighted-average incremental borrowing rate and total undiscounted lease liability in Q3?'
	);
	let contextInput = $state(
		'Note 7 (Leases): Total undiscounted operating lease liabilities as of Q3 stood at $42.8M, discounted at a weighted-average incremental borrowing rate of 5.4% over a remaining term of 6.2 years.'
	);
	let bypassCache = $state(false);
	let applyQueryOverrides = $state(true);

	// Tab B: POST /v1/evaluate-risk
	let riskQueryInput = $state('What penalty applies for SLA shipment delays beyond 48 hours?');
	let riskContextInput = $state(
		'Section 11.3 (Service Credits): Late deliveries beyond 48 hours accrue a 1.5% invoice credit capped at 10% per calendar quarter. No monetary termination penalty is stipulated.'
	);
	let candidateAnswerInput = $state(
		'Missed SLA shipments incur a $25,000 fixed monetary penalty per incident plus immediate contract termination.'
	);
	let applyRiskAlphaOverride = $state(true);

	let submitting = $state(false);
	let errorState = $state<{ status: number; detail: string } | null>(null);
	let queryResult = $state<{ data: QueryResponse; synthetic: boolean } | null>(null);
	let riskResult = $state<{ data: EvaluateRiskResponse; synthetic: boolean } | null>(null);

	function applyQueryPreset(kind: 'grounded' | 'risky' | 'unscored') {
		if (kind === 'grounded') {
			queryInput =
				'What was the weighted-average incremental borrowing rate and total undiscounted lease liability in Q3?';
			contextInput =
				'Note 7 (Leases): Total undiscounted operating lease liabilities as of Q3 stood at $42.8M, discounted at a weighted-average incremental borrowing rate of 5.4% over a remaining term of 6.2 years.';
		} else if (kind === 'risky') {
			queryInput = 'How much is the guaranteed cash refund if the SLA is missed?';
			contextInput =
				'Section 11.3: Service delays earn only a 1.5% billing credit capped at 10% quarterly. Cash refunds and termination penalties are explicitly excluded.';
		} else {
			queryInput = 'Explain the difference between split conformal prediction and Platt scaling in 2 sentences.';
			contextInput = '';
		}
	}

	function applyRiskPreset(kind: 'faithful' | 'contradicted') {
		riskQueryInput = 'What penalty applies for SLA shipment delays beyond 48 hours?';
		riskContextInput =
			'Section 11.3 (Service Credits): Late deliveries beyond 48 hours accrue a 1.5% invoice credit capped at 10% per calendar quarter. No monetary termination penalty is stipulated.';
		candidateAnswerInput =
			kind === 'faithful'
				? 'Delays beyond 48 hours accrue a 1.5% invoice credit capped at 10% per quarter, with no monetary termination penalty.'
				: 'Missed SLA shipments incur a $25,000 fixed monetary penalty per incident plus immediate contract termination.';
	}

	async function submitQuery(useSynthetic = false) {
		errorState = null;
		const trimmedQuery = queryInput.trim();
		if (!trimmedQuery) {
			errorState = { status: 422, detail: 'Query must not be empty or whitespace-only.' };
			return;
		}

		submitting = true;
		try {
			if (useSynthetic) {
				queryResult = {
					data: simulateSyntheticQuery({
						query: trimmedQuery,
						context: contextInput.trim() || null,
						alpha: applyQueryOverrides ? consoleState.alpha : 0.1,
						lambdaCost: applyQueryOverrides ? consoleState.lambdaCost : 0.45
					}),
					synthetic: true
				};
				return;
			}

			const res = await consoleState.client().postQuery({
				query: trimmedQuery,
				context: contextInput.trim() ? contextInput.trim() : null,
				alpha_override: applyQueryOverrides ? consoleState.alpha : null,
				lambda_cost_override: applyQueryOverrides ? consoleState.lambdaCost : null,
				bypass_cache: bypassCache
			});
			queryResult = { data: res, synthetic: false };
			if (!consoleState.syntheticPreview) {
				void consoleState.refreshAll();
			}
		} catch (err) {
			if (err instanceof GatewayApiError) {
				errorState = { status: err.status, detail: err.detail };
			} else {
				errorState = { status: 0, detail: 'Unexpected network error calling POST /v1/query.' };
			}
		} finally {
			submitting = false;
		}
	}

	async function submitEvaluateRisk(useSynthetic = false) {
		errorState = null;
		const q = riskQueryInput.trim();
		const c = riskContextInput.trim();
		const a = candidateAnswerInput.trim();
		if (!q || !c || !a) {
			errorState = {
				status: 422,
				detail: 'Query, Grounding Context, and Candidate Answer are all required for risk scoring.'
			};
			return;
		}

		submitting = true;
		try {
			if (useSynthetic) {
				riskResult = {
					data: simulateSyntheticEvaluateRisk({
						context: c,
						candidate_answer: a,
						alpha: applyRiskAlphaOverride ? consoleState.alpha : 0.1
					}),
					synthetic: true
				};
				return;
			}

			const res = await consoleState.client().postEvaluateRisk({
				query: q,
				context: c,
				candidate_answer: a,
				alpha_override: applyRiskAlphaOverride ? consoleState.alpha : null
			});
			riskResult = { data: res, synthetic: false };
		} catch (err) {
			if (err instanceof GatewayApiError) {
				errorState = { status: err.status, detail: err.detail };
			} else {
				errorState = { status: 0, detail: 'Unexpected network error calling POST /v1/evaluate-risk.' };
			}
		} finally {
			submitting = false;
		}
	}
</script>

<svelte:head>
	<title>Live Query &amp; Risk Playground — Conformal LLM Router</title>
</svelte:head>

<div class="space-y-4">
	<div class="flex flex-wrap items-center justify-between gap-3 border-b border-[#DCD6C8] pb-3">
		<div>
			<h1 class="text-2xl font-bold tracking-[-0.02em] text-[#1C2721]">
				Live Query &amp; Risk Playground
			</h1>
			<p class="text-xs text-[#5C6860] mt-0.5 max-w-[72ch]">
				Exercise live semantic caching, learned Tier-1/Tier-2 routing, automatic escalation on abstention, and standalone MAPIE candidate scoring.
			</p>
		</div>

		<div class="inline-flex border border-[#DCD6C8] bg-[#FCFBF8] rounded-[2px] p-0.5" role="tablist" aria-label="Playground mode">
			<button
				type="button"
				role="tab"
				aria-selected={activeTab === 'query'}
				onclick={() => {
					activeTab = 'query';
					errorState = null;
				}}
				class="px-3 py-1.5 text-xs font-semibold rounded-[2px] transition-colors cursor-pointer {activeTab === 'query'
					? 'bg-[#1C2721] text-[#FAF8F3]'
					: 'text-[#5C6860] hover:text-[#1C2721]'}"
			>
				Route &amp; Guard (/v1/query)
			</button>
			<button
				type="button"
				role="tab"
				aria-selected={activeTab === 'risk'}
				onclick={() => {
					activeTab = 'risk';
					errorState = null;
				}}
				class="px-3 py-1.5 text-xs font-semibold rounded-[2px] transition-colors cursor-pointer {activeTab === 'risk'
					? 'bg-[#1C2721] text-[#FAF8F3]'
					: 'text-[#5C6860] hover:text-[#1C2721]'}"
			>
				Evaluate Risk (/v1/evaluate-risk)
			</button>
		</div>
	</div>

	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
		<!-- Left: Request Form -->
		<section class="specimen-panel p-4 space-y-3.5" aria-label="Playground request inputs">
			{#if activeTab === 'query'}
				<div class="flex flex-wrap items-center justify-between gap-2">
					<h2 class="text-sm font-bold text-[#1C2721]">POST /v1/query</h2>
					<div class="flex flex-wrap gap-1.5">
						<button
							type="button"
							onclick={() => applyQueryPreset('grounded')}
							class="px-2 py-0.5 text-[11px] font-medium border border-[#DCD6C8] bg-[#F6F3EC] rounded-[2px] hover:bg-[#EFECE3] cursor-pointer"
						>
							Preset: Grounded QA
						</button>
						<button
							type="button"
							onclick={() => applyQueryPreset('risky')}
							class="px-2 py-0.5 text-[11px] font-medium border border-[#DCD6C8] bg-[#F6F3EC] rounded-[2px] hover:bg-[#EFECE3] cursor-pointer"
						>
							Preset: Risky Query
						</button>
						<button
							type="button"
							onclick={() => applyQueryPreset('unscored')}
							class="px-2 py-0.5 text-[11px] font-medium border border-[#DCD6C8] bg-[#F6F3EC] rounded-[2px] hover:bg-[#EFECE3] cursor-pointer"
						>
							Preset: No Context
						</button>
					</div>
				</div>

				<form
					class="space-y-3"
					onsubmit={(e) => {
						e.preventDefault();
						void submitQuery(false);
					}}
				>
					<div>
						<label for="pg-query" class="block text-[11px] uppercase tracking-[0.06em] font-semibold text-[#5C6860] mb-1">
							Query <span class="text-[#B84A39]">*</span>
						</label>
						<textarea
							id="pg-query"
							rows="3"
							bind:value={queryInput}
							required
							class="w-full px-3 py-2 text-xs bg-[#F6F3EC] border border-[#DCD6C8] rounded-[2px] text-[#1C2721]"
						></textarea>
					</div>

					<div>
						<label for="pg-context" class="block text-[11px] uppercase tracking-[0.06em] font-semibold text-[#5C6860] mb-1">
							Grounding Context <span class="font-normal normal-case tracking-normal">(Optional — omit for UNSCORED routing)</span>
						</label>
						<textarea
							id="pg-context"
							rows="4"
							bind:value={contextInput}
							placeholder="Leave empty to run routing without conformal scoring…"
							class="w-full px-3 py-2 text-xs bg-[#F6F3EC] border border-[#DCD6C8] rounded-[2px] text-[#1C2721] placeholder:text-[#5C6860]"
						></textarea>
					</div>

					<div class="flex flex-col gap-2 pt-1 text-xs text-[#1C2721]">
						<label class="inline-flex items-center gap-2 cursor-pointer">
							<input type="checkbox" bind:checked={applyQueryOverrides} class="accent-[#236B52]" />
							<span>
								Apply simulation overrides (<code class="font-mono-data">α={consoleState.alpha.toFixed(2)}</code>, <code class="font-mono-data">λ_cost={consoleState.lambdaCost.toFixed(2)}</code>)
							</span>
						</label>
						<label class="inline-flex items-center gap-2 cursor-pointer">
							<input type="checkbox" bind:checked={bypassCache} class="accent-[#236B52]" />
							<span>Bypass semantic cache (<code class="font-mono-data">bypass_cache: true</code>)</span>
						</label>
					</div>

					<div class="flex flex-wrap items-center gap-2 pt-2">
						<button
							type="submit"
							disabled={submitting}
							class="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold bg-[#1C2721] text-[#FAF8F3] rounded-[2px] hover:bg-[#2A3A31] disabled:opacity-50 transition-colors cursor-pointer"
						>
							<Icon name="playground" size={14} />
							<span>{submitting ? 'Routing & Scoring…' : 'Run Routed Query'}</span>
						</button>

						{#if consoleState.syntheticPreview || consoleState.gatewayError?.unreachable}
							<button
								type="button"
								disabled={submitting}
								onclick={() => submitQuery(true)}
								class="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold border border-[#B46914] bg-[#FDF6E9] text-[#8F520E] rounded-[2px] hover:bg-[#FAECCB] cursor-pointer"
							>
								<Icon name="spark" size={14} />
								<span>Simulate [Synthetic]</span>
							</button>
						{/if}
					</div>
				</form>
			{:else}
				<div class="flex flex-wrap items-center justify-between gap-2">
					<h2 class="text-sm font-bold text-[#1C2721]">POST /v1/evaluate-risk</h2>
					<div class="flex flex-wrap gap-1.5">
						<button
							type="button"
							onclick={() => applyRiskPreset('faithful')}
							class="px-2 py-0.5 text-[11px] font-medium border border-[#DCD6C8] bg-[#F6F3EC] rounded-[2px] hover:bg-[#EFECE3] cursor-pointer"
						>
							Preset: Faithful
						</button>
						<button
							type="button"
							onclick={() => applyRiskPreset('contradicted')}
							class="px-2 py-0.5 text-[11px] font-medium border border-[#DCD6C8] bg-[#F6F3EC] rounded-[2px] hover:bg-[#EFECE3] cursor-pointer"
						>
							Preset: Contradicted
						</button>
					</div>
				</div>

				<form
					class="space-y-3"
					onsubmit={(e) => {
						e.preventDefault();
						void submitEvaluateRisk(false);
					}}
				>
					<div>
						<label for="risk-query" class="block text-[11px] uppercase tracking-[0.06em] font-semibold text-[#5C6860] mb-1">
							Query <span class="text-[#B84A39]">*</span>
						</label>
						<textarea
							id="risk-query"
							rows="2"
							bind:value={riskQueryInput}
							required
							class="w-full px-3 py-2 text-xs bg-[#F6F3EC] border border-[#DCD6C8] rounded-[2px] text-[#1C2721]"
						></textarea>
					</div>

					<div>
						<label for="risk-context" class="block text-[11px] uppercase tracking-[0.06em] font-semibold text-[#5C6860] mb-1">
							Grounding Context <span class="text-[#B84A39]">*</span>
						</label>
						<textarea
							id="risk-context"
							rows="3"
							bind:value={riskContextInput}
							required
							class="w-full px-3 py-2 text-xs bg-[#F6F3EC] border border-[#DCD6C8] rounded-[2px] text-[#1C2721]"
						></textarea>
					</div>

					<div>
						<label for="risk-candidate" class="block text-[11px] uppercase tracking-[0.06em] font-semibold text-[#5C6860] mb-1">
							Candidate Answer <span class="text-[#B84A39]">*</span>
						</label>
						<textarea
							id="risk-candidate"
							rows="3"
							bind:value={candidateAnswerInput}
							required
							class="w-full px-3 py-2 text-xs bg-[#F6F3EC] border border-[#DCD6C8] rounded-[2px] text-[#1C2721]"
						></textarea>
					</div>

					<label class="inline-flex items-center gap-2 text-xs text-[#1C2721] cursor-pointer">
						<input type="checkbox" bind:checked={applyRiskAlphaOverride} class="accent-[#236B52]" />
						<span>
							Apply simulation α override (<code class="font-mono-data">α_override={consoleState.alpha.toFixed(2)}</code>)
						</span>
					</label>

					<div class="flex flex-wrap items-center gap-2 pt-2">
						<button
							type="submit"
							disabled={submitting}
							class="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold bg-[#1C2721] text-[#FAF8F3] rounded-[2px] hover:bg-[#2A3A31] disabled:opacity-50 transition-colors cursor-pointer"
						>
							<Icon name="shield" size={14} />
							<span>{submitting ? 'Scoring Risk…' : 'Evaluate Conformal Risk'}</span>
						</button>

						{#if consoleState.syntheticPreview || consoleState.gatewayError?.unreachable}
							<button
								type="button"
								disabled={submitting}
								onclick={() => submitEvaluateRisk(true)}
								class="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold border border-[#B46914] bg-[#FDF6E9] text-[#8F520E] rounded-[2px] hover:bg-[#FAECCB] cursor-pointer"
							>
								<Icon name="spark" size={14} />
								<span>Simulate [Synthetic]</span>
							</button>
						{/if}
					</div>
				</form>
			{/if}

			{#if errorState}
				<div role="alert" class="border border-[#E8BEB8] bg-[#FBE9E7] text-[#8E3224] p-3 rounded-[2px] text-xs space-y-1">
					<div class="font-semibold">
						{errorState.status === 503
							? 'Gateway Returned 503 Service Unavailable'
							: errorState.status === 0
								? 'Gateway Unreachable'
								: `Request Rejected (HTTP ${errorState.status})`}
					</div>
					<p>{errorState.detail}</p>
				</div>
			{/if}
		</section>

		<!-- Right: Result Specimen Panel -->
		<section class="specimen-panel p-4 space-y-3.5" aria-label="Playground evaluation output">
			{#if activeTab === 'query'}
				{#if !queryResult}
					<div class="py-10 text-center text-xs text-[#5C6860] space-y-1.5">
						<p class="font-semibold text-[#1C2721]">No routed response yet</p>
						<p>Submit a prompt to inspect tier selection, provider fallback, calibrated risk score, and grounding evidence.</p>
					</div>
				{:else}
					{@const r = queryResult.data}
					<div class="flex flex-wrap items-center justify-between gap-2 border-b border-[#ECE7DC] pb-3">
						<div class="flex flex-wrap items-center gap-1.5">
							<DecisionBadge variant={r.conformal_decision} />
							<DecisionBadge variant={r.tier === 'tier1' ? 'TIER1' : 'TIER2'} label="{r.tier.toUpperCase()} · {r.provider}" />
							{#if r.escalated_from_tier1}
								<DecisionBadge variant="ESCALATED" label="Escalated T1→T2" />
							{/if}
							{#if r.cache_hit}
								<DecisionBadge variant="CACHE" label="Cache Hit" />
							{/if}
							{#if r.fallback_used}
								<DecisionBadge variant="FALLBACK" label="Provider Fallback" />
							{/if}
							{#if queryResult.synthetic}
								<DecisionBadge variant="SYNTHETIC" label="Synthetic" />
							{/if}
						</div>
						<span class="font-mono-data text-[11px] text-[#5C6860]">{r.request_id}</span>
					</div>

					<div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs border-b border-[#ECE7DC] pb-3">
						<div>
							<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Model Used</div>
							<div class="font-mono-data text-[#1C2721] truncate" title={r.model_used}>{r.model_used}</div>
						</div>
						<div>
							<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Risk Score</div>
							<div class="font-mono-data text-[#1C2721]">
								{r.calibrated_risk_score !== null ? r.calibrated_risk_score.toFixed(4) : 'UNSCORED'}
							</div>
						</div>
						<div>
							<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Est. Cost</div>
							<div class="font-mono-data text-[#1C2721]">{formatUsd(r.estimated_cost_usd)}</div>
						</div>
						<div>
							<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Latency · Params</div>
							<div class="font-mono-data text-[#1C2721]">
								{formatLatencyMs(r.latency_ms)} · α={r.alpha_used.toFixed(2)} λ={r.lambda_cost_used.toFixed(2)}
							</div>
						</div>
					</div>

					<div class="space-y-2.5 text-xs">
						<div>
							<div class="text-[11px] uppercase tracking-wider font-semibold text-[#5C6860] mb-1">
								Gateway Response
							</div>
							<div class="bg-[#F6F3EC] border border-[#DCD6C8] p-3 rounded-[2px] text-[#1C2721] leading-relaxed whitespace-pre-wrap">
								{r.answer}
							</div>
						</div>

						{#if r.draft_answer}
							<div>
								<div class="text-[11px] uppercase tracking-wider font-semibold text-[#9E3829] mb-1">
									Blocked Draft Answer (Held Back by Conformal Gate)
								</div>
								<div class="bg-[#FBE9E7] border border-[#E8BEB8] p-3 rounded-[2px] text-[#1C2721] leading-relaxed whitespace-pre-wrap">
									{r.draft_answer}
								</div>
							</div>
						{/if}
					</div>

					<div class="pt-1 border-t border-[#ECE7DC]">
						<EvidenceStrip
							evidence={r.evidence}
							riskScore={r.calibrated_risk_score}
							mode="detailed"
						/>
					</div>
				{/if}
			{:else}
				{#if !riskResult}
					<div class="py-10 text-center text-xs text-[#5C6860] space-y-1.5">
						<p class="font-semibold text-[#1C2721]">No risk score computed yet</p>
						<p>Score any <code class="font-mono-data">(query, context, candidate_answer)</code> tuple directly against the LightGBM + MAPIE conformal threshold without invoking an LLM.</p>
					</div>
				{:else}
					{@const er = riskResult.data}
					<div class="flex flex-wrap items-center justify-between gap-2 border-b border-[#ECE7DC] pb-3">
						<div class="flex items-center gap-2">
							<DecisionBadge variant={er.conformal_decision} />
							{#if riskResult.synthetic}
								<DecisionBadge variant="SYNTHETIC" label="Synthetic" />
							{/if}
						</div>
						<span class="font-mono-data text-xs text-[#5C6860]">
							α_used = {er.alpha_used.toFixed(2)} · τ = {er.threshold_used.toFixed(4)}
						</span>
					</div>

					<div class="grid grid-cols-3 gap-2 text-xs border-b border-[#ECE7DC] pb-3">
						<div>
							<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Calibrated Risk</div>
							<div class="font-mono-data text-lg font-bold {er.conformal_decision === 'ABSTAIN' ? 'text-[#B84A39]' : 'text-[#236B52]'}">
								{er.calibrated_risk_score.toFixed(4)}
							</div>
						</div>
						<div>
							<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Conformal Cutoff τ</div>
							<div class="font-mono-data text-lg font-bold text-[#1C2721]">
								{er.threshold_used.toFixed(4)}
							</div>
						</div>
						<div>
							<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Margin (τ − Risk)</div>
							<div class="font-mono-data text-lg font-bold text-[#1C2721]">
								{(er.threshold_used - er.calibrated_risk_score).toFixed(4)}
							</div>
						</div>
					</div>

					<EvidenceStrip
						evidence={er.evidence}
						riskScore={er.calibrated_risk_score}
						mode="detailed"
					/>
				{/if}
			{/if}
		</section>
	</div>
</div>
