<script lang="ts">
	import { untrack } from 'svelte';
	import {
		computeNextAuditOffset,
		filterSyntheticAuditRecords,
		formatLatencyMs,
		formatSpecimenId,
		formatTimestamp,
		formatUsd,
		resolveSelectedAuditRecord
	} from '#lib/analytics';
	import { GatewayApiError } from '#lib/api';
	import DecisionBadge from '#lib/components/DecisionBadge.svelte';
	import EvidenceStrip from '#lib/components/EvidenceStrip.svelte';
	import Icon from '#lib/components/Icon.svelte';
	import { consoleState } from '#lib/state.svelte';
	import { SYNTHETIC_AUDIT_RECORDS } from '#lib/synthetic';
	import type { AuditRecord, ConformalDecision, RouterTier } from '#lib/types';

	const PAGE_SIZE = 25;

	let decisionFilter = $state<ConformalDecision | ''>('');
	let tierFilter = $state<RouterTier | ''>('');
	let cacheFilter = $state<'all' | 'hit' | 'miss'>('all');
	let searchQuery = $state('');
	let offset = $state(0);

	let loading = $state(true);
	let errorMsg = $state<string | null>(null);
	let items = $state<AuditRecord[]>([]);
	let total = $state(0);
	let selectedRecord = $state<AuditRecord | null>(null);

	let debounceTimer: ReturnType<typeof setTimeout> | null = null;
	let requestSeq = 0;

	async function loadAudit(resetOffset = false) {
		if (debounceTimer) {
			clearTimeout(debounceTimer);
			debounceTimer = null;
		}
		if (resetOffset) offset = 0;
		const reqId = ++requestSeq;
		loading = true;
		errorMsg = null;

		const cacheHitParam =
			cacheFilter === 'hit' ? true : cacheFilter === 'miss' ? false : null;

		if (consoleState.syntheticPreview) {
			const filtered = filterSyntheticAuditRecords(SYNTHETIC_AUDIT_RECORDS, {
				decision: decisionFilter || null,
				tier: tierFilter || null,
				cache_hit: cacheHitParam,
				q: searchQuery
			});
			if (reqId !== requestSeq) return;
			total = filtered.length;
			items = filtered.slice(offset, offset + PAGE_SIZE);
			selectedRecord = resolveSelectedAuditRecord(items, selectedRecord);
			loading = false;
			return;
		}

		try {
			const res = await consoleState.client().getAuditLogs({
				limit: PAGE_SIZE,
				offset,
				decision: decisionFilter || null,
				tier: tierFilter || null,
				cache_hit: cacheHitParam,
				q: searchQuery || null
			});
			if (reqId !== requestSeq) return;
			items = res.items;
			total = res.total;
			selectedRecord = resolveSelectedAuditRecord(items, selectedRecord);
		} catch (err) {
			if (reqId !== requestSeq) return;
			items = [];
			total = 0;
			selectedRecord = null;
			errorMsg =
				err instanceof GatewayApiError
					? err.detail
					: 'Unable to fetch audit records from gateway.';
		} finally {
			if (reqId === requestSeq) {
				loading = false;
			}
		}
	}

	function scheduleSearch() {
		if (debounceTimer) clearTimeout(debounceTimer);
		debounceTimer = setTimeout(() => {
			debounceTimer = null;
			void loadAudit(true);
		}, 220);
	}

	function clearFilters() {
		decisionFilter = '';
		tierFilter = '';
		cacheFilter = 'all';
		searchQuery = '';
		void loadAudit(true);
	}

	$effect(() => {
		const _isSynthetic = consoleState.syntheticPreview;
		void _isSynthetic;
		untrack(() => {
			void loadAudit(true);
		});
	});
</script>

<svelte:head>
	<title>Audit Log Inspector — Conformal LLM Router</title>
</svelte:head>

<div class="space-y-4">
	<div class="flex flex-wrap items-center justify-between gap-3 border-b border-[#DCD6C8] pb-3">
		<div>
			<h1 class="text-2xl font-bold tracking-[-0.02em] text-[#1C2721]">Audit Log Inspector</h1>
			<p class="text-xs text-[#5C6860] mt-0.5 max-w-[72ch]">
				Inspect individual gateway specimens, grounding contexts, blocked abstention drafts, and conformal evidence strips.
			</p>
		</div>
		<div class="flex items-center gap-2 text-xs font-mono-data text-[#5C6860]">
			{#if consoleState.syntheticPreview}
				<DecisionBadge variant="SYNTHETIC" label="Synthetic" />
			{/if}
			<span>{total} specimen{total === 1 ? '' : 's'} matched</span>
		</div>
	</div>

	<!-- Filter Controls -->
	<section aria-label="Audit log filters" class="specimen-panel p-3.5 flex flex-wrap items-end gap-3">
		<div class="flex-1 min-w-[200px]">
			<label for="audit-search-input" class="block text-[11px] uppercase tracking-[0.06em] font-semibold text-[#5C6860] mb-1">
				Search Query Text
			</label>
			<input
				id="audit-search-input"
				type="search"
				bind:value={searchQuery}
				oninput={scheduleSearch}
				placeholder="Filter by prompt keywords…"
				class="w-full px-3 py-1.5 text-xs bg-[#F6F3EC] border border-[#DCD6C8] rounded-[2px] text-[#1C2721] placeholder:text-[#5C6860]"
			/>
		</div>

		<div class="min-w-[140px]">
			<label for="audit-decision-select" class="block text-[11px] uppercase tracking-[0.06em] font-semibold text-[#5C6860] mb-1">
				Decision
			</label>
			<select
				id="audit-decision-select"
				bind:value={decisionFilter}
				onchange={() => loadAudit(true)}
				class="w-full px-2.5 py-1.5 text-xs bg-[#F6F3EC] border border-[#DCD6C8] rounded-[2px] text-[#1C2721]"
			>
				<option value="">All Decisions</option>
				<option value="PASS">PASS</option>
				<option value="ABSTAIN">ABSTAIN</option>
				<option value="UNSCORED">UNSCORED</option>
			</select>
		</div>

		<div class="min-w-[130px]">
			<label for="audit-tier-select" class="block text-[11px] uppercase tracking-[0.06em] font-semibold text-[#5C6860] mb-1">
				Routed Tier
			</label>
			<select
				id="audit-tier-select"
				bind:value={tierFilter}
				onchange={() => loadAudit(true)}
				class="w-full px-2.5 py-1.5 text-xs bg-[#F6F3EC] border border-[#DCD6C8] rounded-[2px] text-[#1C2721]"
			>
				<option value="">All Tiers</option>
				<option value="tier1">Tier 1 (Small/Fast)</option>
				<option value="tier2">Tier 2 (Frontier)</option>
			</select>
		</div>

		<div class="min-w-[140px]">
			<label for="audit-cache-select" class="block text-[11px] uppercase tracking-[0.06em] font-semibold text-[#5C6860] mb-1">
				Cache State
			</label>
			<select
				id="audit-cache-select"
				bind:value={cacheFilter}
				onchange={() => loadAudit(true)}
				class="w-full px-2.5 py-1.5 text-xs bg-[#F6F3EC] border border-[#DCD6C8] rounded-[2px] text-[#1C2721]"
			>
				<option value="all">All Traffic</option>
				<option value="hit">Cache Hits Only</option>
				<option value="miss">Fresh Inference Only</option>
			</select>
		</div>

		<button
			type="button"
			onclick={clearFilters}
			class="px-3 py-1.5 text-xs font-semibold border border-[#DCD6C8] bg-[#FCFBF8] text-[#1C2721] rounded-[2px] hover:bg-[#EFECE3] cursor-pointer"
		>
			Reset
		</button>
	</section>

	<!-- Table + Detail Inspector -->
	<div class="grid grid-cols-1 xl:grid-cols-[1fr_430px] gap-4 items-start">
		<section class="specimen-panel p-4 min-w-0">
			{#if loading}
				<div class="space-y-2.5 animate-pulse py-2">
					{#each [0, 1, 2, 3, 4] as idx}
						<div class="h-9 bg-[#EFECE3] rounded-[2px]" data-skeleton={idx}></div>
					{/each}
				</div>
			{:else if errorMsg}
				<div class="border-t border-[#E8BEB8] bg-[#FBE9E7] p-4 text-xs text-[#8E3224]">
					<strong>Audit query failed:</strong> {errorMsg}
				</div>
			{:else if items.length === 0}
				<div class="border-t border-[#ECE7DC] bg-[#F6F3EC] p-6 text-center space-y-2">
					<p class="text-sm font-semibold text-[#1C2721]">No matching specimen records</p>
					<p class="text-xs text-[#5C6860]">
						Adjust your filters, execute a query in <a href="/playground" class="font-semibold text-[#236B52] hover:underline">Live Playground</a>, or enable Sample Preview.
					</p>
				</div>
			{:else}
				<div class="overflow-x-auto">
					<table class="w-full min-w-[740px] table-fixed border-collapse text-[13px]">
						<colgroup>
							<col class="w-[106px]" />
							<col class="w-[220px]" />
							<col class="w-[140px]" />
							<col class="w-[74px]" />
							<col class="w-[160px]" />
							<col class="w-[92px]" />
						</colgroup>
						<thead>
							<tr class="border-b border-[#DCD6C8] text-[11px] uppercase tracking-[0.06em] text-[#5C6860]">
								<th class="text-left py-2 px-2 font-semibold">Specimen</th>
								<th class="text-left py-2 px-2 font-semibold">Query Summary</th>
								<th class="text-left py-2 px-2 font-semibold">Tier · Flags</th>
								<th class="text-left py-2 px-2 font-semibold">Risk</th>
								<th class="text-left py-2 px-2 font-semibold">Evidence</th>
								<th class="text-left py-2 px-2 font-semibold">Decision</th>
							</tr>
						</thead>
						<tbody>
							{#each items as row (row.request_id)}
								{@const isSelected = selectedRecord?.request_id === row.request_id}
								<tr
									onclick={() => (selectedRecord = row)}
									class="border-b border-[#ECE7DC] cursor-pointer transition-colors {isSelected
										? 'bg-[#E4F2EC]/60'
										: 'hover:bg-[#F6F3EC]'}"
								>
									<td class="py-2.5 px-2 font-mono-data text-xs font-semibold text-[#1C2721]">
										{formatSpecimenId(row.request_id)}
									</td>
									<td class="py-2.5 px-2 truncate text-[#1C2721]" title={row.query_text}>
										{row.query_text}
									</td>
									<td class="py-2.5 px-2 text-xs">
										<span class="font-semibold uppercase text-[#1C2721]">{row.tier}</span>
										<span class="text-[#5C6860]"> · {row.provider}</span>
										{#if row.escalated_from_tier1}
											<span class="ml-1 text-[11px] font-semibold text-[#9E3829]">ESC</span>
										{/if}
										{#if row.cache_hit}
											<span class="ml-1 text-[11px] font-semibold text-[#1B5642]">HIT</span>
										{/if}
										{#if row.fallback_used}
											<span class="ml-1 text-[11px] font-semibold text-[#8F520E]">FB</span>
										{/if}
									</td>
									<td class="py-2.5 px-2 font-mono-data text-xs">
										{row.calibrated_risk_score !== null ? row.calibrated_risk_score.toFixed(3) : '—'}
									</td>
									<td class="py-2.5 px-2">
										<EvidenceStrip evidence={row.evidence} riskScore={row.calibrated_risk_score} mode="compact" />
									</td>
									<td class="py-2.5 px-2">
										<DecisionBadge variant={row.conformal_decision} />
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>

				<!-- Pagination -->
				<div class="flex items-center justify-between pt-3 mt-2 border-t border-[#ECE7DC] text-xs text-[#5C6860]">
					<span class="font-mono-data">
						Showing {offset + 1}–{Math.min(offset + items.length, total)} of {total}
					</span>
					<div class="flex items-center gap-2">
						<button
							type="button"
							disabled={offset === 0}
							onclick={() => {
								offset = computeNextAuditOffset(offset, PAGE_SIZE, total, 'prev');
								void loadAudit(false);
							}}
							class="px-2.5 py-1 border border-[#DCD6C8] bg-[#FCFBF8] text-[#1C2721] rounded-[2px] disabled:opacity-45 cursor-pointer"
						>
							Previous
						</button>
						<button
							type="button"
							disabled={offset + items.length >= total}
							onclick={() => {
								offset = computeNextAuditOffset(offset, PAGE_SIZE, total, 'next');
								void loadAudit(false);
							}}
							class="px-2.5 py-1 border border-[#DCD6C8] bg-[#FCFBF8] text-[#1C2721] rounded-[2px] disabled:opacity-45 cursor-pointer"
						>
							Next
						</button>
					</div>
				</div>
			{/if}
		</section>

		<!-- Specimen Detail Drawer / Panel -->
		<aside class="specimen-panel p-4 space-y-3.5" aria-label="Selected specimen detail">
			{#if !selectedRecord}
				<div class="py-8 text-center text-xs text-[#5C6860]">
					Select any row in the audit table to inspect its full prompt, retrieval context, response, and grounding evidence.
				</div>
			{:else}
				<div class="flex flex-wrap items-start justify-between gap-2 border-b border-[#ECE7DC] pb-3">
					<div>
						<div class="font-mono-data text-sm font-bold text-[#1C2721]">
							{formatSpecimenId(selectedRecord.request_id)}
						</div>
						<div class="font-mono-data text-[11px] text-[#5C6860] mt-0.5 break-all">
							{selectedRecord.request_id} · {formatTimestamp(selectedRecord.created_at)}
						</div>
					</div>
					<div class="flex flex-wrap gap-1.5">
						<DecisionBadge variant={selectedRecord.conformal_decision} />
						<DecisionBadge
							variant={selectedRecord.tier === 'tier1' ? 'TIER1' : 'TIER2'}
							label="{selectedRecord.tier.toUpperCase()} · {selectedRecord.provider}"
						/>
						{#if selectedRecord.escalated_from_tier1}
							<DecisionBadge variant="ESCALATED" />
						{/if}
						{#if selectedRecord.cache_hit}
							<DecisionBadge variant="CACHE" label="CACHE HIT" />
						{/if}
						{#if selectedRecord.fallback_used}
							<DecisionBadge variant="FALLBACK" />
						{/if}
					</div>
				</div>

				<div class="grid grid-cols-3 gap-2 text-xs border-b border-[#ECE7DC] pb-3">
					<div>
						<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Model</div>
						<div class="font-mono-data text-[#1C2721] truncate" title={selectedRecord.model_used}>
							{selectedRecord.model_used}
						</div>
					</div>
					<div>
						<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Cost / Frontier</div>
						<div class="font-mono-data text-[#1C2721]">
							{formatUsd(selectedRecord.estimated_cost_usd)} / {formatUsd(selectedRecord.frontier_cost_usd)}
						</div>
					</div>
					<div>
						<div class="text-[11px] uppercase text-[#5C6860] font-semibold">Latency · Params</div>
						<div class="font-mono-data text-[#1C2721]">
							{formatLatencyMs(selectedRecord.latency_ms)} · α={selectedRecord.alpha_used.toFixed(2)} λ={selectedRecord.lambda_cost_used.toFixed(2)}
						</div>
					</div>
				</div>

				<div class="space-y-2.5 text-xs">
					<div>
						<div class="text-[11px] uppercase tracking-wider font-semibold text-[#5C6860] mb-1">
							Query Text
						</div>
						<div class="bg-[#F6F3EC] border border-[#DCD6C8] p-2.5 rounded-[2px] text-[#1C2721] leading-relaxed whitespace-pre-wrap">
							{selectedRecord.query_text}
						</div>
					</div>

					<div>
						<div class="text-[11px] uppercase tracking-wider font-semibold text-[#5C6860] mb-1">
							Retrieval Context
						</div>
						{#if selectedRecord.context_text}
							<div class="bg-[#F6F3EC] border border-[#DCD6C8] p-2.5 rounded-[2px] text-[#1C2721] leading-relaxed whitespace-pre-wrap max-h-36 overflow-y-auto">
								{selectedRecord.context_text}
							</div>
						{:else}
							<div class="bg-[#EFECE3] border border-[#DCD6C8] p-2.5 rounded-[2px] text-[#5C6860] italic">
								No retrieval context provided — request was routed without conformal scoring (UNSCORED).
							</div>
						{/if}
					</div>

					<div>
						<div class="text-[11px] uppercase tracking-wider font-semibold text-[#5C6860] mb-1">
							Delivered Answer
						</div>
						<div class="bg-[#FCFBF8] border border-[#DCD6C8] p-2.5 rounded-[2px] text-[#1C2721] leading-relaxed whitespace-pre-wrap">
							{selectedRecord.answer_text}
						</div>
					</div>

					{#if selectedRecord.draft_answer}
						<div>
							<div class="flex items-center gap-1.5 text-[11px] uppercase tracking-wider font-semibold text-[#9E3829] mb-1">
								<Icon name="shield" size={13} />
								<span>Blocked Draft Answer (Abstained by MAPIE Gate)</span>
							</div>
							<div class="bg-[#FBE9E7] border border-[#E8BEB8] p-2.5 rounded-[2px] text-[#1C2721] leading-relaxed whitespace-pre-wrap">
								{selectedRecord.draft_answer}
							</div>
						</div>
					{/if}
				</div>

				<div class="pt-1 border-t border-[#ECE7DC]">
					<EvidenceStrip
						evidence={selectedRecord.evidence}
						riskScore={selectedRecord.calibrated_risk_score}
						mode="detailed"
					/>
				</div>
			{/if}
		</aside>
	</div>
</div>
