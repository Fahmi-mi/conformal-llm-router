<script lang="ts">
	import type { EvidenceBreakdown } from '#lib/types';

	let {
		evidence,
		riskScore = null,
		mode = 'compact'
	}: {
		evidence: EvidenceBreakdown | null;
		riskScore?: number | null;
		mode?: 'compact' | 'detailed';
	} = $props();

	function clamp01(val: number | undefined | null): number {
		if (val === undefined || val === null || !Number.isFinite(val)) return 0;
		return Math.min(1, Math.max(0, val));
	}

	const threshold = $derived(evidence?.threshold_used ?? null);
	const effectiveRisk = $derived(riskScore);
	const exceedsThreshold = $derived(
		effectiveRisk !== null && threshold !== null ? effectiveRisk >= threshold : false
	);
</script>

{#if !evidence}
	<span class="text-xs text-[#5C6860] italic">Unscored (no context)</span>
{:else if mode === 'compact'}
	<div
		class="inline-flex items-center gap-2"
		title="NLI Entailment {(evidence.nli_entailment * 100).toFixed(0)}% · Contradiction {(evidence.nli_contradiction * 100).toFixed(0)}% · Cosine {(evidence.cosine_sim * 100).toFixed(0)}% · Entity {(evidence.entity_overlap * 100).toFixed(0)}%"
	>
		<div class="flex items-center gap-1" aria-label="Evidence signal bars">
			<span class="block w-1.5 h-3.5 rounded-[1px] bg-[#DCD6C8] overflow-hidden relative" title="NLI Entailment">
				<span
					class="absolute bottom-0 inset-x-0 bg-[#236B52]"
					style="height: {Math.round(clamp01(evidence.nli_entailment) * 100)}%"
				></span>
			</span>
			<span class="block w-1.5 h-3.5 rounded-[1px] bg-[#DCD6C8] overflow-hidden relative" title="NLI Contradiction">
				<span
					class="absolute bottom-0 inset-x-0 bg-[#B84A39]"
					style="height: {Math.round(clamp01(evidence.nli_contradiction) * 100)}%"
				></span>
			</span>
			<span class="block w-1.5 h-3.5 rounded-[1px] bg-[#DCD6C8] overflow-hidden relative" title="Cosine Similarity">
				<span
					class="absolute bottom-0 inset-x-0 bg-[#236B52]"
					style="height: {Math.round(clamp01(evidence.cosine_sim) * 100)}%"
				></span>
			</span>
			<span class="block w-1.5 h-3.5 rounded-[1px] bg-[#DCD6C8] overflow-hidden relative" title="Entity Overlap">
				<span
					class="absolute bottom-0 inset-x-0 bg-[#1C2721]"
					style="height: {Math.round(clamp01(evidence.entity_overlap) * 100)}%"
				></span>
			</span>
		</div>
		<span class="font-mono-data text-[11px] text-[#5C6860]">
			E:{evidence.nli_entailment.toFixed(2)} C:{evidence.nli_contradiction.toFixed(2)}
		</span>
	</div>
{:else}
	<div class="space-y-3">
		<div class="grid grid-cols-2 sm:grid-cols-4 gap-px bg-[#DCD6C8] border border-[#DCD6C8] rounded-[2px] overflow-hidden">
			<div class="bg-[#F6F3EC] px-3 py-2">
				<div class="text-[11px] uppercase tracking-wider text-[#5C6860] font-semibold">NLI Entailment</div>
				<div class="font-mono-data text-base font-bold text-[#236B52] mt-0.5">
					{(evidence.nli_entailment * 100).toFixed(1)}%
				</div>
				<div class="h-1.5 bg-[#E3DDD0] rounded-[1px] mt-1.5 overflow-hidden">
					<div class="h-full bg-[#236B52]" style="width: {clamp01(evidence.nli_entailment) * 100}%"></div>
				</div>
			</div>

			<div class="bg-[#F6F3EC] px-3 py-2">
				<div class="text-[11px] uppercase tracking-wider text-[#5C6860] font-semibold">NLI Contradiction</div>
				<div class="font-mono-data text-base font-bold text-[#B84A39] mt-0.5">
					{(evidence.nli_contradiction * 100).toFixed(1)}%
				</div>
				<div class="h-1.5 bg-[#E3DDD0] rounded-[1px] mt-1.5 overflow-hidden">
					<div class="h-full bg-[#B84A39]" style="width: {clamp01(evidence.nli_contradiction) * 100}%"></div>
				</div>
			</div>

			<div class="bg-[#F6F3EC] px-3 py-2">
				<div class="text-[11px] uppercase tracking-wider text-[#5C6860] font-semibold">Cosine Sim</div>
				<div class="font-mono-data text-base font-bold text-[#1C2721] mt-0.5">
					{evidence.cosine_sim.toFixed(3)}
				</div>
				<div class="h-1.5 bg-[#E3DDD0] rounded-[1px] mt-1.5 overflow-hidden">
					<div class="h-full bg-[#236B52]" style="width: {clamp01(evidence.cosine_sim) * 100}%"></div>
				</div>
			</div>

			<div class="bg-[#F6F3EC] px-3 py-2">
				<div class="text-[11px] uppercase tracking-wider text-[#5C6860] font-semibold">Entity Overlap</div>
				<div class="font-mono-data text-base font-bold text-[#1C2721] mt-0.5">
					{(evidence.entity_overlap * 100).toFixed(1)}%
				</div>
				<div class="h-1.5 bg-[#E3DDD0] rounded-[1px] mt-1.5 overflow-hidden">
					<div class="h-full bg-[#1C2721]" style="width: {clamp01(evidence.entity_overlap) * 100}%"></div>
				</div>
			</div>
		</div>

		{#if effectiveRisk !== null || threshold !== null}
			<div class="pt-1">
				<div class="flex flex-wrap items-center justify-between gap-2 text-xs mb-1.5">
					<span class="font-semibold text-[#1C2721]">
						Conformal Risk vs. α-Cutoff
						<span class="font-normal text-[#5C6860]">(Length ratio: {evidence.length_ratio.toFixed(2)} · NLI neutral: {(evidence.nli_neutral * 100).toFixed(1)}%)</span>
					</span>
					<span class="font-mono-data font-semibold {exceedsThreshold ? 'text-[#B84A39]' : 'text-[#236B52]'}">
						Risk {effectiveRisk !== null ? effectiveRisk.toFixed(4) : '—'}
						{threshold !== null ? (exceedsThreshold ? '≥' : '<') + ` τ ${threshold.toFixed(4)}` : ''}
					</span>
				</div>
				<div class="relative h-2.5 bg-[#EFECE3] border border-[#DCD6C8] rounded-[2px]">
					{#if effectiveRisk !== null}
						<div
							class="h-full transition-all duration-200 {exceedsThreshold ? 'bg-[#B84A39]' : 'bg-[#236B52]'}"
							style="width: {clamp01(effectiveRisk) * 100}%"
						></div>
					{/if}
					{#if threshold !== null}
						<div
							class="absolute top-[-3px] bottom-[-3px] w-0.5 bg-[#1C2721]"
							style="left: {clamp01(threshold) * 100}%"
							title="Conformal cutoff τ = {threshold.toFixed(4)}"
						></div>
					{/if}
				</div>
			</div>
		{/if}
	</div>
{/if}
