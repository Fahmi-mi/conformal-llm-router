<script lang="ts">
	import { consoleState } from '#lib/state.svelte';
	import DecisionBadge from './DecisionBadge.svelte';
	import Icon from './Icon.svelte';

	function handleAlphaInput(event: Event) {
		const target = event.currentTarget as HTMLInputElement;
		consoleState.setAlpha(Number(target.value));
	}

	function handleLambdaInput(event: Event) {
		const target = event.currentTarget as HTMLInputElement;
		consoleState.setLambdaCost(Number(target.value));
	}
</script>

<section
	aria-label="Dual-mode risk and routing simulation controls"
	class="specimen-panel px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-3"
>
	<div class="flex items-center gap-3 min-w-[220px]">
		<label for="global-alpha-slider" class="text-[13px] font-semibold text-[#1C2721] whitespace-nowrap">
			Target Risk α:
			<span class="font-mono-data text-[#B84A39] ml-1">{consoleState.alpha.toFixed(2)}</span>
		</label>
		<input
			id="global-alpha-slider"
			type="range"
			min="0.01"
			max="0.30"
			step="0.01"
			value={consoleState.alpha}
			oninput={handleAlphaInput}
			class="specimen-slider w-32 sm:w-36"
			aria-label="Target hallucination risk alpha"
		/>
	</div>

	<div class="flex items-center gap-3 min-w-[235px]">
		<label for="global-lambda-slider" class="text-[13px] font-semibold text-[#1C2721] whitespace-nowrap">
			Cost Weight λ_cost:
			<span class="font-mono-data text-[#236B52] ml-1">{consoleState.lambdaCost.toFixed(2)}</span>
		</label>
		<input
			id="global-lambda-slider"
			type="range"
			min="0.00"
			max="1.00"
			step="0.05"
			value={consoleState.lambdaCost}
			oninput={handleLambdaInput}
			class="specimen-slider w-32 sm:w-36"
			aria-label="Router cost weight lambda_cost"
		/>
	</div>

	<div class="ml-auto flex flex-wrap items-center gap-2.5">
		<span class="hidden xl:inline text-xs text-[#5C6860]">
			Dual-mode simulation · Server defaults preserved
		</span>

		<button
			type="button"
			onclick={() => consoleState.toggleSyntheticPreview()}
			aria-pressed={consoleState.syntheticPreview}
			class="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold border rounded-[2px] transition-colors cursor-pointer {consoleState.syntheticPreview
				? 'bg-[#FDF6E9] text-[#8F520E] border-[#B46914] hover:bg-[#FAECCB]'
				: 'bg-[#F6F3EC] text-[#1C2721] border-[#DCD6C8] hover:bg-[#EFECE3]'}"
		>
			<Icon name="spark" size={14} />
			<span>{consoleState.syntheticPreview ? 'Sample Data Active' : 'Preview Sample Data'}</span>
			<DecisionBadge variant="SYNTHETIC" label="Synthetic" />
		</button>

		<button
			type="button"
			onclick={() => consoleState.refreshAll()}
			disabled={consoleState.refreshing}
			class="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold border border-[#DCD6C8] bg-[#FCFBF8] text-[#1C2721] rounded-[2px] hover:bg-[#EFECE3] disabled:opacity-50 transition-colors cursor-pointer"
		>
			<Icon name="refresh" size={14} class={consoleState.refreshing ? 'animate-spin' : ''} />
			<span>Refresh</span>
		</button>
	</div>
</section>
