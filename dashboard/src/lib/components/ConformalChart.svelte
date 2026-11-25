<script lang="ts">
	import { findNearestConformalPoint } from '#lib/analytics';
	import type { ConformalSweepPoint } from '#lib/types';

	let {
		sweep = [],
		logisticSweep = [],
		activeAlpha = 0.1,
		compact = false,
		showLogistic = true,
		showAbstention = false,
		onSelectAlpha
	}: {
		sweep?: ConformalSweepPoint[];
		logisticSweep?: ConformalSweepPoint[];
		activeAlpha?: number;
		compact?: boolean;
		showLogistic?: boolean;
		showAbstention?: boolean;
		onSelectAlpha?: (alpha: number) => void;
	} = $props();

	let hoveredPoint = $state<ConformalSweepPoint | null>(null);

	const W = $derived(compact ? 540 : 720);
	const H = $derived(compact ? 225 : 340);
	const pad = $derived(
		compact
			? { top: 20, right: 20, bottom: 38, left: 48 }
			: { top: 26, right: showAbstention ? 56 : 28, bottom: 48, left: 58 }
	);

	const plotW = $derived(W - pad.left - pad.right);
	const plotH = $derived(H - pad.top - pad.bottom);

	const sortedSweep = $derived([...sweep].sort((a, b) => a.alpha - b.alpha));
	const sortedLogistic = $derived([...logisticSweep].sort((a, b) => a.alpha - b.alpha));

	const maxY = $derived.by(() => {
		const maxFar = Math.max(
			0.3,
			...sortedSweep.map((p) => p.false_accept_rate),
			...sortedLogistic.map((p) => p.false_accept_rate)
		);
		return Math.min(0.4, Math.ceil(maxFar * 20) / 20);
	});

	function xPx(alpha: number): number {
		const clamped = Math.min(0.3, Math.max(0.0, alpha));
		return pad.left + (clamped / 0.3) * plotW;
	}

	function yPx(rate: number): number {
		const norm = Math.min(1, Math.max(0, rate / maxY));
		return pad.top + (1 - norm) * plotH;
	}

	function yAbstainPx(rate: number): number {
		const norm = Math.min(1, Math.max(0, rate));
		return pad.top + (1 - norm) * plotH;
	}

	const lgbmPath = $derived(
		sortedSweep
			.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${xPx(p.alpha).toFixed(1)} ${yPx(p.false_accept_rate).toFixed(1)}`)
			.join(' ')
	);

	const logisticPath = $derived(
		sortedLogistic
			.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${xPx(p.alpha).toFixed(1)} ${yPx(p.false_accept_rate).toFixed(1)}`)
			.join(' ')
	);

	const abstentionPath = $derived(
		sortedSweep
			.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${xPx(p.alpha).toFixed(1)} ${yAbstainPx(p.abstention_rate).toFixed(1)}`)
			.join(' ')
	);

	const activePoint = $derived(findNearestConformalPoint(sortedSweep, activeAlpha));
	const displayPoint = $derived(hoveredPoint ?? activePoint);

	const xTicks = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
	const yTicks = $derived(compact ? [0, 0.1, 0.2, 0.3] : [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]);
</script>

{#if sortedSweep.length === 0}
	<div class="border-t border-[#ECE7DC] bg-[#F6F3EC] p-6 text-center text-xs text-[#5C6860]">
		Conformal sweep artifact unavailable (TBD). Run Phase 2 calibration or enable Sample Preview.
	</div>
{:else}
	<div class="w-full">
		<svg
			viewBox="0 0 {W} {H}"
			class="w-full h-auto block select-none"
			role="img"
			aria-label="Conformal Risk Calibration across alpha 0.01 to 0.30"
		>
			<!-- Gridlines -->
			{#each yTicks as tick}
				{@const y = yPx(tick)}
				<line x1={pad.left} y1={y} x2={W - pad.right} y2={y} stroke="#ECE7DC" stroke-width="1" />
				<text
					x={pad.left - 8}
					y={y + 3.5}
					text-anchor="end"
					font-size="10.5"
					fill="#5C6860"
					class="font-mono-data"
				>
					{(tick * 100).toFixed(0)}%
				</text>
			{/each}

			{#each xTicks as tick}
				{@const x = xPx(tick)}
				<line x1={x} y1={pad.top} x2={x} y2={H - pad.bottom} stroke="#ECE7DC" stroke-width="1" />
				<text
					x={x}
					y={H - pad.bottom + 16}
					text-anchor="middle"
					font-size="10.5"
					fill="#5C6860"
					class="font-mono-data"
				>
					{tick.toFixed(2)}
				</text>
			{/each}

			<!-- Secondary right axis for Abstention Rate -->
			{#if !compact && showAbstention}
				{#each [0, 0.25, 0.5, 0.75, 1.0] as tick}
					<text
						x={W - pad.right + 8}
						y={yAbstainPx(tick) + 3.5}
						text-anchor="start"
						font-size="10.5"
						fill="#236B52"
						class="font-mono-data"
					>
						{(tick * 100).toFixed(0)}%
					</text>
				{/each}
			{/if}

			<!-- Axes -->
			<line x1={pad.left} y1={H - pad.bottom} x2={W - pad.right} y2={H - pad.bottom} stroke="#CFC7B6" stroke-width="1.25" />
			<line x1={pad.left} y1={pad.top} x2={pad.left} y2={H - pad.bottom} stroke="#CFC7B6" stroke-width="1.25" />

			<!-- Labels -->
			<text x={pad.left + plotW / 2} y={H - 6} text-anchor="middle" font-size="11" fill="#5C6860">
				Target Risk Level α (MAPIE Split Conformal)
			</text>
			{#if !compact}
				<text
					x={14}
					y={pad.top + plotH / 2}
					text-anchor="middle"
					font-size="11"
					fill="#5C6860"
					transform="rotate(-90 14 {pad.top + plotH / 2})"
				>
					Empirical False-Accept Rate
				</text>
			{/if}

			<!-- Theoretical risk bound y = alpha -->
			<line
				x1={xPx(0)}
				y1={yPx(0)}
				x2={xPx(0.3)}
				y2={yPx(0.3)}
				stroke="#8A948D"
				stroke-dasharray="5 5"
				stroke-width="1.5"
			/>
			<text x={xPx(0.27)} y={yPx(0.285)} text-anchor="end" font-size="10.5" fill="#5C6860">
				Bound: FAR = α
			</text>

			<!-- Logistic sweep comparison -->
			{#if showLogistic && logisticPath}
				<path
					d={logisticPath}
					fill="none"
					stroke="#5C6860"
					stroke-width="1.75"
					stroke-dasharray="4 3"
					stroke-linejoin="round"
				/>
			{/if}

			<!-- Abstention overlay -->
			{#if !compact && showAbstention && abstentionPath}
				<path
					d={abstentionPath}
					fill="none"
					stroke="#236B52"
					stroke-width="2"
					stroke-dasharray="2 2"
					stroke-linejoin="round"
				/>
			{/if}

			<!-- LightGBM empirical FAR curve -->
			<path
				d={lgbmPath}
				fill="none"
				stroke="#1C2721"
				stroke-width="2.75"
				stroke-linejoin="round"
				stroke-linecap="round"
			/>

			<!-- Active alpha cutoff line -->
			<line
				x1={xPx(activeAlpha)}
				y1={pad.top}
				x2={xPx(activeAlpha)}
				y2={H - pad.bottom}
				stroke="#B84A39"
				stroke-width="1.75"
			/>

			<!-- Interactive sweep dots -->
			{#each sortedSweep as pt}
				{@const cx = xPx(pt.alpha)}
				{@const cy = yPx(pt.false_accept_rate)}
				{@const isActive = activePoint?.alpha === pt.alpha}
				{#if !compact || isActive}
					{#if onSelectAlpha}
						<circle
							cx={cx}
							cy={cy}
							r={isActive ? 5.5 : 3}
							fill={isActive ? '#B84A39' : '#FCFBF8'}
							stroke={isActive ? '#FCFBF8' : '#1C2721'}
							stroke-width={isActive ? 2 : 1.25}
							class="cursor-pointer"
							role="button"
							tabindex={0}
							aria-label="Alpha {pt.alpha.toFixed(2)}: false-accept {(pt.false_accept_rate * 100).toFixed(2)}%, abstention {(pt.abstention_rate * 100).toFixed(1)}%"
							onmouseenter={() => (hoveredPoint = pt)}
							onmouseleave={() => (hoveredPoint = null)}
							onfocus={() => (hoveredPoint = pt)}
							onblur={() => (hoveredPoint = null)}
							onclick={() => onSelectAlpha(pt.alpha)}
							onkeydown={(e) => {
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									onSelectAlpha(pt.alpha);
								}
							}}
						/>
					{:else}
						<circle
							cx={cx}
							cy={cy}
							r={isActive ? 5.5 : 3}
							fill={isActive ? '#B84A39' : '#FCFBF8'}
							stroke={isActive ? '#FCFBF8' : '#1C2721'}
							stroke-width={isActive ? 2 : 1.25}
						/>
					{/if}
				{/if}
			{/each}

			<!-- Active readout label -->
			{#if displayPoint}
				{@const ax = xPx(displayPoint.alpha)}
				{@const alignRight = displayPoint.alpha > 0.17}
				<text
					x={alignRight ? ax - 8 : ax + 8}
					y={pad.top + 12}
					text-anchor={alignRight ? 'end' : 'start'}
					font-size="11"
					font-weight="700"
					fill="#B84A39"
					class="font-mono-data"
				>
					α={displayPoint.alpha.toFixed(2)} · FAR {(displayPoint.false_accept_rate * 100).toFixed(2)}% · τ={displayPoint.threshold.toFixed(3)}
				</text>
			{/if}
		</svg>
	</div>
{/if}
