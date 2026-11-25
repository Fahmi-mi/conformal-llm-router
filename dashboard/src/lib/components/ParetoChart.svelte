<script lang="ts">
	import { findNearestParetoPoint } from '#lib/analytics';
	import type { LiveParetoPoint, ParetoBaseline, ParetoCurvePoint } from '#lib/types';

	let {
		curve = [],
		baselines = {},
		livePoint = null,
		activeLambda = 0.45,
		compact = false,
		datasetLabel = 'RouterBench In-Domain',
		onSelectLambda
	}: {
		curve?: ParetoCurvePoint[];
		baselines?: Record<string, ParetoBaseline>;
		livePoint?: LiveParetoPoint | null;
		activeLambda?: number;
		compact?: boolean;
		datasetLabel?: string;
		onSelectLambda?: (lambdaCost: number) => void;
	} = $props();

	let hoveredPoint = $state<ParetoCurvePoint | null>(null);

	const W = $derived(compact ? 540 : 720);
	const H = $derived(compact ? 225 : 340);
	const pad = $derived(
		compact
			? { top: 20, right: 20, bottom: 38, left: 48 }
			: { top: 26, right: 28, bottom: 48, left: 58 }
	);

	const plotW = $derived(W - pad.left - pad.right);
	const plotH = $derived(H - pad.top - pad.bottom);

	const sortedCurve = $derived(
		[...curve].sort((a, b) => a.normalized_cost - b.normalized_cost)
	);

	const yBounds = $derived.by(() => {
		const qualities = sortedCurve.map((p) => p.avg_quality);
		for (const b of Object.values(baselines)) {
			if (Number.isFinite(b.avg_quality)) qualities.push(b.avg_quality);
		}
		if (qualities.length === 0) return { min: 0.25, max: 0.85 };
		const rawMin = Math.min(...qualities);
		const rawMax = Math.max(...qualities);
		return {
			min: Math.max(0, Math.floor((rawMin - 0.05) * 20) / 20),
			max: Math.min(1, Math.ceil((rawMax + 0.05) * 20) / 20)
		};
	});

	function xPx(normCost: number): number {
		const clamped = Math.min(1, Math.max(0, normCost));
		return pad.left + clamped * plotW;
	}

	function yPx(quality: number): number {
		const span = Math.max(0.05, yBounds.max - yBounds.min);
		const norm = (quality - yBounds.min) / span;
		return pad.top + (1 - Math.min(1, Math.max(0, norm))) * plotH;
	}

	const pathD = $derived.by(() => {
		if (sortedCurve.length === 0) return '';
		return sortedCurve
			.map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${xPx(pt.normalized_cost).toFixed(1)} ${yPx(pt.avg_quality).toFixed(1)}`)
			.join(' ');
	});

	const activePoint = $derived(findNearestParetoPoint(curve, activeLambda));
	const displayPoint = $derived(hoveredPoint ?? activePoint);

	const yTicks = $derived.by(() => {
		const count = compact ? 3 : 5;
		const span = yBounds.max - yBounds.min;
		return Array.from({ length: count + 1 }, (_, i) => yBounds.min + (span * i) / count);
	});

	const xTicks = [0, 0.25, 0.5, 0.75, 1.0];
</script>

{#if sortedCurve.length === 0}
	<div class="border-t border-[#ECE7DC] bg-[#F6F3EC] p-6 text-center text-xs text-[#5C6860]">
		Pareto sweep curve unavailable (TBD). Run Phase 3 router evaluation or enable Sample Preview.
	</div>
{:else}
	<div class="w-full">
		<svg
			viewBox="0 0 {W} {H}"
			class="w-full h-auto block select-none"
			role="img"
			aria-label="Pareto Cost-Quality Frontier ({datasetLabel})"
		>
			<!-- Horizontal gridlines -->
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

			<!-- Vertical ticks -->
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
					{(tick * 100).toFixed(0)}%
				</text>
			{/each}

			<!-- Axes -->
			<line x1={pad.left} y1={H - pad.bottom} x2={W - pad.right} y2={H - pad.bottom} stroke="#CFC7B6" stroke-width="1.25" />
			<line x1={pad.left} y1={pad.top} x2={pad.left} y2={H - pad.bottom} stroke="#CFC7B6" stroke-width="1.25" />

			<!-- Axis labels -->
			<text
				x={pad.left + plotW / 2}
				y={H - 6}
				text-anchor="middle"
				font-size="11"
				fill="#5C6860"
			>
				Normalized Cost vs. Always-Frontier (%)
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
					Avg Quality Score
				</text>
			{/if}

			<!-- Baseline reference markers -->
			{#if baselines.random_router}
				{@const bx = xPx(baselines.random_router.normalized_cost)}
				{@const by = yPx(baselines.random_router.avg_quality)}
				<rect x={bx - 4} y={by - 4} width="8" height="8" fill="#FCFBF8" stroke="#5C6860" stroke-width="1.75" />
				{#if !compact}
					<text x={bx + 8} y={by + 4} font-size="10.5" fill="#5C6860">Random</text>
				{/if}
			{/if}

			{#if baselines.always_frontier}
				{@const fx = xPx(baselines.always_frontier.normalized_cost)}
				{@const fy = yPx(baselines.always_frontier.avg_quality)}
				<line x1={pad.left} y1={fy} x2={W - pad.right} y2={fy} stroke="#9EA79F" stroke-dasharray="4 4" stroke-width="1" />
				<circle cx={fx} cy={fy} r="4" fill="#1C2721" />
				{#if !compact}
					<text x={fx - 8} y={fy - 8} text-anchor="end" font-size="10.5" fill="#1C2721">Frontier 100%</text>
				{/if}
			{/if}

			<!-- Pareto Frontier Curve -->
			<path d={pathD} fill="none" stroke="#236B52" stroke-width="2.75" stroke-linejoin="round" stroke-linecap="round" />

			<!-- Sweep points -->
			{#each sortedCurve as pt}
				{@const cx = xPx(pt.normalized_cost)}
				{@const cy = yPx(pt.avg_quality)}
				{@const isActive = activePoint?.lambda_cost === pt.lambda_cost}
				{#if !compact || isActive}
					{#if onSelectLambda}
						<circle
							cx={cx}
							cy={cy}
							r={isActive ? 6 : 3.5}
							fill={isActive ? '#B84A39' : '#FCFBF8'}
							stroke={isActive ? '#FCFBF8' : '#236B52'}
							stroke-width={isActive ? 2 : 1.5}
							class="cursor-pointer"
							role="button"
							tabindex={0}
							aria-label="Lambda {pt.lambda_cost.toFixed(2)}: quality {(pt.avg_quality * 100).toFixed(1)}%, cost {(pt.normalized_cost * 100).toFixed(1)}%"
							onmouseenter={() => (hoveredPoint = pt)}
							onmouseleave={() => (hoveredPoint = null)}
							onclick={() => onSelectLambda(pt.lambda_cost)}
							onkeydown={(e) => {
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									onSelectLambda(pt.lambda_cost);
								}
							}}
						/>
					{:else}
						<circle
							cx={cx}
							cy={cy}
							r={isActive ? 6 : 3.5}
							fill={isActive ? '#B84A39' : '#FCFBF8'}
							stroke={isActive ? '#FCFBF8' : '#236B52'}
							stroke-width={isActive ? 2 : 1.5}
						/>
					{/if}
				{/if}
			{/each}

			<!-- Active point callout -->
			{#if displayPoint}
				{@const ax = xPx(displayPoint.normalized_cost)}
				{@const ay = yPx(displayPoint.avg_quality)}
				{@const alignRight = ax > pad.left + plotW * 0.58}
				<line x1={ax} y1={pad.top} x2={ax} y2={H - pad.bottom} stroke="#B84A39" stroke-dasharray="3 3" stroke-width="1" opacity="0.6" />
				<g transform="translate({alignRight ? ax - 10 : ax + 10}, {Math.max(pad.top + 14, ay - 6)})">
					<text
						text-anchor={alignRight ? 'end' : 'start'}
						font-size="11"
						font-weight="700"
						fill="#1C2721"
						class="font-mono-data"
					>
						λ={displayPoint.lambda_cost.toFixed(2)} · Q {(displayPoint.avg_quality * 100).toFixed(1)}% · Cost {(displayPoint.normalized_cost * 100).toFixed(1)}%
					</text>
				</g>
			{/if}

			<!-- Live gateway point indicator (when frontier baseline cost available) -->
			{#if livePoint && livePoint.estimated_frontier_cost_usd > 0}
				{@const liveNormCost = Math.min(1, Math.max(0, livePoint.avg_cost_usd / livePoint.estimated_frontier_cost_usd))}
				<line
					x1={xPx(liveNormCost)}
					y1={pad.top}
					x2={xPx(liveNormCost)}
					y2={H - pad.bottom}
					stroke="#B46914"
					stroke-width="1.5"
					stroke-dasharray="2 2"
				/>
				<text
					x={xPx(liveNormCost)}
					y={pad.top + 11}
					text-anchor={liveNormCost > 0.7 ? 'end' : 'start'}
					dx={liveNormCost > 0.7 ? -4 : 4}
					font-size="10"
					font-weight="600"
					fill="#8F520E"
				>
					Live Cost {(liveNormCost * 100).toFixed(1)}%
				</text>
			{/if}
		</svg>
	</div>
{/if}
