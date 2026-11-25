<script lang="ts">
	type IconName =
		| 'overview'
		| 'pareto'
		| 'conformal'
		| 'audit'
		| 'playground'
		| 'benchmarks'
		| 'refresh'
		| 'menu'
		| 'close'
		| 'warning'
		| 'check'
		| 'shield'
		| 'spark'
		| 'chevron-right'
		| 'copy';

	let {
		name,
		size = 16,
		class: className = ''
	}: {
		name: IconName;
		size?: number;
		class?: string;
	} = $props();
</script>

<svg
	width={size}
	height={size}
	viewBox="0 0 24 24"
	fill="none"
	stroke="currentColor"
	stroke-width="1.75"
	stroke-linecap="round"
	stroke-linejoin="round"
	class={className}
	aria-hidden="true"
>
	{#if name === 'overview'}
		<rect x="3" y="3" width="8" height="8" rx="1" />
		<rect x="13" y="3" width="8" height="5" rx="1" />
		<rect x="13" y="10" width="8" height="11" rx="1" />
		<rect x="3" y="13" width="8" height="8" rx="1" />
	{:else if name === 'pareto'}
		<path d="M3 3v18h18" />
		<path d="M6 16c2.5-6 6.5-9 13-10" />
		<circle cx="11" cy="9.5" r="1.75" />
	{:else if name === 'conformal'}
		<path d="M3 3v18h18" />
		<path d="M4 20L20 4" stroke-dasharray="3 3" />
		<path d="M4 20c5-1.5 10-4.5 16-9" />
		<line x1="11" y1="4" x2="11" y2="20" />
	{:else if name === 'audit'}
		<path d="M4 6h16" />
		<path d="M4 12h16" />
		<path d="M4 18h10" />
		<circle cx="18" cy="18" r="2.5" />
	{:else if name === 'playground'}
		<polygon points="6 4 20 12 6 20 6 4" />
	{:else if name === 'benchmarks'}
		<path d="M9 3h6" />
		<path d="M12 3v18" />
		<path d="M5 8h14" />
		<path d="M5 14h14" />
		<rect x="4" y="5" width="16" height="16" rx="1.5" />
	{:else if name === 'refresh'}
		<path d="M21 12a9 9 0 1 1-2.64-6.36" />
		<polyline points="21 3 21 9 15 9" />
	{:else if name === 'menu'}
		<line x1="4" y1="6" x2="20" y2="6" />
		<line x1="4" y1="12" x2="20" y2="12" />
		<line x1="4" y1="18" x2="20" y2="18" />
	{:else if name === 'close'}
		<line x1="18" y1="6" x2="6" y2="18" />
		<line x1="6" y1="6" x2="18" y2="18" />
	{:else if name === 'warning'}
		<path d="M12 3L2.5 19.5h19L12 3z" />
		<line x1="12" y1="9.5" x2="12" y2="14" />
		<line x1="12" y1="17" x2="12.01" y2="17" />
	{:else if name === 'check'}
		<polyline points="20 6 9 17 4 12" />
	{:else if name === 'shield'}
		<path d="M12 3l7 3v6c0 4.5-3 7.8-7 9-4-1.2-7-4.5-7-9V6l7-3z" />
	{:else if name === 'spark'}
		<path d="M12 3l2.2 6.2L20.5 11l-6.3 1.8L12 19l-2.2-6.2L3.5 11l6.3-1.8L12 3z" />
	{:else if name === 'chevron-right'}
		<polyline points="9 6 15 12 9 18" />
	{:else if name === 'copy'}
		<rect x="9" y="9" width="11" height="11" rx="1.5" />
		<path d="M5 15H4a1.5 1.5 0 0 1-1.5-1.5V4A1.5 1.5 0 0 1 4 2.5h9.5A1.5 1.5 0 0 1 15 4v1" />
	{/if}
</svg>
