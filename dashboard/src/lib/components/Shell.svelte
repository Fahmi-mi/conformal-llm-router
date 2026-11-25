<script lang="ts">
	import type { Snippet } from 'svelte';
	import { page } from '$app/state';
	import { classifyGatewayStatus } from '#lib/analytics';
	import { consoleState } from '#lib/state.svelte';
	import ControlBar from './ControlBar.svelte';
	import DecisionBadge from './DecisionBadge.svelte';
	import Icon from './Icon.svelte';

	let { children }: { children: Snippet } = $props();

	const navItems = [
		{ href: '/', label: 'Executive Overview', index: '01', icon: 'overview' as const },
		{ href: '/pareto', label: 'Pareto Frontier', index: '02', icon: 'pareto' as const },
		{ href: '/conformal', label: 'Conformal Risk', index: '03', icon: 'conformal' as const },
		{ href: '/audit', label: 'Audit Log Inspector', index: '04', icon: 'audit' as const },
		{ href: '/playground', label: 'Live Playground', index: '05', icon: 'playground' as const },
		{ href: '/benchmarks', label: 'Offline Benchmarks', index: '06', icon: 'benchmarks' as const }
	];

	function isCurrentRoute(href: string, pathname: string): boolean {
		if (href === '/') return pathname === '/';
		return pathname.startsWith(href);
	}

	const gatewayStatus = $derived(
		classifyGatewayStatus({
			syntheticPreview: consoleState.syntheticPreview,
			loading: consoleState.loading,
			health: consoleState.health,
			gatewayError: consoleState.gatewayError
		})
	);

	const providerSummary = $derived.by(() => {
		const providers = consoleState.health?.providers;
		if (!providers) return 'Tier 1: — · Tier 2: —';
		const tier1 = ['groq', 'gemini', 'ollama'].filter((k) => providers[k]).map((k) => k[0].toUpperCase() + k.slice(1));
		const tier2 = providers.deepseek ? 'DeepSeek' : 'Unconfigured';
		return `T1: ${tier1.length > 0 ? tier1.join(' / ') : 'None'} · T2: ${tier2}`;
	});
</script>

<div class="min-h-screen flex flex-col md:grid md:grid-cols-[248px_1fr] bg-[#F6F3EC] text-[#1C2721]">
	<!-- Mobile top header -->
	<header class="md:hidden sticky top-0 z-30 bg-[#1C2721] text-[#EDE8DC] border-b border-[#131B17] px-4 py-3 flex items-center justify-between">
		<div>
			<div class="text-sm font-bold text-[#FAF8F3] leading-tight">Conformal LLM Router</div>
			<div class="text-[11px] uppercase tracking-[0.12em] text-[#9FB3A6]">Statistical AI Control</div>
		</div>
		<div class="flex items-center gap-2.5">
			<span class="px-2 py-0.5 rounded-[2px] text-[11px] font-semibold uppercase {gatewayStatus.railClass}">
				{gatewayStatus.railLabel}
			</span>
			<button
				type="button"
				onclick={() => (consoleState.mobileNavOpen = !consoleState.mobileNavOpen)}
				class="p-1.5 rounded-[2px] border border-[#2F4036] text-[#EDE8DC] hover:bg-[#2A3A31]"
				aria-label={consoleState.mobileNavOpen ? 'Close navigation menu' : 'Open navigation menu'}
				aria-expanded={consoleState.mobileNavOpen}
			>
				<Icon name={consoleState.mobileNavOpen ? 'close' : 'menu'} size={18} />
			</button>
		</div>
	</header>

	<!-- Mobile drawer navigation -->
	{#if consoleState.mobileNavOpen}
		<nav
			aria-label="Mobile navigation"
			class="md:hidden bg-[#1C2721] text-[#EDE8DC] border-b border-[#2F4036] px-4 py-3 space-y-1"
		>
			{#each navItems as item}
				{@const active = isCurrentRoute(item.href, page.url.pathname)}
				<a
					href={item.href}
					onclick={() => (consoleState.mobileNavOpen = false)}
					class="flex items-center justify-between px-3 py-2 rounded-[3px] text-sm transition-colors {active
						? 'bg-[#2A3A31] text-[#FAF8F3] font-semibold'
						: 'text-[#C9D2CB] hover:bg-[#24322B] hover:text-[#FAF8F3]'}"
				>
					<span class="flex items-center gap-2.5">
						<Icon name={item.icon} size={15} />
						<span>{item.label}</span>
					</span>
					<span class="font-mono-data text-xs flex items-center gap-2">
						{#if active}
							<span class="w-1.5 h-1.5 rounded-full bg-[#D97757]"></span>
						{/if}
						{item.index}
					</span>
				</a>
			{/each}
		</nav>
	{/if}

	<!-- Desktop persistent 248px left rail -->
	<aside class="hidden md:flex md:flex-col md:justify-between md:sticky md:top-0 md:h-screen bg-[#1C2721] text-[#EDE8DC] px-4 py-6 border-r border-[#131B17]">
		<div>
			<div class="px-2 mb-6">
				<div class="text-[18px] font-bold leading-snug text-[#FAF8F3]">Conformal LLM Router</div>
				<div class="text-[11px] uppercase tracking-[0.14em] text-[#9FB3A6] mt-1">
					Statistical AI Control
				</div>
			</div>

			<nav aria-label="Primary console workspaces" class="flex flex-col gap-1">
				{#each navItems as item}
					{@const active = isCurrentRoute(item.href, page.url.pathname)}
					<a
						href={item.href}
						aria-current={active ? 'page' : undefined}
						class="flex items-center justify-between px-3 py-2.5 rounded-[3px] text-[13.5px] transition-colors {active
							? 'bg-[#2A3A31] text-[#FAF8F3] font-semibold'
							: 'text-[#C9D2CB] hover:bg-[#24322B] hover:text-[#FAF8F3]'}"
					>
						<span class="flex items-center gap-2.5 min-w-0">
							<Icon name={item.icon} size={15} class="shrink-0" />
							<span class="truncate">{item.label}</span>
						</span>
						<span class="font-mono-data text-xs flex items-center gap-1.5 shrink-0 {active ? 'text-[#FAF8F3]' : 'text-[#9FB3A6]'}">
							{#if active}
								<span class="w-1.5 h-1.5 rounded-full bg-[#D97757]" aria-hidden="true"></span>
							{/if}
							{item.index}
						</span>
					</a>
				{/each}
			</nav>
		</div>

		<div class="border-t border-[#2F4036] pt-4 px-2 space-y-2 text-xs text-[#C9D2CB]">
			<div class="flex items-center justify-between gap-2">
				<span>Gateway Health</span>
				<strong class="text-[#FAF8F3]">{gatewayStatus.railLabel}</strong>
			</div>
			<div class="text-xs leading-relaxed text-[#A9B8AE]">
				{providerSummary}
			</div>
			<div class="font-mono-data text-xs text-[#9FB3A6] truncate" title={consoleState.gatewayBaseUrl}>
				{consoleState.gatewayBaseUrl}
			</div>
			<div class="pt-1 flex flex-wrap gap-1.5">
				<span class="inline-block px-2 py-0.5 rounded-[2px] text-[11px] font-semibold {gatewayStatus.pillClass}">
					{gatewayStatus.pillLabel}
				</span>
			</div>
		</div>
	</aside>

	<!-- Right analytical workbench -->
	<main class="min-w-0 px-4 py-5 sm:px-6 lg:px-8 flex flex-col gap-4">
		{#if consoleState.syntheticPreview}
			<div
				role="status"
				class="border border-[#B46914] bg-[#FDF6E9] text-[#8F520E] px-4 py-2.5 rounded-[3px] flex flex-wrap items-center justify-between gap-3 text-xs"
			>
				<div class="flex items-center gap-2">
					<Icon name="spark" size={15} class="shrink-0" />
					<span>
						<strong>Sample Preview Mode [Synthetic]</strong> — Displaying fixture telemetry for exploration. Live FastAPI telemetry is paused.
					</span>
				</div>
				<button
					type="button"
					onclick={() => consoleState.setSyntheticPreview(false)}
					class="px-2.5 py-1 border border-[#B46914] bg-[#FCFBF8] text-[#8F520E] font-semibold rounded-[2px] hover:bg-[#FAECCB] cursor-pointer"
				>
					Exit Preview
				</button>
			</div>
		{:else if consoleState.gatewayError}
			<div
				role="alert"
				class="border border-[#B84A39] bg-[#FBE9E7] text-[#8E3224] px-4 py-3 rounded-[3px] flex flex-wrap items-center justify-between gap-3 text-xs"
			>
				<div class="flex items-start sm:items-center gap-2.5 min-w-0">
					<Icon name="warning" size={16} class="shrink-0 mt-0.5 sm:mt-0" />
					<div>
						{#if consoleState.gatewayError.unreachable}
							<strong>Gateway Unreachable</strong> at <code class="font-mono-data">{consoleState.gatewayBaseUrl}</code>
							<span class="block sm:inline sm:ml-1 text-[#7B2D21]">
								({consoleState.gatewayError.detail}). Start FastAPI or enable Sample Preview to inspect the interface.
							</span>
						{:else}
							<strong>
								Gateway Error{consoleState.gatewayError.status > 0 ? ` (HTTP ${consoleState.gatewayError.status})` : ''}
							</strong>
							at <code class="font-mono-data">{consoleState.gatewayBaseUrl}</code>
							<span class="block sm:inline sm:ml-1 text-[#7B2D21]">
								— {consoleState.gatewayError.detail}
							</span>
						{/if}
					</div>
				</div>
				<div class="flex items-center gap-2 shrink-0">
					<button
						type="button"
						onclick={() => consoleState.refreshAll()}
						class="px-2.5 py-1 border border-[#B84A39] bg-[#FCFBF8] text-[#8E3224] font-semibold rounded-[2px] hover:bg-[#F5D6D2] cursor-pointer"
					>
						Retry
					</button>
					<button
						type="button"
						onclick={() => consoleState.setSyntheticPreview(true)}
						class="px-2.5 py-1 border border-[#B46914] bg-[#FDF6E9] text-[#8F520E] font-semibold rounded-[2px] hover:bg-[#FAECCB] cursor-pointer"
					>
						Preview Sample Data [Synthetic]
					</button>
				</div>
			</div>
		{:else if consoleState.health?.status === 'degraded'}
			<div
				role="status"
				class="border border-[#DFB97D] bg-[#FDF6E9] text-[#8F520E] px-4 py-2.5 rounded-[3px] flex flex-wrap items-center justify-between gap-2 text-xs"
			>
				<div class="flex items-center gap-2">
					<Icon name="warning" size={15} class="shrink-0" />
					<span>
						<strong>Gateway Degraded:</strong> Missing required artifacts or storage dependencies ({Object.entries(
							{ ...consoleState.health.artifacts, ...consoleState.health.storage }
						)
							.filter(([, ok]) => !ok)
							.map(([name]) => name)
							.join(', ') || 'check /health'}).
					</span>
				</div>
				<DecisionBadge variant="FALLBACK" label="DEGRADED" />
			</div>
		{/if}

		<ControlBar />

		{@render children()}
	</main>
</div>
