<script lang="ts">
	import { onMount } from 'svelte';
	import './layout.css';
	import favicon from '#lib/assets/favicon.svg';
	import Shell from '#lib/components/Shell.svelte';
	import { consoleState } from '#lib/state.svelte';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();

	onMount(() => {
		void consoleState.ensureInitialized();
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Conformal LLM Router — Statistical AI Control</title>
</svelte:head>

<Shell>
	{@render children()}
</Shell>
