<script lang="ts">
	type BadgeVariant =
		| 'PASS'
		| 'ABSTAIN'
		| 'UNSCORED'
		| 'SYNTHETIC'
		| 'FALLBACK'
		| 'ESCALATED'
		| 'CACHE'
		| 'READY'
		| 'TBD'
		| 'TIER1'
		| 'TIER2'
		| 'NEUTRAL';

	let {
		variant,
		label = ''
	}: {
		variant: BadgeVariant;
		label?: string;
	} = $props();

	const styleMap: Record<BadgeVariant, string> = {
		PASS: 'bg-[#E4F2EC] text-[#1B5642] border-[#B6D9C9]',
		ABSTAIN: 'bg-[#FBE9E7] text-[#9E3829] border-[#E8BEB8]',
		UNSCORED: 'bg-[#EFECE3] text-[#4A554E] border-[#DCD6C8]',
		SYNTHETIC: 'bg-[#FDF6E9] text-[#8F520E] border-[#DFB97D]',
		FALLBACK: 'bg-[#FDF6E9] text-[#8F520E] border-[#DFB97D]',
		ESCALATED: 'bg-[#FBE9E7] text-[#9E3829] border-[#E8BEB8]',
		CACHE: 'bg-[#E4F2EC] text-[#1B5642] border-[#B6D9C9]',
		READY: 'bg-[#E4F2EC] text-[#1B5642] border-[#B6D9C9]',
		TBD: 'bg-[#FDF6E9] text-[#8F520E] border-[#DFB97D]',
		TIER1: 'bg-[#EFECE3] text-[#1C2721] border-[#DCD6C8]',
		TIER2: 'bg-[#1C2721] text-[#EDE8DC] border-[#1C2721]',
		NEUTRAL: 'bg-[#EFECE3] text-[#4A554E] border-[#DCD6C8]'
	};

	const text = $derived(label || variant);
</script>

<span
	class="inline-flex items-center px-2 py-0.5 text-[11px] font-semibold tracking-wide uppercase border rounded-[2px] whitespace-nowrap {styleMap[variant]}"
>
	{text}
</span>
