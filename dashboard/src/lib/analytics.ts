import type {
	AuditRecord,
	ConformalSweepPoint,
	HealthResponse,
	ParetoCurvePoint
} from './types';

export interface GatewayErrorState {
	status: number;
	detail: string;
	unreachable: boolean;
}

export interface GatewayStatusSummary {
	kind: 'synthetic' | 'checking' | 'unreachable' | 'http-error' | 'degraded' | 'nominal' | 'offline';
	railLabel: string;
	railClass: string;
	pillLabel: string;
	pillClass: string;
}

export function classifyGatewayStatus(opts: {
	syntheticPreview: boolean;
	loading: boolean;
	health: HealthResponse | null;
	gatewayError: GatewayErrorState | null;
}): GatewayStatusSummary {
	if (opts.syntheticPreview) {
		return {
			kind: 'synthetic',
			railLabel: 'Nominal (Preview)',
			railClass: 'bg-[#B46914] text-[#FAF8F3]',
			pillLabel: 'PREVIEW · [SYNTHETIC]',
			pillClass: 'bg-[#B46914] text-[#FAF8F3]'
		};
	}
	if (opts.loading && !opts.health && !opts.gatewayError) {
		return {
			kind: 'checking',
			railLabel: 'Checking…',
			railClass: 'bg-[#2A3A31] text-[#EDE8DC]',
			pillLabel: 'CHECKING',
			pillClass: 'bg-[#2A3A31] text-[#EDE8DC]'
		};
	}
	if (opts.gatewayError) {
		if (opts.gatewayError.unreachable) {
			return {
				kind: 'unreachable',
				railLabel: 'Unreachable',
				railClass: 'bg-[#B84A39] text-[#FAF8F3]',
				pillLabel: 'OFFLINE / UNREACHABLE',
				pillClass: 'bg-[#B84A39] text-[#FAF8F3]'
			};
		}
		const statusLabel = opts.gatewayError.status > 0 ? `HTTP ${opts.gatewayError.status}` : 'Error';
		return {
			kind: 'http-error',
			railLabel: statusLabel,
			railClass: 'bg-[#B84A39] text-[#FAF8F3]',
			pillLabel: `GATEWAY ${statusLabel}`,
			pillClass: 'bg-[#B84A39] text-[#FAF8F3]'
		};
	}
	if (!opts.health) {
		return {
			kind: 'offline',
			railLabel: 'Unreachable',
			railClass: 'bg-[#B84A39] text-[#FAF8F3]',
			pillLabel: 'OFFLINE / STANDBY',
			pillClass: 'bg-[#2A3A31] text-[#EDE8DC]'
		};
	}
	if (opts.health.status === 'ok') {
		return {
			kind: 'nominal',
			railLabel: 'Nominal',
			railClass: 'bg-[#236B52] text-[#FAF8F3]',
			pillLabel: 'LIVE GATEWAY',
			pillClass: 'bg-[#236B52] text-[#FAF8F3]'
		};
	}
	return {
		kind: 'degraded',
		railLabel: 'Degraded',
		railClass: 'bg-[#B46914] text-[#FAF8F3]',
		pillLabel: 'GATEWAY DEGRADED',
		pillClass: 'bg-[#B46914] text-[#FAF8F3]'
	};
}

export function computeNextAuditOffset(
	currentOffset: number,
	pageSize: number,
	total: number,
	direction: 'prev' | 'next'
): number {
	if (direction === 'prev') {
		return Math.max(0, currentOffset - pageSize);
	}
	const next = currentOffset + pageSize;
	return next < total ? next : currentOffset;
}

export function resolveSelectedAuditRecord(
	items: AuditRecord[],
	current: AuditRecord | null
): AuditRecord | null {
	if (items.length === 0) return null;
	if (current) {
		const updated = items.find((r) => r.request_id === current.request_id);
		if (updated) return updated;
	}
	return items[0];
}

export function clampAlpha(value: number): number {
	if (!Number.isFinite(value)) return 0.1;
	return Math.min(0.3, Math.max(0.01, Math.round(value * 100) / 100));
}

export function clampLambda(value: number): number {
	if (!Number.isFinite(value)) return 0.45;
	return Math.min(1.0, Math.max(0.0, Math.round(value * 100) / 100));
}

export function findNearestConformalPoint(
	sweep: ConformalSweepPoint[] | undefined,
	targetAlpha: number
): ConformalSweepPoint | null {
	if (!sweep || sweep.length === 0) return null;
	let best = sweep[0];
	let bestDiff = Math.abs(best.alpha - targetAlpha);
	for (let i = 1; i < sweep.length; i++) {
		const diff = Math.abs(sweep[i].alpha - targetAlpha);
		if (diff < bestDiff) {
			best = sweep[i];
			bestDiff = diff;
		}
	}
	return best;
}

export function findNearestParetoPoint(
	curve: ParetoCurvePoint[] | undefined,
	targetLambda: number
): ParetoCurvePoint | null {
	if (!curve || curve.length === 0) return null;
	let best = curve[0];
	let bestDiff = Math.abs(best.lambda_cost - targetLambda);
	for (let i = 1; i < curve.length; i++) {
		const diff = Math.abs(curve[i].lambda_cost - targetLambda);
		if (diff < bestDiff) {
			best = curve[i];
			bestDiff = diff;
		}
	}
	return best;
}

export function formatPct(ratio: number | null | undefined, digits = 1): string {
	if (ratio === null || ratio === undefined || !Number.isFinite(ratio)) return 'TBD';
	return `${(ratio * 100).toFixed(digits)}%`;
}

export function formatPercentValue(pct: number | null | undefined, digits = 1): string {
	if (pct === null || pct === undefined || !Number.isFinite(pct)) return 'TBD';
	return `${pct.toFixed(digits)}%`;
}

export function formatFixed(value: number | null | undefined, digits = 3): string {
	if (value === null || value === undefined || !Number.isFinite(value)) return 'TBD';
	return value.toFixed(digits);
}

export function formatUsd(amount: number | null | undefined): string {
	if (amount === null || amount === undefined || !Number.isFinite(amount)) return 'TBD';
	if (amount === 0) return '$0.0000';
	if (amount < 0.001) return `$${amount.toFixed(5)}`;
	return `$${amount.toFixed(4)}`;
}

export function formatLatencyMs(ms: number | null | undefined): string {
	if (ms === null || ms === undefined || !Number.isFinite(ms)) return 'TBD';
	if (ms < 1000) return `${Math.round(ms)}ms`;
	return `${(ms / 1000).toFixed(2)}s`;
}

export function formatSpecimenId(requestId: string): string {
	if (!requestId) return '#SPEC-0000';
	if (requestId.startsWith('#')) return requestId;
	if (requestId.startsWith('spec-')) {
		return `#SPEC-${requestId.slice(5, 9).toUpperCase()}`;
	}
	return `#${requestId.slice(0, 8).toUpperCase()}`;
}

export function formatTimestamp(iso: string): string {
	const d = new Date(iso);
	if (Number.isNaN(d.getTime())) return iso;
	return d.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
}

export function filterSyntheticAuditRecords(
	items: AuditRecord[],
	filters: {
		decision?: string | null;
		tier?: string | null;
		cache_hit?: boolean | null;
		q?: string | null;
	}
): AuditRecord[] {
	const queryLower = filters.q?.trim().toLowerCase() ?? '';
	return items.filter((item) => {
		if (filters.decision && item.conformal_decision !== filters.decision) return false;
		if (filters.tier && item.tier !== filters.tier) return false;
		if (filters.cache_hit !== null && filters.cache_hit !== undefined && item.cache_hit !== filters.cache_hit) {
			return false;
		}
		if (queryLower && !item.query_text.toLowerCase().includes(queryLower)) return false;
		return true;
	});
}

export function isArtifactReady(status: string | undefined | null): boolean {
	return status?.toLowerCase() === 'ready';
}

export function formatQualityRetention(
	pointQuality: number | undefined,
	frontierQuality: number | undefined
): string {
	if (
		pointQuality === undefined ||
		frontierQuality === undefined ||
		frontierQuality <= 0 ||
		!Number.isFinite(pointQuality) ||
		!Number.isFinite(frontierQuality)
	) {
		return 'TBD';
	}
	return `${((pointQuality / frontierQuality) * 100).toFixed(1)}%`;
}
