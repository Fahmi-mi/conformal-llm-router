import type {
	AuditRecord,
	ConformalSweepPoint,
	ParetoCurvePoint
} from './types';

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
