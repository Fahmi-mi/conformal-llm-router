import { describe, expect, it } from 'vitest';
import {
	clampAlpha,
	clampLambda,
	classifyGatewayStatus,
	computeNextAuditOffset,
	filterSyntheticAuditRecords,
	findNearestConformalPoint,
	findNearestParetoPoint,
	formatLatencyMs,
	formatPct,
	formatQualityRetention,
	formatSpecimenId,
	formatUsd,
	isArtifactReady,
	resolveSelectedAuditRecord
} from './analytics';
import {
	SYNTHETIC_AUDIT_RECORDS,
	SYNTHETIC_CONFORMAL_SWEEP,
	SYNTHETIC_HEALTH,
	SYNTHETIC_PARETO_CURVE
} from './synthetic';

describe('analytics helpers', () => {
	it('clamps alpha and lambda within backend schema bounds', () => {
		expect(clampAlpha(0.001)).toBe(0.01);
		expect(clampAlpha(0.45)).toBe(0.3);
		expect(clampAlpha(0.123)).toBe(0.12);
		expect(clampLambda(-0.2)).toBe(0);
		expect(clampLambda(1.4)).toBe(1);
		expect(clampLambda(0.454)).toBe(0.45);
	});

	it('finds nearest conformal and pareto sweep points', () => {
		const cPt = findNearestConformalPoint(SYNTHETIC_CONFORMAL_SWEEP, 0.1);
		expect(cPt).not.toBeNull();
		expect(cPt?.alpha).toBe(0.1);

		const pPt = findNearestParetoPoint(SYNTHETIC_PARETO_CURVE, 0.44);
		expect(pPt).not.toBeNull();
		expect(pPt?.lambda_cost).toBe(0.45);
		expect(findNearestParetoPoint([], 0.5)).toBeNull();
	});

	it('formats percentages, usd, latency, specimen ids, and TBD values', () => {
		expect(formatPct(0.0415, 2)).toBe('4.15%');
		expect(formatPct(null)).toBe('TBD');
		expect(formatUsd(0)).toBe('$0.0000');
		expect(formatUsd(0.00014)).toBe('$0.00014');
		expect(formatLatencyMs(410.4)).toBe('410ms');
		expect(formatLatencyMs(1250)).toBe('1.25s');
		expect(formatSpecimenId('spec-1042-9a18')).toBe('#SPEC-1042');
		expect(formatSpecimenId('a1b2c3d4-9999')).toBe('#A1B2C3D4');
		expect(formatSpecimenId('#SPEC-1042')).toBe('#SPEC-1042');
		expect(isArtifactReady('ready')).toBe(true);
		expect(isArtifactReady('tbd')).toBe(false);
		expect(formatQualityRetention(0.748, 0.778)).toBe('96.1%');
	});

	it('filters synthetic audit records by decision, tier, cache_hit, and query', () => {
		const abstains = filterSyntheticAuditRecords(SYNTHETIC_AUDIT_RECORDS, { decision: 'ABSTAIN' });
		expect(abstains).toHaveLength(1);
		expect(abstains[0].draft_answer).toBeTruthy();

		const cached = filterSyntheticAuditRecords(SYNTHETIC_AUDIT_RECORDS, { cache_hit: true });
		expect(cached).toHaveLength(1);
		expect(cached[0].estimated_cost_usd).toBe(0);

		const leaseMatches = filterSyntheticAuditRecords(SYNTHETIC_AUDIT_RECORDS, {
			tier: 'tier1',
			q: 'lease'
		});
		expect(leaseMatches).toHaveLength(2);
	});

	it('classifies HTTP 503 vs unreachable vs degraded vs synthetic gateway states', () => {
		const err503 = classifyGatewayStatus({
			syntheticPreview: false,
			loading: false,
			health: SYNTHETIC_HEALTH,
			gatewayError: { status: 503, detail: 'Model artifacts missing', unreachable: false }
		});
		expect(err503.kind).toBe('http-error');
		expect(err503.railLabel).toBe('HTTP 503');
		expect(err503.pillLabel).toBe('GATEWAY HTTP 503');

		const unreachable = classifyGatewayStatus({
			syntheticPreview: false,
			loading: false,
			health: null,
			gatewayError: { status: 0, detail: 'fetch failed', unreachable: true }
		});
		expect(unreachable.kind).toBe('unreachable');
		expect(unreachable.railLabel).toBe('Unreachable');

		const degraded = classifyGatewayStatus({
			syntheticPreview: false,
			loading: false,
			health: { ...SYNTHETIC_HEALTH, status: 'degraded' },
			gatewayError: null
		});
		expect(degraded.kind).toBe('degraded');
		expect(degraded.railLabel).toBe('Degraded');

		const preview = classifyGatewayStatus({
			syntheticPreview: true,
			loading: false,
			health: null,
			gatewayError: { status: 503, detail: 'ignored in preview', unreachable: false }
		});
		expect(preview.kind).toBe('synthetic');
		expect(preview.railLabel).toBe('Nominal (Preview)');
	});

	it('computes audit pagination offsets and preserves selected record', () => {
		expect(computeNextAuditOffset(0, 25, 60, 'prev')).toBe(0);
		expect(computeNextAuditOffset(0, 25, 60, 'next')).toBe(25);
		expect(computeNextAuditOffset(25, 25, 60, 'next')).toBe(50);
		expect(computeNextAuditOffset(50, 25, 60, 'next')).toBe(50);

		const second = SYNTHETIC_AUDIT_RECORDS[1];
		expect(resolveSelectedAuditRecord(SYNTHETIC_AUDIT_RECORDS, second)?.request_id).toBe(second.request_id);
		expect(resolveSelectedAuditRecord(SYNTHETIC_AUDIT_RECORDS.slice(2), second)?.request_id).toBe(
			SYNTHETIC_AUDIT_RECORDS[2].request_id
		);
		expect(resolveSelectedAuditRecord([], second)).toBeNull();
	});
});
