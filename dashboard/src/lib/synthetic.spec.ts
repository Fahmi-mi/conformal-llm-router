import { describe, expect, it } from 'vitest';
import {
	buildSyntheticParetoTelemetry,
	simulateSyntheticEvaluateRisk,
	simulateSyntheticQuery,
	SYNTHETIC_AUDIT_RECORDS,
	SYNTHETIC_BENCHMARKS_SUMMARY,
	SYNTHETIC_HEALTH
} from './synthetic';

describe('synthetic fixtures and simulator', () => {
	it('provides valid health and benchmark summary structures', () => {
		expect(SYNTHETIC_HEALTH.status).toBe('ok');
		expect(SYNTHETIC_BENCHMARKS_SUMMARY.judge_audit.status).toBe('tbd');
		expect(SYNTHETIC_BENCHMARKS_SUMMARY.conformal.status).toBe('ready');
		expect(SYNTHETIC_BENCHMARKS_SUMMARY.pareto.status).toBe('ready');
		expect(SYNTHETIC_BENCHMARKS_SUMMARY.pareto.ood_available).toBe(true);
		expect(SYNTHETIC_AUDIT_RECORDS.some((r) => r.conformal_decision === 'ABSTAIN' && r.draft_answer)).toBe(true);
		expect(SYNTHETIC_AUDIT_RECORDS.some((r) => r.conformal_decision === 'UNSCORED' && r.evidence === null)).toBe(true);
	});

	it('updates synthetic pareto telemetry dynamically with alpha and lambda', () => {
		const lowLam = buildSyntheticParetoTelemetry(0.05, 0.2);
		const highLam = buildSyntheticParetoTelemetry(0.2, 0.8);
		expect(highLam.tier1_share).toBeGreaterThan(lowLam.tier1_share);
		expect(highLam.abstention_rate).toBeLessThan(lowLam.abstention_rate);
		expect(highLam.alpha_filter).toBe(0.2);
		expect(highLam.lambda_cost_filter).toBe(0.8);
	});

	it('simulates query and evaluate-risk decisions consistently', () => {
		const unscored = simulateSyntheticQuery({
			query: 'Explain conformal guarantees',
			context: null,
			alpha: 0.1,
			lambdaCost: 0.45
		});
		expect(unscored.conformal_decision).toBe('UNSCORED');
		expect(unscored.evidence).toBeNull();

		const abstain = simulateSyntheticQuery({
			query: 'What penalty applies for SLA failure?',
			context: 'Only billing credits apply.',
			alpha: 0.1,
			lambdaCost: 0.45
		});
		expect(abstain.conformal_decision).toBe('ABSTAIN');
		expect(abstain.draft_answer).toBeTruthy();

		const riskEval = simulateSyntheticEvaluateRisk({
			context: 'Lease liability was $42.8M at 5.4% incremental borrowing rate.',
			candidate_answer: 'Lease liability was $42.8M at 5.4% borrowing rate.',
			alpha: 0.1
		});
		expect(riskEval.threshold_used).toBeGreaterThan(0);
		expect(['PASS', 'ABSTAIN']).toContain(riskEval.conformal_decision);
	});
});
