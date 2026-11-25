import type {
	AuditRecord,
	BenchmarksSummaryResponse,
	ConformalSweepPoint,
	EvaluateRiskResponse,
	HealthResponse,
	ParetoCurvePoint,
	ParetoTelemetryResponse,
	QueryResponse
} from './types';

export const SYNTHETIC_HEALTH: HealthResponse = {
	status: 'ok',
	artifacts: {
		calibrator: true,
		router: true
	},
	models: {
		embedder_loaded: true,
		nli_loaded: true
	},
	storage: {
		sqlite: true,
		faiss: true
	},
	providers: {
		groq: true,
		gemini: true,
		ollama: true,
		deepseek: true
	}
};

export const SYNTHETIC_PARETO_CURVE: ParetoCurvePoint[] = [
	{ lambda_cost: 0.0, avg_quality: 0.778, avg_cost: 0.0032, normalized_cost: 1.0, cost_savings: 0.0, tier1_rate: 0.0 },
	{ lambda_cost: 0.1, avg_quality: 0.778, avg_cost: 0.00317, normalized_cost: 0.992, cost_savings: 0.008, tier1_rate: 0.007 },
	{ lambda_cost: 0.2, avg_quality: 0.7777, avg_cost: 0.00314, normalized_cost: 0.981, cost_savings: 0.019, tier1_rate: 0.015 },
	{ lambda_cost: 0.3, avg_quality: 0.7749, avg_cost: 0.00308, normalized_cost: 0.961, cost_savings: 0.039, tier1_rate: 0.029 },
	{ lambda_cost: 0.4, avg_quality: 0.7632, avg_cost: 0.003, normalized_cost: 0.936, cost_savings: 0.064, tier1_rate: 0.066 },
	{ lambda_cost: 0.45, avg_quality: 0.748, avg_cost: 0.00293, normalized_cost: 0.915, cost_savings: 0.085, tier1_rate: 0.1 },
	{ lambda_cost: 0.5, avg_quality: 0.7244, avg_cost: 0.00283, normalized_cost: 0.884, cost_savings: 0.116, tier1_rate: 0.151 },
	{ lambda_cost: 0.6, avg_quality: 0.5772, avg_cost: 0.00218, normalized_cost: 0.68, cost_savings: 0.32, tier1_rate: 0.448 },
	{ lambda_cost: 0.65, avg_quality: 0.4417, avg_cost: 0.00097, normalized_cost: 0.304, cost_savings: 0.696, tier1_rate: 0.748 },
	{ lambda_cost: 0.7, avg_quality: 0.3627, avg_cost: 0.00039, normalized_cost: 0.123, cost_savings: 0.877, tier1_rate: 0.902 },
	{ lambda_cost: 0.8, avg_quality: 0.3119, avg_cost: 0.00006, normalized_cost: 0.018, cost_savings: 0.982, tier1_rate: 0.992 },
	{ lambda_cost: 0.9, avg_quality: 0.3071, avg_cost: 0.00005, normalized_cost: 0.014, cost_savings: 0.986, tier1_rate: 1.0 },
	{ lambda_cost: 1.0, avg_quality: 0.3071, avg_cost: 0.00005, normalized_cost: 0.014, cost_savings: 0.986, tier1_rate: 1.0 }
];

export const SYNTHETIC_OOD_CURVE: ParetoCurvePoint[] = [
	{ lambda_cost: 0.0, avg_quality: 0.6862, avg_cost: 0.00938, normalized_cost: 1.0, cost_savings: 0.0, tier1_rate: 0.0 },
	{ lambda_cost: 0.25, avg_quality: 0.6862, avg_cost: 0.00936, normalized_cost: 0.998, cost_savings: 0.002, tier1_rate: 0.002 },
	{ lambda_cost: 0.3, avg_quality: 0.6815, avg_cost: 0.00909, normalized_cost: 0.969, cost_savings: 0.031, tier1_rate: 0.026 },
	{ lambda_cost: 0.35, avg_quality: 0.644, avg_cost: 0.00829, normalized_cost: 0.884, cost_savings: 0.116, tier1_rate: 0.11 },
	{ lambda_cost: 0.4, avg_quality: 0.5855, avg_cost: 0.0066, normalized_cost: 0.704, cost_savings: 0.296, tier1_rate: 0.295 },
	{ lambda_cost: 0.45, avg_quality: 0.4965, avg_cost: 0.00386, normalized_cost: 0.412, cost_savings: 0.588, tier1_rate: 0.588 },
	{ lambda_cost: 0.5, avg_quality: 0.3864, avg_cost: 0.00134, normalized_cost: 0.143, cost_savings: 0.857, tier1_rate: 0.859 },
	{ lambda_cost: 0.6, avg_quality: 0.3489, avg_cost: 0.0003, normalized_cost: 0.032, cost_savings: 0.968, tier1_rate: 0.977 },
	{ lambda_cost: 0.8, avg_quality: 0.3443, avg_cost: 0.00005, normalized_cost: 0.005, cost_savings: 0.995, tier1_rate: 1.0 },
	{ lambda_cost: 1.0, avg_quality: 0.3443, avg_cost: 0.00005, normalized_cost: 0.005, cost_savings: 0.995, tier1_rate: 1.0 }
];

export const SYNTHETIC_CONFORMAL_SWEEP: ConformalSweepPoint[] = Array.from({ length: 30 }, (_, idx) => {
	const alpha = Number(((idx + 1) * 0.01).toFixed(2));
	const far = Number((alpha * 0.43 + 0.002).toFixed(4));
	const abstention = Number(Math.max(0.28, 0.91 - alpha * 1.95).toFixed(4));
	const threshold = Number((0.09 + alpha * 1.42).toFixed(4));
	return {
		alpha,
		threshold,
		false_accept_rate: far,
		abstention_rate: abstention,
		ece: 0.0761,
		brier_score: 0.1734
	};
});

export const SYNTHETIC_LOGISTIC_SWEEP: ConformalSweepPoint[] = Array.from({ length: 30 }, (_, idx) => {
	const alpha = Number(((idx + 1) * 0.01).toFixed(2));
	const far = Number((alpha * 0.46 + 0.003).toFixed(4));
	const abstention = Number(Math.max(0.35, 0.93 - alpha * 1.82).toFixed(4));
	const threshold = Number((0.16 + alpha * 1.15).toFixed(4));
	return {
		alpha,
		threshold,
		false_accept_rate: far,
		abstention_rate: abstention,
		ece: 0.0775,
		brier_score: 0.1894
	};
});

export const SYNTHETIC_AUDIT_RECORDS: AuditRecord[] = [
	{
		request_id: 'spec-1042-9a18',
		created_at: '2026-11-25T09:24:12Z',
		query_text: 'Summarize the quarterly lease liability note and discount rate disclosed in Section 7.',
		context_text:
			'Note 7 (Leases): Total undiscounted operating lease liabilities as of Q3 stood at $42.8M, discounted at a weighted-average incremental borrowing rate of 5.4% over a remaining term of 6.2 years.',
		answer_text:
			'Section 7 reports $42.8M in undiscounted operating lease liabilities in Q3, using a 5.4% weighted-average incremental borrowing rate across a 6.2-year remaining term.',
		draft_answer: null,
		model_used: 'llama-3.1-8b-instant',
		tier: 'tier1',
		provider: 'groq',
		escalated_from_tier1: false,
		cache_hit: false,
		fallback_used: false,
		estimated_cost_usd: 0.00008,
		frontier_cost_usd: 0.00072,
		alpha_used: 0.1,
		lambda_cost_used: 0.45,
		calibrated_risk_score: 0.064,
		conformal_decision: 'PASS',
		evidence: {
			nli_entailment: 0.91,
			nli_contradiction: 0.02,
			nli_neutral: 0.07,
			cosine_sim: 0.86,
			entity_overlap: 0.89,
			length_ratio: 0.88,
			threshold_used: 0.2956
		},
		latency_ms: 214
	},
	{
		request_id: 'spec-1041-4c02',
		created_at: '2026-11-25T09:22:45Z',
		query_text: 'What penalty clause applies if supplier shipments miss the 48-hour SLA window?',
		context_text:
			'Section 11.3 (Service Credits): Late deliveries beyond 48 hours accrue a 1.5% invoice credit capped at 10% per calendar quarter. No monetary termination penalty is stipulated.',
		answer_text:
			'Abstained: generated response exceeded the calibrated hallucination risk bound.',
		draft_answer:
			'Missed SLA shipments trigger a $25,000 fixed monetary penalty per incident plus immediate contract termination rights.',
		model_used: 'deepseek-chat',
		tier: 'tier2',
		provider: 'deepseek',
		escalated_from_tier1: true,
		cache_hit: false,
		fallback_used: false,
		estimated_cost_usd: 0.00094,
		frontier_cost_usd: 0.00086,
		alpha_used: 0.1,
		lambda_cost_used: 0.45,
		calibrated_risk_score: 0.438,
		conformal_decision: 'ABSTAIN',
		evidence: {
			nli_entailment: 0.08,
			nli_contradiction: 0.84,
			nli_neutral: 0.08,
			cosine_sim: 0.42,
			entity_overlap: 0.31,
			length_ratio: 0.74,
			threshold_used: 0.2956
		},
		latency_ms: 682
	},
	{
		request_id: 'spec-1040-7f31',
		created_at: '2026-11-25T09:19:03Z',
		query_text: 'Extract the conformal calibration split count and default alpha from the RAGTruth spec.',
		context_text:
			'The RAGTruth pipeline reserves 12,072 training samples, 3,018 held-out calibration samples for MAPIE LAC quantiles, and 2,700 test samples with default target risk alpha = 0.10.',
		answer_text:
			'The RAGTruth evaluation uses 3,018 calibration samples (with 12,072 training and 2,700 test rows) and a default target risk level of alpha = 0.10.',
		draft_answer: null,
		model_used: 'gemini-2.0-flash-lite',
		tier: 'tier1',
		provider: 'gemini',
		escalated_from_tier1: false,
		cache_hit: false,
		fallback_used: true,
		estimated_cost_usd: 0.00014,
		frontier_cost_usd: 0.00069,
		alpha_used: 0.1,
		lambda_cost_used: 0.45,
		calibrated_risk_score: 0.088,
		conformal_decision: 'PASS',
		evidence: {
			nli_entailment: 0.88,
			nli_contradiction: 0.03,
			nli_neutral: 0.09,
			cosine_sim: 0.83,
			entity_overlap: 0.81,
			length_ratio: 0.85,
			threshold_used: 0.2956
		},
		latency_ms: 345
	},
	{
		request_id: 'spec-1039-2b90',
		created_at: '2026-11-25T09:15:18Z',
		query_text: 'Explain why split conformal prediction yields finite-sample marginal coverage guarantees.',
		context_text: null,
		answer_text:
			'Under exchangeability between calibration and test points, the (1 - alpha)(1 + 1/n) empirical quantile of nonconformity scores guarantees P(Y in C(X)) >= 1 - alpha in finite samples without distributional assumptions.',
		draft_answer: null,
		model_used: 'deepseek-chat',
		tier: 'tier2',
		provider: 'deepseek',
		escalated_from_tier1: false,
		cache_hit: false,
		fallback_used: false,
		estimated_cost_usd: 0.00079,
		frontier_cost_usd: 0.00079,
		alpha_used: 0.1,
		lambda_cost_used: 0.45,
		calibrated_risk_score: null,
		conformal_decision: 'UNSCORED',
		evidence: null,
		latency_ms: 510
	},
	{
		request_id: 'spec-1038-1d44',
		created_at: '2026-11-25T09:11:09Z',
		query_text: 'Summarize the quarterly lease liability note and discount rate disclosed in Section 7.',
		context_text:
			'Note 7 (Leases): Total undiscounted operating lease liabilities as of Q3 stood at $42.8M, discounted at a weighted-average incremental borrowing rate of 5.4% over a remaining term of 6.2 years.',
		answer_text:
			'Section 7 reports $42.8M in undiscounted operating lease liabilities in Q3, using a 5.4% weighted-average incremental borrowing rate across a 6.2-year remaining term.',
		draft_answer: null,
		model_used: 'llama-3.1-8b-instant',
		tier: 'tier1',
		provider: 'groq',
		escalated_from_tier1: false,
		cache_hit: true,
		fallback_used: false,
		estimated_cost_usd: 0.0,
		frontier_cost_usd: 0.00072,
		alpha_used: 0.1,
		lambda_cost_used: 0.45,
		calibrated_risk_score: 0.064,
		conformal_decision: 'PASS',
		evidence: {
			nli_entailment: 0.91,
			nli_contradiction: 0.02,
			nli_neutral: 0.07,
			cosine_sim: 0.86,
			entity_overlap: 0.89,
			length_ratio: 0.88,
			threshold_used: 0.2956
		},
		latency_ms: 19
	},
	{
		request_id: 'spec-1037-8e52',
		created_at: '2026-11-25T09:08:50Z',
		query_text: 'Confirm whether ISO-27001 certification covers the APAC telemetry region in 2026.',
		context_text:
			'Compliance Annex A: SOC 2 Type II covers US-East and EU-Central clusters. ISO-27001 audit scoping for APAC is scheduled for Q4 and has not yet completed certification.',
		answer_text:
			'Based on Compliance Annex A, APAC ISO-27001 certification is not yet complete; audit scoping is scheduled for Q4, while SOC 2 Type II currently covers US-East and EU-Central.',
		draft_answer: null,
		model_used: 'deepseek-chat',
		tier: 'tier2',
		provider: 'deepseek',
		escalated_from_tier1: true,
		cache_hit: false,
		fallback_used: false,
		estimated_cost_usd: 0.00088,
		frontier_cost_usd: 0.00082,
		alpha_used: 0.1,
		lambda_cost_used: 0.45,
		calibrated_risk_score: 0.142,
		conformal_decision: 'PASS',
		evidence: {
			nli_entailment: 0.79,
			nli_contradiction: 0.06,
			nli_neutral: 0.15,
			cosine_sim: 0.78,
			entity_overlap: 0.75,
			length_ratio: 0.92,
			threshold_used: 0.2956
		},
		latency_ms: 486
	}
];

export function buildSyntheticParetoTelemetry(
	alpha: number,
	lambdaCost: number
): ParetoTelemetryResponse {
	const tier1Share = Math.min(0.95, Math.max(0.1, 0.2 + lambdaCost * 0.65));
	const abstentionRate = Math.min(0.42, Math.max(0.06, 0.26 - alpha * 0.85));
	const avgCost = Number((0.00082 * (1 - tier1Share * 0.72)).toFixed(5));
	const frontierCost = 0.00082;
	const savingsPct = ((frontierCost - avgCost) / frontierCost) * 100;

	return {
		total_requests: 128,
		cache_hit_rate: 0.281,
		abstention_rate: abstentionRate,
		avg_cost_usd: avgCost,
		estimated_frontier_cost_usd: frontierCost,
		realized_cost_savings_pct: savingsPct,
		p50_latency_ms: 192,
		p95_latency_ms: 410,
		tier1_share: tier1Share,
		alpha_filter: alpha,
		lambda_cost_filter: lambdaCost,
		live_points: [
			{
				alpha_used: 0.05,
				lambda_cost_used: 0.35,
				total_requests: 34,
				cache_hit_rate: 0.235,
				abstention_rate: 0.235,
				avg_cost_usd: 0.00062,
				estimated_frontier_cost_usd: 0.00082,
				realized_cost_savings_pct: 24.4,
				p50_latency_ms: 230,
				p95_latency_ms: 495,
				tier1_share: 0.353
			},
			{
				alpha_used: alpha,
				lambda_cost_used: lambdaCost,
				total_requests: 64,
				cache_hit_rate: 0.281,
				abstention_rate: abstentionRate,
				avg_cost_usd: avgCost,
				estimated_frontier_cost_usd: frontierCost,
				realized_cost_savings_pct: savingsPct,
				p50_latency_ms: 192,
				p95_latency_ms: 410,
				tier1_share: tier1Share
			},
			{
				alpha_used: 0.15,
				lambda_cost_used: 0.65,
				total_requests: 30,
				cache_hit_rate: 0.333,
				abstention_rate: 0.133,
				avg_cost_usd: 0.00028,
				estimated_frontier_cost_usd: 0.00082,
				realized_cost_savings_pct: 65.9,
				p50_latency_ms: 145,
				p95_latency_ms: 340,
				tier1_share: 0.767
			}
		],
		offline_curve: SYNTHETIC_PARETO_CURVE
	};
}

export const SYNTHETIC_BENCHMARKS_SUMMARY: BenchmarksSummaryResponse = {
	judge_audit: {
		status: 'tbd',
		reason: 'Synthetic preview preserves unrun Judge Audit provenance until an API judge run is executed.',
		sample_size: 0,
		requested_sample_size: 300,
		features_parquet: 'data/processed/ragtruth_features.parquet',
		metrics: null,
		corrected_metrics: null,
		bias_correction: null
	},
	conformal: {
		status: 'ready',
		model: 'lightgbm',
		default_alpha: 0.1,
		n_train: 12072,
		n_cal: 3018,
		n_test: 2700,
		sweep: SYNTHETIC_CONFORMAL_SWEEP,
		logistic_sweep: SYNTHETIC_LOGISTIC_SWEEP
	},
	risk_metrics: {
		status: 'ready',
		dataset: 'data/processed/ragtruth_features.parquet',
		split_sizes: { train: 12072, cal: 3018, test: 2700 },
		uncalibrated_heuristic: {
			guardrail_method: 'Uncalibrated Heuristic Threshold',
			target_alpha: null,
			threshold: 0.5,
			false_accept_rate: 0.1352,
			abstention_rate: 0.3452,
			ece: 0.065,
			brier_score: 0.1963
		},
		logistic_mapie: {
			'0.05': {
				guardrail_method: 'Logistic + Split Conformal (MAPIE)',
				target_alpha: 0.05,
				threshold: 0.2561,
				false_accept_rate: 0.0233,
				abstention_rate: 0.7659,
				ece: 0.0775,
				brier_score: 0.1894,
				alpha: 0.05
			},
			'0.10': {
				guardrail_method: 'Logistic + Split Conformal (MAPIE)',
				target_alpha: 0.1,
				threshold: 0.3017,
				false_accept_rate: 0.0467,
				abstention_rate: 0.6537,
				ece: 0.0775,
				brier_score: 0.1894,
				alpha: 0.1
			}
		},
		lightgbm_mapie: {
			'0.05': {
				guardrail_method: 'LightGBM + Split Conformal (MAPIE)',
				target_alpha: 0.05,
				threshold: 0.2112,
				false_accept_rate: 0.0196,
				abstention_rate: 0.7493,
				ece: 0.0761,
				brier_score: 0.1734,
				alpha: 0.05
			},
			'0.10': {
				guardrail_method: 'LightGBM + Split Conformal (MAPIE)',
				target_alpha: 0.1,
				threshold: 0.2956,
				false_accept_rate: 0.0415,
				abstention_rate: 0.6315,
				ece: 0.0761,
				brier_score: 0.1734,
				alpha: 0.1
			}
		}
	},
	pareto: {
		status: 'ready',
		n_train: 21642,
		n_test: 7214,
		n_ood: 427,
		default_lambda: 0.45,
		curve: SYNTHETIC_PARETO_CURVE,
		baselines: {
			always_frontier: { avg_quality: 0.778, avg_cost: 0.0032, normalized_cost: 1.0, cost_savings: 0.0, tier1_rate: 0.0 },
			always_small: { avg_quality: 0.3071, avg_cost: 0.000046, normalized_cost: 0.0143, cost_savings: 0.9857, tier1_rate: 1.0 },
			random_router: { avg_quality: 0.5373, avg_cost: 0.001598, normalized_cost: 0.4993, cost_savings: 0.5007, tier1_rate: 0.5068 },
			learned_pareto_router: {
				lambda_cost: 0.45,
				avg_quality: 0.748,
				avg_cost: 0.002928,
				normalized_cost: 0.915,
				cost_savings: 0.085,
				tier1_rate: 0.1002
			}
		},
		auc_qc: 0.5206,
		cost_savings_at_95pct_quality: 0.085,
		ood_available: true,
		ood_tasks: ['mbpp'],
		ood_curve: SYNTHETIC_OOD_CURVE,
		ood_baselines: {
			always_frontier: { avg_quality: 0.6862, avg_cost: 0.009377, normalized_cost: 1.0, cost_savings: 0.0, tier1_rate: 0.0 },
			always_small: { avg_quality: 0.3443, avg_cost: 0.000048, normalized_cost: 0.0052, cost_savings: 0.9948, tier1_rate: 1.0 },
			random_router: { avg_quality: 0.5059, avg_cost: 0.004752, normalized_cost: 0.5068, cost_savings: 0.4932, tier1_rate: 0.5035 },
			learned_pareto_router: {
				lambda_cost: 0.45,
				avg_quality: 0.4965,
				avg_cost: 0.003862,
				normalized_cost: 0.4118,
				cost_savings: 0.5882,
				tier1_rate: 0.5878
			}
		},
		ood_auc_qc: 0.5148,
		ood_cost_savings_at_95pct_quality: 0.0307
	}
};

export function simulateSyntheticQuery(payload: {
	query: string;
	context?: string | null;
	alpha: number;
	lambdaCost: number;
}): QueryResponse {
	const hasContext = Boolean(payload.context && payload.context.trim().length > 0);
	const lower = payload.query.toLowerCase() + ' ' + (payload.context ?? '').toLowerCase();
	const risky = lower.includes('penalty') || lower.includes('guarantee') || lower.includes('contradict');
	const threshold = Number((0.09 + payload.alpha * 1.42).toFixed(4));
	const risk = risky ? Number((threshold + 0.12).toFixed(4)) : Number((threshold * 0.34).toFixed(4));
	const decision = !hasContext ? 'UNSCORED' : risk < threshold ? 'PASS' : 'ABSTAIN';
	const tier = payload.lambdaCost >= 0.4 && !risky ? 'tier1' : 'tier2';

	return {
		request_id: `syn-${Date.now().toString(36)}`,
		answer:
			decision === 'ABSTAIN'
				? 'Abstained: generated response exceeded the calibrated hallucination risk bound.'
				: `[Synthetic preview] Evaluated query "${payload.query.slice(0, 72)}" under ${tier.toUpperCase()} routing.`,
		draft_answer:
			decision === 'ABSTAIN'
				? '[Synthetic draft] Unverified claim blocked by MAPIE conformal threshold check.'
				: null,
		model_used: tier === 'tier1' ? 'llama-3.1-8b-instant' : 'deepseek-chat',
		tier,
		provider: tier === 'tier1' ? 'groq' : 'deepseek',
		escalated_from_tier1: decision === 'ABSTAIN' || (tier === 'tier2' && payload.lambdaCost >= 0.4),
		cache_hit: false,
		fallback_used: false,
		estimated_cost_usd: tier === 'tier1' ? 0.00009 : 0.00089,
		alpha_used: payload.alpha,
		lambda_cost_used: payload.lambdaCost,
		calibrated_risk_score: hasContext ? risk : null,
		conformal_decision: decision,
		evidence: hasContext
			? {
					nli_entailment: decision === 'ABSTAIN' ? 0.11 : 0.89,
					nli_contradiction: decision === 'ABSTAIN' ? 0.81 : 0.03,
					nli_neutral: 0.08,
					cosine_sim: decision === 'ABSTAIN' ? 0.44 : 0.84,
					entity_overlap: decision === 'ABSTAIN' ? 0.32 : 0.86,
					length_ratio: 0.82,
					threshold_used: threshold
				}
			: null,
		latency_ms: tier === 'tier1' ? 185 : 540
	};
}

export function simulateSyntheticEvaluateRisk(payload: {
	context: string;
	candidate_answer: string;
	alpha: number;
}): EvaluateRiskResponse {
	const threshold = Number((0.09 + payload.alpha * 1.42).toFixed(4));
	const overlap =
		payload.context
			.toLowerCase()
			.split(/\W+/)
			.filter((w) => w.length > 3 && payload.candidate_answer.toLowerCase().includes(w)).length / 8;
	const grounded = Math.min(1, Math.max(0.15, overlap));
	const risk = Number((1 - grounded * 0.85).toFixed(4));
	const decision = risk < threshold ? 'PASS' : 'ABSTAIN';

	return {
		calibrated_risk_score: risk,
		threshold_used: threshold,
		alpha_used: payload.alpha,
		conformal_decision: decision,
		evidence: {
			nli_entailment: Number(grounded.toFixed(3)),
			nli_contradiction: Number((1 - grounded).toFixed(3)),
			nli_neutral: 0.08,
			cosine_sim: Number((0.4 + grounded * 0.5).toFixed(3)),
			entity_overlap: Number(grounded.toFixed(3)),
			length_ratio: Number(
				Math.min(2, payload.candidate_answer.length / Math.max(1, payload.context.length)).toFixed(3)
			),
			threshold_used: threshold
		}
	};
}
