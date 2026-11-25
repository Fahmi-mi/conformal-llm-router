export type ConformalDecision = 'PASS' | 'ABSTAIN' | 'UNSCORED';
export type RouterTier = 'tier1' | 'tier2';
export type ProvenanceMode = 'live' | 'offline' | 'synthetic' | 'tbd';

export interface EvidenceBreakdown {
	nli_entailment: number;
	nli_contradiction: number;
	nli_neutral: number;
	cosine_sim: number;
	entity_overlap: number;
	length_ratio: number;
	threshold_used?: number | null;
}

export interface QueryRequest {
	query: string;
	context?: string | null;
	alpha_override?: number | null;
	lambda_cost_override?: number | null;
	bypass_cache?: boolean;
}

export interface QueryResponse {
	request_id: string;
	answer: string;
	draft_answer: string | null;
	model_used: string;
	tier: RouterTier;
	provider: string;
	escalated_from_tier1: boolean;
	cache_hit: boolean;
	fallback_used: boolean;
	estimated_cost_usd: number;
	alpha_used: number;
	lambda_cost_used: number;
	calibrated_risk_score: number | null;
	conformal_decision: ConformalDecision;
	evidence: EvidenceBreakdown | null;
	latency_ms: number;
}

export interface EvaluateRiskRequest {
	query: string;
	context: string;
	candidate_answer: string;
	alpha_override?: number | null;
}

export interface EvaluateRiskResponse {
	calibrated_risk_score: number;
	threshold_used: number;
	alpha_used: number;
	conformal_decision: 'PASS' | 'ABSTAIN';
	evidence: EvidenceBreakdown;
}

export interface AuditRecord {
	request_id: string;
	created_at: string;
	query_text: string;
	context_text: string | null;
	answer_text: string;
	draft_answer: string | null;
	model_used: string;
	tier: RouterTier;
	provider: string;
	escalated_from_tier1: boolean;
	cache_hit: boolean;
	fallback_used: boolean;
	estimated_cost_usd: number;
	frontier_cost_usd: number;
	alpha_used: number;
	lambda_cost_used: number;
	calibrated_risk_score: number | null;
	conformal_decision: ConformalDecision;
	evidence: EvidenceBreakdown | null;
	latency_ms: number;
}

export interface AuditListResponse {
	items: AuditRecord[];
	total: number;
	limit: number;
	offset: number;
}

export interface AuditQueryFilters {
	limit?: number;
	offset?: number;
	decision?: ConformalDecision | null;
	tier?: RouterTier | null;
	cache_hit?: boolean | null;
	q?: string | null;
}

export interface ParetoCurvePoint {
	lambda_cost: number;
	avg_quality: number;
	avg_cost: number;
	normalized_cost: number;
	cost_savings: number;
	tier1_rate: number;
}

export interface LiveParetoPoint {
	alpha_used: number;
	lambda_cost_used: number;
	total_requests: number;
	cache_hit_rate: number;
	abstention_rate: number;
	avg_cost_usd: number;
	estimated_frontier_cost_usd: number;
	realized_cost_savings_pct: number;
	p50_latency_ms: number;
	p95_latency_ms: number;
	tier1_share: number;
}

export interface ParetoTelemetryResponse {
	total_requests: number;
	cache_hit_rate: number;
	abstention_rate: number;
	avg_cost_usd: number;
	estimated_frontier_cost_usd: number;
	realized_cost_savings_pct: number;
	p50_latency_ms: number;
	p95_latency_ms: number;
	tier1_share: number;
	alpha_filter: number;
	lambda_cost_filter: number;
	live_points: LiveParetoPoint[];
	offline_curve: ParetoCurvePoint[];
}

export interface ConformalSweepPoint {
	threshold: number;
	false_accept_rate: number;
	abstention_rate: number;
	ece: number;
	brier_score: number;
	alpha: number;
}

export interface ConformalSweepArtifact {
	status: string;
	model?: string;
	default_alpha?: number;
	n_train?: number;
	n_cal?: number;
	n_test?: number;
	sweep?: ConformalSweepPoint[];
	logistic_sweep?: ConformalSweepPoint[];
}

export interface RiskMetricEntry {
	guardrail_method: string;
	target_alpha: number | null;
	threshold: number;
	false_accept_rate: number;
	abstention_rate: number;
	ece: number;
	brier_score: number;
	alpha?: number;
}

export interface RiskMetricsArtifact {
	status: string;
	dataset?: string;
	split_sizes?: {
		train: number;
		cal: number;
		test: number;
	};
	uncalibrated_heuristic?: RiskMetricEntry;
	logistic_mapie?: Record<string, RiskMetricEntry>;
	lightgbm_mapie?: Record<string, RiskMetricEntry>;
}

export interface ParetoBaseline {
	lambda_cost?: number;
	avg_quality: number;
	avg_cost: number;
	normalized_cost: number;
	cost_savings: number;
	tier1_rate: number;
}

export interface ParetoArtifact {
	status: string;
	n_train?: number;
	n_test?: number;
	n_ood?: number;
	default_lambda?: number;
	lambdas?: number[];
	curve?: ParetoCurvePoint[];
	baselines?: Record<string, ParetoBaseline>;
	auc_qc?: number;
	cost_savings_at_95pct_quality?: number | null;
	ood_available?: boolean;
	ood_tasks?: string[];
	ood_curve?: ParetoCurvePoint[];
	ood_baselines?: Record<string, ParetoBaseline>;
	ood_auc_qc?: number;
	ood_cost_savings_at_95pct_quality?: number | null;
}

export interface JudgeAgreementMetrics {
	n_samples: number;
	cohens_kappa: number;
	spearman_r: number;
	spearman_p: number;
	accuracy: number;
	confusion_matrix: number[][];
	threshold?: number;
	by_task?: Record<string, JudgeAgreementMetrics>;
}

export interface JudgeBiasCorrection {
	n_samples: number;
	verbosity_coef: number;
	position_coef: number;
	human_label_coef: number;
	intercept: number;
	length_mean: number;
	position_mean: number;
	raw_spearman_r: number;
	corrected_spearman_r: number;
	raw_cohens_kappa: number;
	corrected_cohens_kappa: number;
}

export interface JudgeAuditArtifact {
	status: string;
	reason?: string;
	provider?: string;
	sample_size?: number;
	requested_sample_size?: number;
	paired_evaluations?: number;
	features_parquet?: string;
	metrics: JudgeAgreementMetrics | null;
	corrected_metrics: JudgeAgreementMetrics | null;
	bias_correction: JudgeBiasCorrection | null;
}

export interface BenchmarksSummaryResponse {
	judge_audit: JudgeAuditArtifact;
	conformal: ConformalSweepArtifact;
	risk_metrics: RiskMetricsArtifact;
	pareto: ParetoArtifact;
}

export interface HealthResponse {
	status: 'ok' | 'degraded';
	artifacts: Record<string, boolean>;
	models: Record<string, boolean>;
	storage: Record<string, boolean>;
	providers: Record<string, boolean>;
}
