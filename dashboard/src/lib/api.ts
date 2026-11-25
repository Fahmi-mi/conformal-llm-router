import type {
	AuditListResponse,
	AuditQueryFilters,
	BenchmarksSummaryResponse,
	EvaluateRiskRequest,
	EvaluateRiskResponse,
	HealthResponse,
	ParetoTelemetryResponse,
	QueryRequest,
	QueryResponse
} from './types';

export const DEFAULT_GATEWAY_URL = 'http://localhost:8000';

export class GatewayApiError extends Error {
	readonly status: number;
	readonly detail: string;
	readonly unreachable: boolean;

	constructor(message: string, opts: { status: number; detail: string; unreachable: boolean }) {
		super(message);
		this.name = 'GatewayApiError';
		this.status = opts.status;
		this.detail = opts.detail;
		this.unreachable = opts.unreachable;
	}
}

export function normalizeBaseUrl(rawUrl?: string | null): string {
	const trimmed = rawUrl?.trim();
	if (!trimmed) return DEFAULT_GATEWAY_URL;
	return trimmed.replace(/\/+$/, '');
}

async function parseErrorDetail(resp: Response): Promise<string> {
	try {
		const data = (await resp.json()) as { detail?: unknown };
		if (typeof data?.detail === 'string' && data.detail.trim()) {
			return data.detail;
		}
		if (Array.isArray(data?.detail)) {
			return data.detail
				.map((item) => (typeof item?.msg === 'string' ? item.msg : JSON.stringify(item)))
				.join('; ');
		}
	} catch {
		// Fall through to status text when response body is non-JSON.
	}
	return resp.statusText || `HTTP ${resp.status}`;
}

export interface GatewayClient {
	readonly baseUrl: string;
	getHealth(): Promise<HealthResponse>;
	getParetoTelemetry(params?: { alpha?: number; lambda_cost?: number }): Promise<ParetoTelemetryResponse>;
	getAuditLogs(filters?: AuditQueryFilters): Promise<AuditListResponse>;
	getBenchmarksSummary(): Promise<BenchmarksSummaryResponse>;
	postQuery(payload: QueryRequest): Promise<QueryResponse>;
	postEvaluateRisk(payload: EvaluateRiskRequest): Promise<EvaluateRiskResponse>;
}

export function createGatewayClient(options?: {
	baseUrl?: string | null;
	fetchFn?: typeof fetch;
}): GatewayClient {
	const baseUrl = normalizeBaseUrl(options?.baseUrl);
	const fetchFn = options?.fetchFn ?? fetch;

	async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
		let resp: Response;
		try {
			resp = await fetchFn(`${baseUrl}${path}`, {
				...init,
				headers: {
					'Content-Type': 'application/json',
					...(init?.headers ?? {})
				}
			});
		} catch (err) {
			const msg = err instanceof Error ? err.message : 'Network error';
			throw new GatewayApiError(`Gateway unreachable at ${baseUrl}: ${msg}`, {
				status: 0,
				detail: msg,
				unreachable: true
			});
		}

		if (!resp.ok) {
			const detail = await parseErrorDetail(resp);
			throw new GatewayApiError(`Gateway request failed (${resp.status}): ${detail}`, {
				status: resp.status,
				detail,
				unreachable: false
			});
		}

		return (await resp.json()) as T;
	}

	return {
		baseUrl,
		getHealth(): Promise<HealthResponse> {
			return requestJson<HealthResponse>('/health', { method: 'GET' });
		},
		getParetoTelemetry(params?: { alpha?: number; lambda_cost?: number }): Promise<ParetoTelemetryResponse> {
			const qs = new URLSearchParams();
			if (params?.alpha !== undefined) qs.set('alpha', params.alpha.toFixed(2));
			if (params?.lambda_cost !== undefined) qs.set('lambda_cost', params.lambda_cost.toFixed(2));
			const suffix = qs.toString() ? `?${qs.toString()}` : '';
			return requestJson<ParetoTelemetryResponse>(`/v1/telemetry/pareto${suffix}`, { method: 'GET' });
		},
		getAuditLogs(filters?: AuditQueryFilters): Promise<AuditListResponse> {
			const qs = new URLSearchParams();
			if (filters?.limit !== undefined) qs.set('limit', String(filters.limit));
			if (filters?.offset !== undefined) qs.set('offset', String(filters.offset));
			if (filters?.decision) qs.set('decision', filters.decision);
			if (filters?.tier) qs.set('tier', filters.tier);
			if (filters?.cache_hit !== undefined && filters.cache_hit !== null) {
				qs.set('cache_hit', String(filters.cache_hit));
			}
			if (filters?.q && filters.q.trim()) qs.set('q', filters.q.trim());
			const suffix = qs.toString() ? `?${qs.toString()}` : '';
			return requestJson<AuditListResponse>(`/v1/telemetry/audit${suffix}`, { method: 'GET' });
		},
		getBenchmarksSummary(): Promise<BenchmarksSummaryResponse> {
			return requestJson<BenchmarksSummaryResponse>('/v1/benchmarks/summary', { method: 'GET' });
		},
		postQuery(payload: QueryRequest): Promise<QueryResponse> {
			return requestJson<QueryResponse>('/v1/query', {
				method: 'POST',
				body: JSON.stringify(payload)
			});
		},
		postEvaluateRisk(payload: EvaluateRiskRequest): Promise<EvaluateRiskResponse> {
			return requestJson<EvaluateRiskResponse>('/v1/evaluate-risk', {
				method: 'POST',
				body: JSON.stringify(payload)
			});
		}
	};
}
