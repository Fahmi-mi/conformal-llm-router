import { PUBLIC_GATEWAY_URL } from '$app/env/public';
import { clampAlpha, clampLambda, type GatewayErrorState } from './analytics';
import { createGatewayClient, DEFAULT_GATEWAY_URL, GatewayApiError } from './api';
import {
	buildSyntheticParetoTelemetry,
	SYNTHETIC_AUDIT_RECORDS,
	SYNTHETIC_BENCHMARKS_SUMMARY,
	SYNTHETIC_HEALTH
} from './synthetic';
import type {
	AuditListResponse,
	BenchmarksSummaryResponse,
	HealthResponse,
	ParetoTelemetryResponse
} from './types';

function resolveGatewayUrl(): string {
	return PUBLIC_GATEWAY_URL?.trim() || DEFAULT_GATEWAY_URL;
}

class ConsoleState {
	alpha = $state(0.1);
	lambdaCost = $state(0.45);
	syntheticPreview = $state(false);
	loading = $state(true);
	refreshing = $state(false);
	health = $state<HealthResponse | null>(null);
	paretoTelemetry = $state<ParetoTelemetryResponse | null>(null);
	benchmarks = $state<BenchmarksSummaryResponse | null>(null);
	recentAudit = $state<AuditListResponse | null>(null);
	gatewayError = $state<GatewayErrorState | null>(null);
	lastSyncedAt = $state<string | null>(null);
	mobileNavOpen = $state(false);

	private paretoTimer: ReturnType<typeof setTimeout> | null = null;
	private initialized = false;

	private clearParetoTimer(): void {
		if (this.paretoTimer) {
			clearTimeout(this.paretoTimer);
			this.paretoTimer = null;
		}
	}

	get gatewayBaseUrl(): string {
		return resolveGatewayUrl();
	}

	client() {
		return createGatewayClient({ baseUrl: resolveGatewayUrl() });
	}

	async ensureInitialized(): Promise<void> {
		if (this.initialized) return;
		this.initialized = true;
		await this.refreshAll();
	}

	async refreshAll(): Promise<void> {
		this.clearParetoTimer();
		if (this.syntheticPreview) {
			this.health = SYNTHETIC_HEALTH;
			this.paretoTelemetry = buildSyntheticParetoTelemetry(this.alpha, this.lambdaCost);
			this.benchmarks = SYNTHETIC_BENCHMARKS_SUMMARY;
			this.recentAudit = {
				items: SYNTHETIC_AUDIT_RECORDS,
				total: SYNTHETIC_AUDIT_RECORDS.length,
				limit: 8,
				offset: 0
			};
			this.gatewayError = null;
			this.loading = false;
			this.refreshing = false;
			this.lastSyncedAt = new Date().toISOString();
			return;
		}

		this.refreshing = true;
		const api = this.client();

		const [healthRes, paretoRes, benchmarksRes, auditRes] = await Promise.allSettled([
			api.getHealth(),
			api.getParetoTelemetry({ alpha: this.alpha, lambda_cost: this.lambdaCost }),
			api.getBenchmarksSummary(),
			api.getAuditLogs({ limit: 8, offset: 0 })
		]);

		let firstErr: GatewayApiError | null = null;
		const captureErr = (reason: unknown) => {
			if (!firstErr && reason instanceof GatewayApiError) {
				firstErr = reason;
			}
		};

		if (healthRes.status === 'fulfilled') {
			this.health = healthRes.value;
		} else {
			this.health = null;
			captureErr(healthRes.reason);
		}

		if (paretoRes.status === 'fulfilled') {
			this.paretoTelemetry = paretoRes.value;
		} else {
			this.paretoTelemetry = null;
			captureErr(paretoRes.reason);
		}

		if (benchmarksRes.status === 'fulfilled') {
			this.benchmarks = benchmarksRes.value;
		} else {
			this.benchmarks = null;
			captureErr(benchmarksRes.reason);
		}

		if (auditRes.status === 'fulfilled') {
			this.recentAudit = auditRes.value;
		} else {
			this.recentAudit = null;
			captureErr(auditRes.reason);
		}

		if (
			healthRes.status === 'rejected' &&
			paretoRes.status === 'rejected' &&
			benchmarksRes.status === 'rejected' &&
			auditRes.status === 'rejected'
		) {
			const err = firstErr as GatewayApiError | null;
			this.gatewayError = {
				status: err?.status ?? 0,
				detail: err?.detail ?? 'Unable to connect to FastAPI gateway.',
				unreachable: err?.unreachable ?? true
			};
		} else if (firstErr) {
			const err = firstErr as GatewayApiError;
			this.gatewayError = {
				status: err.status,
				detail: err.detail,
				unreachable: err.unreachable
			};
		} else {
			this.gatewayError = null;
		}

		this.loading = false;
		this.refreshing = false;
		this.lastSyncedAt = new Date().toISOString();
	}

	private scheduleParetoRefresh(): void {
		this.clearParetoTimer();
		if (this.syntheticPreview) {
			this.paretoTelemetry = buildSyntheticParetoTelemetry(this.alpha, this.lambdaCost);
			return;
		}
		this.paretoTimer = setTimeout(async () => {
			if (this.syntheticPreview) return;
			try {
				const data = await this.client().getParetoTelemetry({
					alpha: this.alpha,
					lambda_cost: this.lambdaCost
				});
				if (this.syntheticPreview) return;
				this.paretoTelemetry = data;
				if (this.gatewayError) {
					this.gatewayError = null;
				}
			} catch (err) {
				if (this.syntheticPreview) return;
				if (err instanceof GatewayApiError) {
					this.gatewayError = {
						status: err.status,
						detail: err.detail,
						unreachable: err.unreachable
					};
				}
			}
		}, 160);
	}

	setAlpha(next: number): void {
		this.alpha = clampAlpha(next);
		this.scheduleParetoRefresh();
	}

	setLambdaCost(next: number): void {
		this.lambdaCost = clampLambda(next);
		this.scheduleParetoRefresh();
	}

	async setSyntheticPreview(enabled: boolean): Promise<void> {
		this.clearParetoTimer();
		this.syntheticPreview = enabled;
		await this.refreshAll();
	}

	async toggleSyntheticPreview(): Promise<void> {
		await this.setSyntheticPreview(!this.syntheticPreview);
	}
}

export const consoleState = new ConsoleState();
