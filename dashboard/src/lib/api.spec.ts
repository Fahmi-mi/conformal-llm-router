import { describe, expect, it } from 'vitest';
import { createGatewayClient, GatewayApiError, normalizeBaseUrl } from './api';

describe('gateway api client', () => {
	it('normalizes trailing slashes and defaults empty base urls', () => {
		expect(normalizeBaseUrl('http://localhost:8000/')).toBe('http://localhost:8000');
		expect(normalizeBaseUrl('')).toBe('http://localhost:8000');
		expect(normalizeBaseUrl(null)).toBe('http://localhost:8000');
	});

	it('serializes pareto and audit query parameters cleanly', async () => {
		const calls: string[] = [];
		const client = createGatewayClient({
			baseUrl: 'http://127.0.0.1:8000/',
			fetchFn: async (input) => {
				calls.push(String(input));
				return new Response(JSON.stringify({ items: [], total: 0, limit: 25, offset: 0 }), {
					status: 200,
					headers: { 'Content-Type': 'application/json' }
				});
			}
		});

		await client.getParetoTelemetry({ alpha: 0.1, lambda_cost: 0.45 });
		await client.getAuditLogs({
			limit: 25,
			offset: 50,
			decision: 'ABSTAIN',
			tier: 'tier2',
			cache_hit: false,
			q: 'lease'
		});

		expect(calls[0]).toBe('http://127.0.0.1:8000/v1/telemetry/pareto?alpha=0.10&lambda_cost=0.45');
		expect(calls[1]).toBe(
			'http://127.0.0.1:8000/v1/telemetry/audit?limit=25&offset=50&decision=ABSTAIN&tier=tier2&cache_hit=false&q=lease'
		);
	});

	it('surfaces 503 FastAPI detail and unreachable network errors cleanly', async () => {
		const client503 = createGatewayClient({
			baseUrl: 'http://localhost:8000',
			fetchFn: async () =>
				new Response(JSON.stringify({ detail: 'Gateway model artifacts are not loaded.' }), {
					status: 503,
					headers: { 'Content-Type': 'application/json' }
				})
		});

		await expect(client503.postQuery({ query: 'hello' })).rejects.toMatchObject({
			status: 503,
			detail: 'Gateway model artifacts are not loaded.',
			unreachable: false
		});

		const clientDown = createGatewayClient({
			baseUrl: 'http://localhost:8000',
			fetchFn: async () => {
				throw new TypeError('fetch failed');
			}
		});

		try {
			await clientDown.getHealth();
			expect.unreachable();
		} catch (err) {
			expect(err).toBeInstanceOf(GatewayApiError);
			expect((err as GatewayApiError).unreachable).toBe(true);
			expect((err as GatewayApiError).status).toBe(0);
		}
	});
});
