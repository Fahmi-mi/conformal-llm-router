import { defineEnvVars } from '@sveltejs/kit/env';

export const variables = defineEnvVars({
	PUBLIC_GATEWAY_URL: {
		public: true,
		schema: (value) => value?.trim() || 'http://localhost:8000',
		description: 'Base URL for the FastAPI inference gateway (e.g. http://localhost:8000).'
	}
});
