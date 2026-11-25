# Conformal LLM Router — Monitoring Console (`dashboard/`)

SvelteKit + Tailwind CSS console (`@sveltejs/adapter-node`, Node 22) for inspecting conformal hallucination control, Pareto cost-quality routing, audit telemetry, live inference, and offline benchmarks.

## Configuration

- `PUBLIC_GATEWAY_URL` (default: `http://localhost:8000`) — FastAPI gateway base URL read at runtime (`src/env.ts` -> `$app/env/public`).

## Commands

```sh
npm install
npm run check     # svelte-kit sync && svelte-check
npm test          # Vitest unit tests
npm run dev       # Vite dev server on http://localhost:5173
npm run build     # Production Node build -> build/index.js
```

## Docker

```sh
docker build -t conformal-router-dashboard ./dashboard
docker run --rm -p 3000:3000 -e PUBLIC_GATEWAY_URL=http://localhost:8000 conformal-router-dashboard
```
