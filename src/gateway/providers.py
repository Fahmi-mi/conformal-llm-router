"""Unified async LLM provider router with rate-limit/error fallback and cost estimation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# USD per 1M tokens (input, output)
_PROVIDER_RATES_PER_M: dict[str, tuple[float, float]] = {
    "groq": (0.05, 0.08),
    "gemini": (0.075, 0.30),
    "ollama": (0.0, 0.0),
    "deepseek": (0.27, 1.10),
}

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
_GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class UpstreamUnavailableError(RuntimeError):
    """Raised when all configured providers for a tier fail or none are configured."""


class GatewaySettings(BaseSettings):
    """Runtime settings for the inference gateway loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str | None = "http://localhost:11434"

    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-2.0-flash-lite"
    ollama_model: str = "llama3.2:3b"
    deepseek_model: str = "deepseek-chat"

    gateway_alpha_default: float = Field(default=0.10, ge=0.01, le=0.30)
    gateway_lambda_default: float = Field(default=0.45, ge=0.0, le=1.0)
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(_DEFAULT_CORS_ORIGINS)
    )

    artifacts_dir: Path = Path("artifacts")
    telemetry_db_path: Path = Path("data/telemetry.db")
    faiss_index_path: Path = Path("data/faiss.index")
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)

    @field_validator(
        "deepseek_api_key",
        "groq_api_key",
        "gemini_api_key",
        "ollama_base_url",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None:
            return list(_DEFAULT_CORS_ORIGINS)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return list(_DEFAULT_CORS_ORIGINS)
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]


@dataclass(slots=True)
class ProviderResult:
    """Normalized LLM completion result across Tier-1 and Tier-2 providers."""

    answer: str
    model_used: str
    provider: str
    tier: Literal["tier1", "tier2"]
    fallback_used: bool = False
    estimated_cost_usd: float = 0.0
    frontier_cost_usd: float = 0.0


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _extract_text(raw: Any, provider: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Provider '{provider}' returned null or empty completion.")
    return raw.strip()


def _summarize_provider_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return f"transport error ({type(exc).__name__})"
    return type(exc).__name__


def _compute_cost_usd(provider: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _PROVIDER_RATES_PER_M[provider]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0


def _build_messages(query: str, context: str | None) -> tuple[str, list[dict[str, str]]]:
    if context is not None and context.strip():
        system_prompt = (
            "Answer the question concisely and accurately using only the provided context."
        )
        user_prompt = f"Context:\n{context.strip()}\n\nQuestion:\n{query}"
    else:
        system_prompt = "You are a helpful, concise assistant."
        user_prompt = query
    return user_prompt, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class ProviderRouter:
    """Async provider dispatcher with Tier-1 fallback (Groq -> Gemini -> Ollama) and DeepSeek Tier 2."""

    def __init__(
        self,
        settings: GatewaySettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or GatewaySettings()
        self._client = client

    def available_providers(self) -> dict[str, bool]:
        """Return configuration readiness flags for each supported upstream provider."""
        return {
            "groq": self.settings.groq_api_key is not None,
            "gemini": self.settings.gemini_api_key is not None,
            "ollama": self.settings.ollama_base_url is not None,
            "deepseek": self.settings.deepseek_api_key is not None,
        }

    def _tier_chain(self, tier: Literal["tier1", "tier2"]) -> list[str]:
        avail = self.available_providers()
        candidates = ["groq", "gemini", "ollama"] if tier == "tier1" else ["deepseek"]
        return [name for name in candidates if avail[name]]

    async def _call_openai_compat(
        self,
        client: httpx.AsyncClient,
        *,
        provider: str,
        url: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        prompt_text: str,
    ) -> tuple[str, str, int, int]:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = _extract_text(data["choices"][0]["message"].get("content"), provider)
        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or _approx_tokens(prompt_text))
        out_tok = int(usage.get("completion_tokens") or _approx_tokens(content))
        return content, model, in_tok, out_tok

    async def _call_gemini(
        self,
        client: httpx.AsyncClient,
        *,
        user_prompt: str,
        system_prompt: str,
    ) -> tuple[str, str, int, int]:
        assert self.settings.gemini_api_key is not None
        model = self.settings.gemini_model
        url = _GEMINI_URL_TEMPLATE.format(model=model)
        resp = await client.post(
            url,
            headers={"x-goog-api-key": self.settings.gemini_api_key},
            json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        text_chunks = [
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        content = _extract_text("".join(text_chunks), "gemini")
        usage = data.get("usageMetadata") or {}
        in_tok = int(usage.get("promptTokenCount") or _approx_tokens(user_prompt))
        out_tok = int(usage.get("candidatesTokenCount") or _approx_tokens(content))
        return content, model, in_tok, out_tok

    async def _call_ollama(
        self,
        client: httpx.AsyncClient,
        *,
        messages: list[dict[str, str]],
        prompt_text: str,
    ) -> tuple[str, str, int, int]:
        assert self.settings.ollama_base_url is not None
        model = self.settings.ollama_model
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        resp = await client.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = _extract_text(data["message"].get("content"), "ollama")
        in_tok = int(data.get("prompt_eval_count") or _approx_tokens(prompt_text))
        out_tok = int(data.get("eval_count") or _approx_tokens(content))
        return content, model, in_tok, out_tok

    async def _dispatch_provider(
        self,
        client: httpx.AsyncClient,
        provider: str,
        query: str,
        context: str | None,
    ) -> tuple[str, str, int, int]:
        user_prompt, messages = _build_messages(query, context)
        prompt_text = f"{messages[0]['content']}\n{user_prompt}"

        if provider == "groq":
            assert self.settings.groq_api_key is not None
            return await self._call_openai_compat(
                client,
                provider="groq",
                url=_GROQ_URL,
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_model,
                messages=messages,
                prompt_text=prompt_text,
            )
        if provider == "deepseek":
            assert self.settings.deepseek_api_key is not None
            return await self._call_openai_compat(
                client,
                provider="deepseek",
                url=_DEEPSEEK_URL,
                api_key=self.settings.deepseek_api_key,
                model=self.settings.deepseek_model,
                messages=messages,
                prompt_text=prompt_text,
            )
        if provider == "gemini":
            return await self._call_gemini(
                client,
                user_prompt=user_prompt,
                system_prompt=messages[0]["content"],
            )
        if provider == "ollama":
            return await self._call_ollama(
                client,
                messages=messages,
                prompt_text=prompt_text,
            )
        raise ValueError(f"Unsupported provider: {provider}")

    async def generate(
        self,
        tier: Literal["tier1", "tier2"],
        query: str,
        context: str | None = None,
    ) -> ProviderResult:
        """Generate an answer for the requested tier with automatic fallback on errors/429."""
        chain = self._tier_chain(tier)
        if not chain:
            raise UpstreamUnavailableError(f"No upstream providers configured for tier '{tier}'.")

        errors: list[str] = []
        if self._client is not None:
            return await self._run_chain(self._client, tier, chain, query, context, errors)

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            return await self._run_chain(client, tier, chain, query, context, errors)

    async def _run_chain(
        self,
        client: httpx.AsyncClient,
        tier: Literal["tier1", "tier2"],
        chain: list[str],
        query: str,
        context: str | None,
        errors: list[str],
    ) -> ProviderResult:
        for idx, provider in enumerate(chain):
            try:
                answer, model_used, in_tok, out_tok = await self._dispatch_provider(
                    client, provider, query, context
                )
                return ProviderResult(
                    answer=answer,
                    model_used=model_used,
                    provider=provider,
                    tier=tier,
                    fallback_used=idx > 0,
                    estimated_cost_usd=_compute_cost_usd(provider, in_tok, out_tok),
                    frontier_cost_usd=_compute_cost_usd("deepseek", in_tok, out_tok),
                )
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                summary = _summarize_provider_error(exc)
                logger.warning("Provider '%s' failed for tier '%s': %s", provider, tier, summary)
                errors.append(f"{provider}: {summary}")

        raise UpstreamUnavailableError(
            f"All providers failed for tier '{tier}': {'; '.join(errors)}"
        )
