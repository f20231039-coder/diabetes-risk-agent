"""One small interface over LLM providers so they can be swapped by env var.

Groq, Gemini, Ollama and OpenAI all expose an OpenAI-compatible
/chat/completions endpoint with tool calling, so one client covers them.
LLM_PROVIDER=rules gives an offline regex baseline (no key needed; used in CI).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx

log = logging.getLogger("health_agent.llm")

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-3.5-flash"),
    "ollama": ("http://localhost:11434/v1", None, "qwen3.5:4b"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
}


# Backup models used when the primary is overloaded. Set LLM_FALLBACK_MODELS=""
# to disable (evals pin one model so the numbers belong to that model).
DEFAULT_FALLBACKS = {
    "gemini": "gemini-3.5-flash-lite,gemini-3-flash-preview",
    "groq": "llama-3.1-8b-instant",
}


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model
    # Provider-specific data that must be echoed back in the history
    # (e.g. Gemini 3's extra_content.google.thought_signature).
    extra_content: dict | None = None

    def as_message_part(self) -> dict:
        part = {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}
        if self.extra_content:
            part["extra_content"] = self.extra_content
        return part


@dataclass
class LLMResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    name: str

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> LLMResult: ...


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_text(text: str | None) -> str:
    """Strip reasoning blocks some local models emit."""
    return _THINK_RE.sub("", text or "").strip()


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 1,
        extra_body: dict | None = None,
        name: str = "openai-compat",
        fallback_models: list[str] | None = None,
        temperature: float = 0.0,
    ):
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Tried in order when the primary model is overloaded (429/5xx/timeout).
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model]
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.name = f"{name}:{model}"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    async def chat(self, messages, tools=None, json_mode=False) -> LLMResult:
        body: dict = {"model": self.model, "messages": messages, "temperature": self.temperature, **self.extra_body}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        last_error: Exception | None = None
        models = [self.model, *self.fallback_models]
        for attempt in range(self.max_retries + 1 + len(self.fallback_models)):
            body["model"] = models[min(attempt, len(models) - 1)] if self.fallback_models else self.model
            try:
                resp = await self._client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"provider returned {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    # 4xx other than 429 will not succeed on retry.
                    raise LLMError(f"provider rejected request ({resp.status_code}): {resp.text[:300]}")
                return self._parse(resp.json())
            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                retryable = not (isinstance(exc, LLMError) and "rejected request" in str(exc))
                log.warning("LLM call failed (attempt %d, model %s): %s", attempt + 1, body["model"], (str(exc) or type(exc).__name__)[:120])
                if attempt < self.max_retries + len(self.fallback_models) and retryable:
                    await asyncio.sleep(2.0 * 2**attempt)  # 2s, 4s, 8s ... (exponential backoff)
                    continue
                break
        raise LLMError(str(last_error) or type(last_error).__name__)

    @staticmethod
    def _parse(data: dict) -> LLMResult:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from exc
        calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "{}",
                extra_content=tc.get("extra_content"),
            )
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        usage = data.get("usage") or {}
        return LLMResult(
            content=clean_text(msg.get("content")),
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )


def load_env_file(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines from the project's .env (git- and docker-ignored).
    Real environment variables win, so Docker/CI secrets are never overridden."""
    from pathlib import Path

    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def make_client(provider: str | None = None, model: str | None = None, temperature: float = 0.0) -> LLMClient:
    load_env_file()
    provider = (provider or os.environ.get("LLM_PROVIDER") or "").lower()
    if not provider:
        # Pick the first provider with a key, else the offline baseline.
        provider = next((p for p, (_, key, _) in PROVIDERS.items() if key and os.environ.get(key)), "rules")
    if provider == "rules":
        from .rules import RuleBasedClient

        return RuleBasedClient()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM_PROVIDER {provider!r}; use one of {sorted(PROVIDERS) + ['rules']}")
    base_url, key_env, default_model = PROVIDERS[provider]
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise ValueError(f"LLM_PROVIDER={provider} needs the {key_env} environment variable")
    extra: dict = {}
    if provider == "ollama":
        # Qwen 3.x "thinking" is slow and not needed for extraction/tool calls.
        extra["reasoning_effort"] = os.environ.get("OLLAMA_REASONING", "none")
    return OpenAICompatClient(
        base_url=os.environ.get("LLM_BASE_URL", base_url),
        model=model or os.environ.get("LLM_MODEL", default_model),
        api_key=api_key,
        timeout=float(os.environ.get("LLM_TIMEOUT", "240" if provider == "ollama" else "30")),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", "2")),
        fallback_models=os.environ.get("LLM_FALLBACK_MODELS", DEFAULT_FALLBACKS.get(provider, "")).split(","),
        temperature=temperature,
        extra_body=extra,
        name=provider,
    )


# Models offered in the chat page's settings, per provider (only providers
# whose key is configured are shown). Override with LLM_CHOICES_<PROVIDER>.
MODEL_CHOICES = {
    "gemini": ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-3.8-flash"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "openai": ["gpt-4o-mini"],
    "ollama": ["qwen3.5:4b", "qwen3.5:9b"],
}


def configured_llm_choices() -> list[dict]:
    """LLM options the server can actually use, for the settings dropdown."""
    load_env_file()
    out = []
    for provider, (_, key_env, _) in PROVIDERS.items():
        enabled = bool(os.environ.get(key_env)) if key_env else os.environ.get("LLM_PROVIDER") == "ollama" or bool(os.environ.get("OLLAMA_ENABLED"))
        if not enabled:
            continue
        env_models = os.environ.get(f"LLM_CHOICES_{provider.upper()}")
        for m in (env_models.split(",") if env_models else MODEL_CHOICES.get(provider, [])):
            out.append({"id": f"{provider}:{m}", "provider": provider, "model": m, "label": f"{provider.title()} · {m}"})
    out.append({"id": "rules:regex-baseline", "provider": "rules", "model": "regex-baseline", "label": "Offline regex baseline (no LLM)"})
    return out


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences / chatter."""
    text = clean_text(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj
