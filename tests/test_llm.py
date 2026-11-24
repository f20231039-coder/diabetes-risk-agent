"""LLM client tests with a mocked HTTP transport (no network)."""

import asyncio

import httpx
import pytest

from health_agent import llm
from health_agent.llm import LLMError, OpenAICompatClient, parse_json_object

OK = {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


def client_with(handler, **kw):
    c = OpenAICompatClient(base_url="https://x.test/v1", model="primary", **kw)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def fast(_):
        return None

    monkeypatch.setattr(llm.asyncio, "sleep", fast)


def test_falls_back_to_backup_model_on_503():
    seen = []

    def handler(request):
        model = __import__("json").loads(request.content)["model"]
        seen.append(model)
        return httpx.Response(503, text="busy") if model == "primary" else httpx.Response(200, json=OK)

    res = asyncio.run(client_with(handler, fallback_models=["backup"]).chat([{"role": "user", "content": "hi"}]))
    assert res.content == "hello" and res.prompt_tokens == 5
    assert seen == ["primary", "backup"]


def test_retries_then_raises():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, text="down")

    with pytest.raises(LLMError):
        asyncio.run(client_with(handler, max_retries=1).chat([{"role": "user", "content": "hi"}]))
    assert len(calls) == 2  # one try + one retry


def test_400_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="bad request")

    with pytest.raises(LLMError):
        asyncio.run(client_with(handler, max_retries=2).chat([{"role": "user", "content": "hi"}]))
    assert len(calls) == 1


def test_parse_json_tolerates_fences_and_think_blocks():
    assert parse_json_object('<think>hmm</think>```json\n{"age": 45}\n```') == {"age": 45}
    assert parse_json_object('Sure! {"bmi": 29} hope that helps') == {"bmi": 29}
