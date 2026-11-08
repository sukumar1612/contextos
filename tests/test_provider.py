"""The single-server provider: request body, retries, accounting and waiting through a server restart."""

import json
import time as clock

import httpx
import pytest

from contextos.config import Config
from contextos.provider import EpisodeProvider, Provider, api_key
from contextos.tokens import Counter


def provider_config(tmp_path, **overrides) -> Config:
    cfg = Config()
    cfg.provider.api_key_env, cfg.provider.api_key_file = "TESTKEY", None
    cfg.context.input_tokens, cfg.context.margin_tokens = 100_000, 0
    for k, v in overrides.items():
        setattr(cfg.provider, k, v)
    return cfg


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("TESTKEY", "secret-0")


def ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "qwen3.6-35b-a3b",
            "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 2},
        },
    )


def test_api_key_comes_from_the_environment_or_a_file(tmp_path, monkeypatch):
    cfg = provider_config(tmp_path)
    monkeypatch.delenv("TESTKEY", raising=False)
    with pytest.raises(RuntimeError, match="No API key"):
        api_key(cfg)
    (tmp_path / "api_key").write_text("file-secret\n")
    cfg.provider.api_key_file = str(tmp_path / "api_key")
    assert api_key(cfg) == "file-secret"
    monkeypatch.setenv("TESTKEY", "env-secret")
    assert api_key(cfg) == "env-secret"  # the variable wins when both exist


def test_provider_sends_the_template_switches_and_records_the_key_name(tmp_path, key):
    bodies, headers = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        headers.append(request.headers["authorization"])
        return ok_response()

    cfg = provider_config(tmp_path, reasoning_effort="low", chat_template_kwargs={"enable_thinking": False})
    provider = Provider(cfg, Counter(cfg.context), transport=httpx.MockTransport(handler))
    reply = provider.complete([{"role": "user", "content": "hi"}], "foreground", max_tokens=8)
    provider.close()
    assert reply.message["content"] == "OK" and reply.key == "TESTKEY" and headers == ["Bearer secret-0"]
    assert bodies[-1]["reasoning_effort"] == "low" and bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert bodies[-1]["model"] == "qwen3.6-35b-a3b" and bodies[-1]["max_tokens"] == 8


def test_transient_and_server_errors_are_retried_and_charged_conservatively(tmp_path, key):
    statuses = iter([429, 503])

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            return httpx.Response(next(statuses), text="busy")
        except StopIteration:
            return ok_response()

    cfg = provider_config(tmp_path, retries=3)
    provider = Provider(cfg, Counter(cfg.context), transport=httpx.MockTransport(handler))
    provider.sleep = lambda seconds: None
    tracked = EpisodeProvider(provider, cfg, "fixture")
    reply = tracked.complete([{"role": "user", "content": "hi"}], "foreground", max_tokens=8)
    provider.close()
    records = tracked.resources()["requests"]
    assert [r["status"] for r in records] == ["transient", "unknown", "complete"]
    assert records[0]["cost_usd"] == 0.0 and records[1]["cost_usd"] > 0  # a 5xx may have run: worst case charged
    assert reply.cost_usd == pytest.approx(sum(r["cost_usd"] for r in records))


def test_rejected_requests_are_not_retried(tmp_path, key):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="unauthorized")

    cfg = provider_config(tmp_path, retries=3)
    provider = Provider(cfg, Counter(cfg.context), transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="HTTP 401"):
        provider.complete([{"role": "user", "content": "hi"}], "foreground", max_tokens=8)
    provider.close()
    assert len(calls) == 1


def test_requests_wait_for_a_replaced_server_named_in_the_endpoint_file(tmp_path, key, monkeypatch):
    """When the vLLM job ends, requests stall until a replacement answers (its address arrives through
    endpoint_file) instead of failing the episode; no retry attempt is consumed by the outage."""
    state = {"up": False}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.url.path))
        alive = state["up"] and request.url.host == "new"
        if not alive:
            raise httpx.ConnectError("Connection refused", request=request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return ok_response()

    endpoint = tmp_path / "endpoint"
    cfg = provider_config(
        tmp_path, retries=1, base_url="http://old/v1", endpoint_file=str(endpoint), server_wait_seconds=60
    )
    provider = Provider(cfg, Counter(cfg.context), transport=httpx.MockTransport(handler))
    provider.server_poll_seconds = 0
    polls = []

    def tick(seconds):
        polls.append(seconds)
        if len(polls) == 3:  # the replacement comes up and the driver publishes its address
            endpoint.write_text("http://new/v1\n")
            state["up"] = True

    provider.sleep = tick
    reply = provider.complete([{"role": "user", "content": "hi"}], "foreground", max_tokens=8)
    provider.close()
    assert reply.message["content"] == "OK" and len(polls) >= 3
    assert ("old", "/v1/chat/completions") in seen and ("new", "/v1/chat/completions") in seen
    assert provider.base == "http://new/v1/"


def test_requests_fail_when_no_server_returns_in_time(tmp_path, key, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    cfg = provider_config(tmp_path, retries=1, server_wait_seconds=0.01)
    provider = Provider(cfg, Counter(cfg.context), transport=httpx.MockTransport(handler))
    provider.server_poll_seconds = 0
    provider.sleep = lambda seconds: None
    monkeypatch.setattr(clock, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="unreachable"):
        provider.complete([{"role": "user", "content": "hi"}], "foreground", max_tokens=8)
    provider.close()


def test_probe_reports_the_served_model(tmp_path, key):
    cfg = provider_config(tmp_path)
    provider = Provider(cfg, Counter(cfg.context), transport=httpx.MockTransport(lambda request: ok_response()))
    result = provider.probe()
    provider.close()
    assert result["ok"] and result["model"] == "qwen3.6-35b-a3b" and result["reply"] == "OK"
