"""OpenAI-compatible chat completions against one vLLM server, with per-episode request accounting."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import httpx

from contextos.agent import SUMMARY_HEADINGS
from contextos.config import Config
from contextos.tokens import Counter, api_messages
from contextos.types import ContextLimit, Reply, Usage

log = logging.getLogger("contextos")

# A logical call may make several HTTP attempts. The episode wrapper sets this on
# its calling thread, so foreground and background calls remain correctly paired
# even when many episodes share one Provider.
_attempt_records: ContextVar[list[dict] | None] = ContextVar("request_attempt_records", default=None)


def record_attempt(
    started: float,
    *,
    status: str,
    key: str = "",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float = 0.0,
    estimated_cost_usd: float = 0.0,
    **extra: Any,
) -> None:
    records = _attempt_records.get()
    if records is not None:
        ended = time.time()
        records.append(
            {
                "started_at": started,
                "ended_at": ended,
                "duration_seconds": ended - started,
                "status": status,
                "key": key,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "estimated_cost_usd": estimated_cost_usd,
                **extra,
            }
        )


class EpisodeProvider:
    """Attach usage to one episode without changing the shared provider's API.

    Records physical requests, including rejected and ambiguous retries. Unknown
    charges remain estimates; token-price arithmetic is never labelled an invoice.
    """

    def __init__(self, provider: Provider | FakeProvider, cfg: Config, name: str):
        self.provider, self.cfg, self.name = provider, cfg, name
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.calls = 0

    def complete(
        self,
        messages: list[dict],
        purpose: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        step: int = 0,
        *,
        source: int | None = None,
        summary_job: str | None = None,
    ) -> Reply:
        with self.lock:
            call = self.calls
            self.calls += 1
        records: list[dict] = []
        token = _attempt_records.set(records)
        started = time.time()
        try:
            reply = self.provider.complete(messages, purpose, tools, max_tokens, step)
            if not records:  # FakeProvider and deliberately small test doubles.
                record_attempt(
                    started,
                    status="complete",
                    key=reply.key,
                    input_tokens=reply.usage.input_tokens,
                    output_tokens=reply.usage.output_tokens,
                    cost_usd=reply.cost_usd,
                    usage_estimated=self.cfg.provider.kind == "fake",
                )
            return reply
        finally:
            _attempt_records.reset(token)
            if not records:
                # No HTTP attempt means, for example, a local guard rejected the call.
                # Do not invent tokens or charges for it.
                records.append(
                    {
                        "started_at": started,
                        "ended_at": time.time(),
                        "duration_seconds": time.time() - started,
                        "status": "not_sent",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "estimated_cost_usd": 0.0,
                    }
                )
            for attempt, record in enumerate(records):
                record.update(
                    request_id=f"{self.name}:{call}:{attempt}",
                    call_id=call,
                    attempt=attempt,
                    purpose=purpose,
                    source=source,
                    summary_job=summary_job,
                    call_started_at=started,
                    call_ended_at=time.time(),
                )
            with self.lock:
                self.requests.extend(records)

    def job_cost(self, job: str) -> float:
        with self.lock:
            return sum(r["cost_usd"] for r in self.requests if r["summary_job"] == job)

    def resources(self) -> dict:
        with self.lock:
            requests = [dict(r) for r in self.requests]
        usage: dict[str, Any] = {}
        for label, purpose in (("coding", "foreground"), ("summary", "summary"), ("all", None)):
            selected = [r for r in requests if purpose is None or r["purpose"] == purpose]
            usage[label] = {
                field: sum(r.get(field) or 0 for r in selected) for field in ("input_tokens", "output_tokens")
            }
        usage["estimated_requests"] = sum(bool(r.get("usage_estimated")) for r in requests)
        usage["unknown_usage_requests"] = sum(
            r.get("input_tokens") is None or r.get("output_tokens") is None for r in requests
        )
        cost = sum(r["cost_usd"] for r in requests)
        estimated = sum(r["estimated_cost_usd"] for r in requests)
        return {
            "requests": requests,
            "usage": usage,
            "cost_usd": cost,
            "reference_cost_usd": cost - estimated,
            "estimated_cost_usd": estimated,
            "actual_cost_usd": None,  # no billing receipts exist for a self-hosted server
            "cost_kind": "reference",
            "token_unit": "fixture_byte_estimate" if self.cfg.provider.kind == "fake" else "tokens",
        }


def classify(status: int) -> str:
    """transient: nothing ran, try again. unknown: the server may have run it. rejected: do not retry."""
    if status == 429:
        return "transient"
    if status >= 500:
        return "unknown"
    return "rejected"


def reference_cost(cfg: Config, usage: Usage) -> float:
    """Token prices in USD; only the reward uses them (a self-hosted server has no invoice)."""
    p = cfg.provider
    return (usage.input_tokens * p.input_price + usage.output_tokens * p.output_price) / 1_000_000


def api_key(cfg: Config) -> str:
    """The bearer token: the named environment variable, else the named file (for example ~/vllm/api_key)."""
    p = cfg.provider
    key = os.environ.get(p.api_key_env, "").strip()
    if not key and p.api_key_file:
        path = Path(p.api_key_file).expanduser()
        if path.is_file():
            key = path.read_text().strip()
    if not key:
        raise RuntimeError(f"No API key: set {p.api_key_env} or provider.api_key_file")
    return key


class Provider:
    """One HTTP client for the configured server. Coding and summary requests share a concurrency limit;
    a request whose server stopped answering waits for it (or a replacement named in endpoint_file)."""

    def __init__(self, cfg: Config, counter: Counter, transport: httpx.BaseTransport | None = None):
        self.cfg, self.counter = cfg, counter
        p = cfg.provider
        if p.max_concurrent_requests < 1:
            raise ValueError("provider.max_concurrent_requests must be positive")
        self.base = p.base_url.rstrip("/") + "/"
        self.key_name = p.api_key_env  # recorded with every request; the secret itself is never written
        self.client = httpx.Client(
            base_url=self.base,
            headers={"Authorization": f"Bearer {api_key(cfg)}"},
            timeout=p.timeout_seconds,
            transport=transport,
        )
        self.slots = threading.BoundedSemaphore(p.max_concurrent_requests)
        self.sleep = time.sleep  # retry backoff; tests replace it
        self.server_poll_seconds = 15.0
        self._waiting, self._wait_lock = 0, threading.Lock()

    def cost(self, usage: Usage) -> float:
        return reference_cost(self.cfg, usage)

    # ---- server availability ------------------------------------------------------------------------
    def reachable(self) -> bool:
        """Does the server answer at all? A cheap listing, not a completion."""
        try:
            return self.client.get("models", timeout=5.0).status_code < 500
        except httpx.TransportError:
            return False

    def refresh_endpoint(self) -> bool:
        """Point the client at the address in endpoint_file when it changed (a replaced server)."""
        path = self.cfg.provider.endpoint_file
        if not path:
            return False
        try:
            lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
        except OSError:
            return False
        if not lines:
            return False
        url = lines[-1].removeprefix("BASE=").strip("'\"")
        base = url.rstrip("/") + "/"
        if not url.startswith("http") or base == self.base:
            return False
        self.client.base_url = base  # type: ignore[assignment]
        self.base = base
        log.warning("provider endpoint changed to %s", base)
        return True

    def await_server(self) -> None:
        """Block until the server, or its replacement named in endpoint_file, answers; raise after
        server_wait_seconds. A vLLM job hitting its time limit stalls the episodes in flight, not fails them."""
        wait = self.cfg.provider.server_wait_seconds
        deadline = time.monotonic() + wait
        with self._wait_lock:
            self._waiting += 1
            if self._waiting == 1:
                log.warning("server unreachable; requests wait up to %.0f s for it (or a replacement) to answer", wait)
        try:
            while True:
                self.refresh_endpoint()
                if self.reachable():
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Server unreachable for {wait:.0f} s")
                self.sleep(self.server_poll_seconds)
        finally:
            with self._wait_lock:
                self._waiting -= 1
                if self._waiting == 0:
                    log.info("server answers again")

    # ---- requests ----------------------------------------------------------------------------------
    def _body(self, messages: list[dict], purpose: str, tools: list[dict] | None, max_tokens: int) -> dict:
        p = self.cfg.provider
        body: dict[str, Any] = {
            "model": p.model,
            "messages": api_messages(messages),
            "stream": False,
            "temperature": 0.0 if purpose == "summary" else p.agent_temperature,
            "max_tokens": max_tokens,
        }
        if p.reasoning_effort is not None:
            body["reasoning_effort"] = p.reasoning_effort
        if p.chat_template_kwargs:
            body["chat_template_kwargs"] = p.chat_template_kwargs
        if tools:
            body.update(tools=tools, tool_choice="auto")
        return body

    def complete(
        self,
        messages: list[dict],
        purpose: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        step: int = 0,
    ) -> Reply:
        p = self.cfg.provider
        guarded = self.counter.prompt(messages, tools)
        if guarded > self.cfg.context.input_tokens:
            raise ContextLimit(f"Prompt estimate {guarded} exceeds the cap")
        maximum = max_tokens or p.output_tokens
        body = self._body(messages, purpose, tools, maximum)
        estimate = self.cost(Usage(guarded, maximum))  # worst case; charged when the outcome stays unknown
        attempt, last, unknown_cost = 0, "", 0.0
        while attempt <= p.retries:
            started = time.time()
            try:
                with self.slots:  # coding and background summaries share this limit
                    response = self.client.post("chat/completions", json=body)
            except httpx.TransportError as exc:
                unknown_cost += estimate
                record_attempt(
                    started,
                    status="transport_unknown",
                    key=self.key_name,
                    cost_usd=estimate,
                    estimated_cost_usd=estimate,
                    error_type=type(exc).__name__,
                )
                last = f"{type(exc).__name__}: {exc}"
                if p.server_wait_seconds > 0 and not self.reachable():
                    self.await_server()  # a restarted server costs no attempt; the request goes again
                    continue
                attempt += 1
                self.sleep(min(30.0, 2.0**attempt))
                continue
            if response.status_code == 200:
                return self._reply(response, started, guarded, maximum, unknown_cost, estimate)
            kind = classify(response.status_code)
            record_attempt(
                started,
                status=kind,
                key=self.key_name,
                http_status=response.status_code,
                input_tokens=None if kind == "unknown" else 0,
                output_tokens=None if kind == "unknown" else 0,
                cost_usd=estimate if kind == "unknown" else 0.0,
                estimated_cost_usd=estimate if kind == "unknown" else 0.0,
            )
            last = f"HTTP {response.status_code}: {response.text[:160]}"
            if kind == "rejected":
                raise RuntimeError(last)
            if kind == "unknown":
                unknown_cost += estimate
            attempt += 1
            self.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(f"Request failed after {p.retries + 1} attempts; last: {last}")

    def _reply(
        self, response: httpx.Response, started: float, guarded: int, maximum: int, unknown_cost: float, estimate: float
    ) -> Reply:
        try:
            raw = response.json()
            if not isinstance(raw, dict):
                raise ValueError("Response must be an object")
            choice, reported = raw["choices"][0], raw.get("usage") or {}
            if not isinstance(choice, dict) or not isinstance(reported, dict):
                raise ValueError("Invalid choice or usage object")
            reported_input, reported_output = reported.get("prompt_tokens"), reported.get("completion_tokens")
            for count in (reported_input, reported_output):
                if count is not None and (not isinstance(count, int) or count < 0):
                    raise ValueError("Invalid token usage")
            usage = Usage(
                guarded if reported_input is None else reported_input,
                maximum if reported_output is None else reported_output,
            )
            message = choice["message"]
            if not isinstance(message, dict):
                raise ValueError("Message must be an object")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            record_attempt(
                started, status="malformed_unknown", key=self.key_name, cost_usd=estimate, estimated_cost_usd=estimate
            )
            raise RuntimeError(f"Malformed response: {exc}") from exc
        cost = self.cost(usage)
        estimated = self.cost(
            Usage(guarded if reported_input is None else 0, maximum if reported_output is None else 0)
        )
        details = reported.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            details = {}
        record_attempt(
            started,
            status="complete",
            key=self.key_name,
            input_tokens=reported_input,
            output_tokens=reported_output,
            cost_usd=cost,
            estimated_cost_usd=estimated,
            usage_estimated=reported_input is None or reported_output is None,
            cached_input_tokens=details.get("cached_tokens"),
            http_status=200,
        )
        return Reply(message, usage, choice.get("finish_reason") or "stop", cost + unknown_cost, self.key_name)

    def probe(self) -> dict:
        """One tiny request, for `contextos probe`: reachability, latency and the served model."""
        body = self._body([{"role": "user", "content": "Reply with the single word OK."}], "foreground", None, 8)
        body["temperature"] = 0.0
        started = time.time()
        try:
            response = self.client.post("chat/completions", json=body)
        except httpx.TransportError as exc:
            return {"ok": False, "endpoint": self.base, "error": f"{type(exc).__name__}: {exc}"[:120]}
        seconds = round(time.time() - started, 1)
        if response.status_code != 200:
            return {
                "ok": False,
                "endpoint": self.base,
                "status": response.status_code,
                "seconds": seconds,
                "error": response.text[:120],
            }
        raw = response.json()
        usage = raw.get("usage") or {}
        cost = self.cost(Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))))
        return {
            "ok": True,
            "endpoint": self.base,
            "seconds": seconds,
            "model": raw.get("model"),
            "reply": (raw["choices"][0]["message"].get("content") or "").strip()[:20],
            "cost_usd": round(cost, 6),
        }

    def close(self) -> None:
        self.client.close()


class FakeProvider:
    """Offline fixture: eight bash calls then submit; summaries always carry every heading."""

    def __init__(self, cfg: Config, counter: Counter):
        self.cfg, self.counter, self.calls = cfg, counter, 0

    def cost(self, usage: Usage) -> float:
        return reference_cost(self.cfg, usage)

    def complete(
        self,
        messages: list[dict],
        purpose: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        step: int = 0,
    ) -> Reply:
        prompt = self.counter.prompt(messages, tools)
        if prompt > self.cfg.context.input_tokens:
            raise ContextLimit("Fake request exceeded the cap")
        self.calls += 1
        if purpose == "summary":
            content = "\n".join(f"{i}. {h}\nUnknown." for i, h in enumerate(SUMMARY_HEADINGS, 1))
            message: dict[str, Any] = {"role": "assistant", "content": content}
        else:
            name = "submit" if step >= 8 else "bash"
            args = {} if name == "submit" else {"command": f"inspect {step}"}
            message = {
                "role": "assistant",
                "content": "Inspecting the fixture.",
                "tool_calls": [
                    {
                        "id": f"call_{step}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        usage = Usage(prompt, self.counter.text(json.dumps(message)))
        return Reply(message, usage, "tool_calls" if "tool_calls" in message else "stop", self.cost(usage), "FAKE")

    def close(self) -> None:
        pass
