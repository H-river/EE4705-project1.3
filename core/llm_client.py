# Owner: backbone (ALL)
"""Configurable OpenAI-compatible chat-completions client with disk cache.

Used by Student A (VLM) and Student B (LLM).  Features:
* explicit model / base URL / API key configuration (never from hard-coded
  defaults pointing at a paid service),
* request timeout and bounded retries for retryable failures (HTTP 408/429/
  5xx and transport errors),
* structured-output parsing/validation against a minimal JSON schema subset,
* per-client accounting: requests, attempts, cache hits, tokens, latency,
* original (raw) response retention alongside parsed output,
* content-addressed disk cache whose key covers provider/base URL, model,
  prompts, image content, schema, generation settings and the cache format
  version,
* cache-only mode that works without credentials (a live request without an
  API key raises APIError; a cache-only miss raises CacheMissError).

Credentials are never logged and never included in cache keys or entries.

Automated tests use FakeTransport: no credentials, no network.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

CACHE_FORMAT_VERSION = 1


class APIError(RuntimeError):
    """Configuration or provider failure (including missing credentials)."""


class CacheMissError(APIError):
    """Cache-only mode was requested but no cached entry exists."""


class SchemaError(APIError):
    """The provider response failed structured-output validation."""


@dataclass
class LLMConfig:
    model: str
    base_url: str  # e.g. "https://api.openai.com/v1"
    api_key: Optional[str] = None
    timeout_s: float = 60.0
    max_retries: int = 2  # retries AFTER the first attempt
    retry_backoff_s: float = 1.0
    temperature: float = 0.0
    max_tokens: int = 1024
    cache_dir: Optional[str] = None
    cache_only: bool = False


@dataclass
class LLMResponse:
    text: str
    parsed: Optional[dict] = None
    raw: Optional[dict] = None  # original provider response, retained
    cached: bool = False  # explicit cache-hit vs live-request reporting
    attempts: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMStats:
    requests: int = 0
    live_requests: int = 0
    cache_hits: int = 0
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Transport(Protocol):
    """POSTs a JSON payload to ``url`` and returns (status_code, json_body).
    Implementations must honor ``timeout_s`` and raise TransportError on
    connection-level failures."""

    def post(self, url: str, headers: dict[str, str], payload: dict, timeout_s: float) -> tuple[int, dict]: ...


class TransportError(RuntimeError):
    pass


class RequestsTransport:
    """Live HTTP transport (the only component that touches the network)."""

    def post(self, url: str, headers: dict[str, str], payload: dict, timeout_s: float) -> tuple[int, dict]:
        import requests

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        except requests.RequestException as exc:
            raise TransportError(str(exc)) from exc
        try:
            body = resp.json()
        except ValueError:
            body = {"error": {"message": f"non-JSON response (status {resp.status_code})"}}
        return resp.status_code, body


class FakeTransport:
    """Test transport: replays canned responses; records requests.  Never
    requires credentials or network."""

    def __init__(self, responses: Optional[list] = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[dict] = []

    @staticmethod
    def completion(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> tuple[int, dict]:
        return 200, {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }

    def post(self, url: str, headers: dict[str, str], payload: dict, timeout_s: float) -> tuple[int, dict]:
        self.requests.append({"url": url, "payload": payload})
        if not self.responses:
            raise TransportError("FakeTransport has no more responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------------- schema

def validate_schema(value: Any, schema: dict, path: str = "$") -> None:
    """Minimal JSON-schema subset: type, properties, required, items, enum."""
    t = schema.get("type")
    type_map = {"object": dict, "array": list, "string": str, "number": (int, float),
                "integer": int, "boolean": bool}
    if t is not None:
        expected = type_map.get(t)
        if expected is None:
            raise SchemaError(f"{path}: unsupported schema type {t!r}")
        if t == "number" and isinstance(value, bool) or not isinstance(value, expected) or (
            t == "integer" and isinstance(value, bool)
        ):
            raise SchemaError(f"{path}: expected {t}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: {value!r} not in enum {schema['enum']}")
    if t == "object":
        for key in schema.get("required", []):
            if key not in value:
                raise SchemaError(f"{path}: missing required key {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                validate_schema(value[key], sub, f"{path}.{key}")
    if t == "array" and "items" in schema:
        for i, item in enumerate(value):
            validate_schema(item, schema["items"], f"{path}[{i}]")


def _extract_json(text: str) -> dict:
    """Parse a JSON object from model text, tolerating markdown fences."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    try:
        out = json.loads(s)
    except json.JSONDecodeError as exc:
        # last resort: first {...} span
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            try:
                out = json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                raise SchemaError(f"response is not valid JSON: {exc}") from exc
        else:
            raise SchemaError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(out, dict):
        raise SchemaError("structured response must be a JSON object")
    return out


# ---------------------------------------------------------------- client

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LLMClient:
    def __init__(self, config: LLMConfig, transport: Optional[Transport] = None) -> None:
        self.config = config
        self.transport = transport if transport is not None else RequestsTransport()
        self.stats = LLMStats()
        self._cache_dir = pathlib.Path(config.cache_dir) if config.cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- public API

    def call_llm(self, prompt: str, system: Optional[str] = None,
                 json_schema: Optional[dict] = None) -> LLMResponse:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._call(messages, json_schema)

    def call_vlm(self, image_png_bytes: bytes, prompt: str, system: Optional[str] = None,
                 json_schema: Optional[dict] = None) -> LLMResponse:
        b64 = base64.b64encode(image_png_bytes).decode("ascii")
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        })
        return self._call(messages, json_schema)

    # ---------------- internals

    def _cache_key(self, messages: list, json_schema: Optional[dict]) -> str:
        # Key covers everything that affects the response.  NEVER the API key.
        material = json.dumps(
            {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "base_url": self.config.base_url,
                "model": self.config.model,
                "messages": messages,  # includes image data URLs (content-addressed)
                "schema": json_schema,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Optional[pathlib.Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{key}.json"

    def _call(self, messages: list, json_schema: Optional[dict]) -> LLMResponse:
        self.stats.requests += 1
        key = self._cache_key(messages, json_schema)
        path = self._cache_path(key)

        if path is not None and path.exists():
            entry = json.loads(path.read_text())
            resp = LLMResponse(text=entry["text"], parsed=entry.get("parsed"),
                               raw=entry.get("raw"), cached=True)
            self.stats.cache_hits += 1
            return resp

        if self.config.cache_only:
            raise CacheMissError(
                f"cache-only mode: no cached entry for this request (key {key[:12]}...)")

        if not self.config.api_key:
            raise APIError("live request requires an API key (none configured)")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}",
                   "Content-Type": "application/json"}

        attempts = 0
        start = time.monotonic()
        last_error = "no attempt made"
        body: Optional[dict] = None
        while attempts <= self.config.max_retries:
            attempts += 1
            self.stats.attempts += 1
            try:
                status, body = self.transport.post(url, headers, payload, self.config.timeout_s)
            except TransportError as exc:
                last_error = f"transport error: {exc}"
                body = None
            else:
                if status == 200:
                    break
                last_error = f"HTTP {status}: {body.get('error', {}).get('message', '')!s}"
                if status not in _RETRYABLE_STATUS:
                    raise APIError(last_error)
                body = None
            if attempts <= self.config.max_retries:
                time.sleep(self.config.retry_backoff_s * attempts)
        if body is None:
            raise APIError(f"request failed after {attempts} attempt(s): {last_error}")

        latency = time.monotonic() - start
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError(f"malformed provider response: {exc}") from exc
        usage = body.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        parsed: Optional[dict] = None
        if json_schema is not None:
            parsed = _extract_json(text)
            validate_schema(parsed, json_schema)

        resp = LLMResponse(text=text, parsed=parsed, raw=body, cached=False,
                           attempts=attempts, latency_s=latency,
                           prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self.stats.live_requests += 1
        self.stats.prompt_tokens += prompt_tokens
        self.stats.completion_tokens += completion_tokens
        self.stats.total_latency_s += latency

        if path is not None:
            entry = {"cache_format_version": CACHE_FORMAT_VERSION, "text": text,
                     "parsed": parsed, "raw": body}
            path.write_text(json.dumps(entry))
        return resp
