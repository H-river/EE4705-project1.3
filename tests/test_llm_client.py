# Owner: backbone (ALL)
"""LLM client tests: fake transport only — no credentials, no network."""

from __future__ import annotations

import pytest

from core.llm_client import (
    APIError,
    CacheMissError,
    FakeTransport,
    LLMClient,
    LLMConfig,
    SchemaError,
    TransportError,
    validate_schema,
)


def cfg(**kw) -> LLMConfig:
    base = dict(model="test-model", base_url="https://fake.example/v1", api_key="sk-test",
                retry_backoff_s=0.0)
    base.update(kw)
    return LLMConfig(**base)


def test_basic_call_and_accounting():
    t = FakeTransport([FakeTransport.completion("hello", prompt_tokens=12, completion_tokens=7)])
    c = LLMClient(cfg(), transport=t)
    r = c.call_llm("hi")
    assert r.text == "hello" and r.cached is False and r.attempts == 1
    assert r.raw is not None and r.raw["choices"][0]["message"]["content"] == "hello"
    assert c.stats.requests == 1 and c.stats.live_requests == 1 and c.stats.cache_hits == 0
    assert c.stats.prompt_tokens == 12 and c.stats.completion_tokens == 7
    assert t.requests[0]["payload"]["model"] == "test-model"


def test_missing_api_key_raises_on_live_request():
    c = LLMClient(cfg(api_key=None), transport=FakeTransport([FakeTransport.completion("x")]))
    with pytest.raises(APIError, match="API key"):
        c.call_llm("hi")


def test_bounded_retries_on_retryable_failures():
    t = FakeTransport([
        (503, {"error": {"message": "overloaded"}}),
        TransportError("connection reset"),
        FakeTransport.completion("ok"),
    ])
    c = LLMClient(cfg(max_retries=2), transport=t)
    r = c.call_llm("hi")
    assert r.text == "ok" and r.attempts == 3
    assert c.stats.attempts == 3 and c.stats.live_requests == 1


def test_retries_are_bounded():
    t = FakeTransport([TransportError("down")] * 10)
    c = LLMClient(cfg(max_retries=2), transport=t)
    with pytest.raises(APIError, match="after 3 attempt"):
        c.call_llm("hi")
    assert len(t.requests) == 3  # 1 + 2 retries, never more


def test_non_retryable_status_fails_immediately():
    t = FakeTransport([(401, {"error": {"message": "bad key"}}), FakeTransport.completion("x")])
    c = LLMClient(cfg(max_retries=3), transport=t)
    with pytest.raises(APIError, match="401"):
        c.call_llm("hi")
    assert len(t.requests) == 1


def test_structured_output_parsing_and_validation():
    schema = {"type": "object", "required": ["target"],
              "properties": {"target": {"type": "string"},
                             "count": {"type": "integer"}}}
    t = FakeTransport([FakeTransport.completion('```json\n{"target": "stone", "count": 2}\n```')])
    c = LLMClient(cfg(), transport=t)
    r = c.call_llm("hi", json_schema=schema)
    assert r.parsed == {"target": "stone", "count": 2}

    t2 = FakeTransport([FakeTransport.completion('{"count": 2}')])
    c2 = LLMClient(cfg(), transport=t2)
    with pytest.raises(SchemaError, match="target"):
        c2.call_llm("hi", json_schema=schema)


def test_schema_validator_subset():
    validate_schema({"a": [1, 2]}, {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}}})
    with pytest.raises(SchemaError):
        validate_schema({"a": "x"}, {"type": "object", "properties": {"a": {"type": "number"}}})
    with pytest.raises(SchemaError):
        validate_schema("SEARCH2", {"type": "string", "enum": ["SEARCH", "GRASP"]})
    with pytest.raises(SchemaError):
        validate_schema(True, {"type": "integer"})  # bool is not an integer here


def test_disk_cache_hit_and_key_sensitivity(tmp_path):
    t = FakeTransport([FakeTransport.completion("first"), FakeTransport.completion("second")])
    c = LLMClient(cfg(cache_dir=str(tmp_path)), transport=t)
    r1 = c.call_llm("same prompt")
    r2 = c.call_llm("same prompt")  # cache hit: transport NOT called again
    assert r1.text == r2.text == "first"
    assert r1.cached is False and r2.cached is True  # explicit reporting
    assert c.stats.cache_hits == 1 and c.stats.live_requests == 1
    r3 = c.call_llm("different prompt")  # different key -> live request
    assert r3.text == "second" and r3.cached is False

    # model change changes the key
    c2 = LLMClient(cfg(model="other-model", cache_dir=str(tmp_path)),
                   transport=FakeTransport([FakeTransport.completion("third")]))
    assert c2.call_llm("same prompt").text == "third"

    # image content is part of the key
    tv = FakeTransport([FakeTransport.completion("img1"), FakeTransport.completion("img2")])
    cv = LLMClient(cfg(cache_dir=str(tmp_path)), transport=tv)
    assert cv.call_vlm(b"\x89PNG-A", "describe").text == "img1"
    assert cv.call_vlm(b"\x89PNG-A", "describe").cached is True
    assert cv.call_vlm(b"\x89PNG-B", "describe").text == "img2"


def test_cache_only_mode(tmp_path):
    # populate the cache with credentials...
    c = LLMClient(cfg(cache_dir=str(tmp_path)), transport=FakeTransport([FakeTransport.completion("warm")]))
    assert c.call_llm("q").text == "warm"
    # ...then cache-only WITHOUT credentials returns the entry
    c2 = LLMClient(cfg(api_key=None, cache_only=True, cache_dir=str(tmp_path)),
                   transport=FakeTransport([]))
    r = c2.call_llm("q")
    assert r.text == "warm" and r.cached is True
    # a cache miss fails clearly (not with a confusing credentials error)
    with pytest.raises(CacheMissError):
        c2.call_llm("never asked before")


def test_credentials_never_written_to_cache(tmp_path):
    c = LLMClient(cfg(api_key="sk-SECRET", cache_dir=str(tmp_path)),
                  transport=FakeTransport([FakeTransport.completion("x")]))
    c.call_llm("q")
    for f in tmp_path.glob("*.json"):
        assert "sk-SECRET" not in f.read_text()
