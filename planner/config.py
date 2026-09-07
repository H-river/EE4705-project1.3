"""Explicit Qwen settings. Loading configuration never sends a request."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from core.llm_client import LLMConfig

DEFAULT_MODEL = "qwen3.7-plus-2026-05-26"


class PlannerConfigError(ValueError):
    pass


def _boolean(env, name, default=None):
    value = env.get(name, "").strip().lower()
    if not value:
        return default
    if value not in ("true", "false"):
        raise PlannerConfigError(f"{name} must be true or false")
    return value == "true"


@dataclass
class QwenPlannerConfig:
    llm: LLMConfig
    audit_dir: str = "runs/qwen_b_audit"
    max_repairs: int = 1

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        base = env.get("EE4705_QWEN_BASE_URL", "").strip().rstrip("/")
        if not base:
            raise PlannerConfigError("Set EE4705_QWEN_BASE_URL from your Qwen console. "
                                     "Without an API, run: python -m planner.run --offline-demo")
        parts = urlsplit(base)
        if (parts.scheme not in ("https", "http") or not parts.hostname or parts.username
                or parts.password or parts.query or parts.fragment or "{" in base or "}" in base
                or "<" in base or ">" in base or any(c.isspace() for c in base)
                or not parts.path.endswith("/v1")):
            raise PlannerConfigError("EE4705_QWEN_BASE_URL must be a complete /v1 base URL, "
                                     "without placeholders, credentials, query or fragment")
        if parts.scheme == "http" and parts.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise PlannerConfigError("Use HTTPS for a remote endpoint; HTTP is allowed only on loopback")
        cache_only = _boolean(env, "EE4705_QWEN_CACHE_ONLY", False)
        key = env.get("EE4705_QWEN_API_KEY") or env.get("DASHSCOPE_API_KEY")
        if not cache_only and not key:
            raise PlannerConfigError("Set DASHSCOPE_API_KEY (or EE4705_QWEN_API_KEY). "
                                     "No key is needed for --offline-demo")
        try:
            timeout = float(env.get("EE4705_QWEN_TIMEOUT_S", "60"))
            retries = int(env.get("EE4705_QWEN_MAX_RETRIES", "1"))
            tokens = int(env.get("EE4705_QWEN_MAX_TOKENS", "2048"))
            if not math.isfinite(timeout) or not 0 < timeout <= 300 or not 0 <= retries <= 3:
                raise ValueError("timeout must be in (0, 300], retries in [0, 3]")
            model = env.get("EE4705_QWEN_MODEL", DEFAULT_MODEL).strip()
            if not model:
                raise ValueError("model cannot be empty")
            llm = LLMConfig(model=model, base_url=base, api_key=key,
                            timeout_s=timeout, max_retries=retries, max_tokens=tokens,
                            temperature=0, cache_only=cache_only,
                            cache_dir=env.get("EE4705_QWEN_CACHE_DIR", "runs/qwen_cache") or None,
                            structured_output_mode=env.get("EE4705_QWEN_OUTPUT_MODE", "json_schema"),
                            enable_thinking=_boolean(env, "EE4705_QWEN_THINKING"))
        except ValueError as exc:
            raise PlannerConfigError(f"Invalid Qwen configuration: {exc}") from exc
        return cls(llm, audit_dir=env.get("EE4705_QWEN_AUDIT_DIR", "runs/qwen_b_audit"))

    def public_settings(self):
        """Allowlist: never serialize credentials or the full config object."""
        return {"model": self.llm.model, "base_url": self.llm.base_url,
                "output_mode": self.llm.structured_output_mode,
                "cache_only": self.llm.cache_only, "cache_dir": self.llm.cache_dir,
                "timeout_s": self.llm.timeout_s, "max_retries": self.llm.max_retries,
                "max_tokens": self.llm.max_tokens, "enable_thinking": self.llm.enable_thinking,
                "audit_dir": self.audit_dir, "max_repairs": self.max_repairs,
                "api_key_configured": bool(self.llm.api_key)}
