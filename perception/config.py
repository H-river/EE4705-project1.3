"""Separate Qwen-VL settings; never inherit B's text model or schema mode."""
from dataclasses import dataclass, fields
import math
import os
from urllib.parse import urlsplit

from core.llm_client import LLMConfig


@dataclass
class VisionConfig:
    llm: LLMConfig
    audit_dir: str = 'runs/qwen_a_audit'

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        base = env.get('EE4705_VLM_BASE_URL') or env.get('EE4705_QWEN_BASE_URL', '')
        base = base.strip().rstrip('/')
        u = urlsplit(base)
        if (not u.hostname or u.scheme not in ('https','http') or u.username or u.password
                or u.query or u.fragment or not u.path.endswith('/v1')
                or any(c in base for c in '{}<>') or any(c.isspace() for c in base)
                or (u.scheme=='http' and u.hostname not in ('127.0.0.1','localhost','::1'))):
            raise ValueError('Set a complete HTTPS EE4705_VLM_BASE_URL or EE4705_QWEN_BASE_URL ending in /v1')
        key = env.get('EE4705_VLM_API_KEY') or env.get('EE4705_QWEN_API_KEY') or env.get('DASHSCOPE_API_KEY')
        if not key:
            raise ValueError('Set DASHSCOPE_API_KEY or EE4705_VLM_API_KEY; use --offline-demo without a key')
        timeout = float(env.get('EE4705_VLM_TIMEOUT_S','60'))
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('EE4705_VLM_TIMEOUT_S must be finite and in (0,300]')
        model = env.get('EE4705_VLM_MODEL','qwen3-vl-plus').strip()
        if not model: raise ValueError('EE4705_VLM_MODEL cannot be empty')
        return cls(LLMConfig(model=model,base_url=base,api_key=key,timeout_s=timeout,
                             max_retries=1,max_tokens=2048,temperature=0,
                             structured_output_mode='json_object',enable_thinking=False,
                             cache_dir=env.get('EE4705_VLM_CACHE_DIR') or None),
                   env.get('EE4705_VLM_AUDIT_DIR','runs/qwen_a_audit'))

    def public_settings(self):
        return {f.name:getattr(self.llm,f.name) for f in fields(self.llm) if f.name!='api_key'}
