# Owner: Student B
"""Qwen planner adapter. Offline contract tests do not establish model accuracy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from uuid import uuid4

from core.interfaces import Planner
from core.llm_client import APIError, LLMClient, SchemaError
from core.types import ExecutionContext
from planner.config import QwenPlannerConfig
from planner.contract import PlanContractError, WIRE_SCHEMA, compile_plan
from planner.prompts import PROMPT_VERSION, SYSTEM_PROMPT, json_value, planning_input


class PlannerError(RuntimeError):
    """Service or output failure. The orchestrator stops; this is not INFEASIBLE."""


class StudentBPlanner(Planner):
    IMPLEMENTED = True  # interface implemented; real Qwen accuracy not yet evaluated

    def __init__(self, config=None, *, client=None):
        self.config = config or (QwenPlannerConfig(client.config) if client else QwenPlannerConfig.from_env())
        if self.config.max_repairs not in (0, 1):
            raise ValueError("max_repairs must be 0 or 1")
        if client and client.config != self.config.llm:
            raise ValueError("Injected client and planner must use the same LLMConfig")
        self.client = client or LLMClient(self.config.llm)
        self.audit_root = Path(self.config.audit_dir)
        self.reset()

    def reset(self):
        self._instruction = None
        self._goal = None
        self._known = {}
        self._clarification = None
        self._episode_id = uuid4().hex
        self._call_index = 0
        self.last_diagnostics = None

    @property
    def goal(self):
        return copy.deepcopy(self._goal)

    @property
    def LABEL(self):
        return f"B: Qwen planner adapter (source={self.client.response_source}, model={self.client.config.model})"

    def plan(self, instruction, scene):
        self.reset()
        self._instruction = instruction
        return self._make(instruction, scene, [], ExecutionContext(scene), None)

    def replan(self, instruction, scene, history, context, clarification=None):
        if self._instruction is None:
            raise PlannerError("Call plan() before replan() so the original goal is available")
        if instruction != self._instruction:
            raise PlannerError("Instruction changed; call plan() to start a new task")
        # Use the explicit current A result rather than an older context.scene.
        context = ExecutionContext(scene, context.held_instance_id, context.last_release_instance_id)
        if clarification:
            self._clarification = clarification
        return self._make(instruction, scene, history, context, self._clarification)

    def _safe(self, value):
        text = json.dumps(json_value(value), ensure_ascii=False, allow_nan=False)
        key = self.client.config.api_key
        if key:
            text = text.replace(key, "[REDACTED]")
        return json.loads(text)

    def _save(self, audit):
        path = self.audit_root / self._episode_id / f"{self._call_index:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        audit["audit_path"] = str(path.resolve())
        self.last_diagnostics = self._safe(audit)
        path.write_text(json.dumps(self.last_diagnostics, indent=2, ensure_ascii=False) + "\n")

    def _make(self, instruction, scene, history, context, clarification):
        ids = [g.instance_id for g in scene.objects + scene.regions]
        if not instruction.strip() or len(ids) != len(set(ids)):
            raise PlannerError("Instruction must be nonempty and scene IDs must be unique")
        self._call_index += 1
        data = planning_input(instruction, scene, history, context, self._goal, clarification)
        audit = {"schema": "student-b-audit-v1", "prompt_version": PROMPT_VERSION,
                 "settings": self.config.public_settings(), "input": data,
                 "system_prompt": SYSTEM_PROMPT, "responses": [], "accepted": False,
                 "response_source": self.client.response_source, "error": None}
        last_error = None
        for attempt in range(self.config.max_repairs + 1):
            response = None
            try:
                response = self.client.call_llm(json.dumps(data, ensure_ascii=False, allow_nan=False),
                                                system=SYSTEM_PROMPT, json_schema=WIRE_SCHEMA)
                plan, goal = compile_plan(response.parsed, context, self._goal, self._known)
            except (SchemaError, PlanContractError) as exc:
                response = response or getattr(exc, "response", None)
                last_error = str(exc)
                audit["responses"].append({"response": json_value(response), "validation_error": last_error})
                data = {**data, "repair": {"validation_error": last_error,
                        "previous_output": response.text if response else "",
                        "instruction": "Return corrected JSON; keep the original task and goal."}}
                continue
            except APIError as exc:
                last_error = str(exc)
                break  # HTTP/transport errors are retried only by the bounded client
            else:
                plan.raw_llm_output = response.text
                self._goal = goal
                self._known.update({g.instance_id: copy.deepcopy(g) for g in scene.objects + scene.regions})
                audit["responses"].append({"response": json_value(response), "validation_error": None})
                audit.update(accepted=True, compiled_plan=json_value(plan), goal=goal,
                             repair_count=attempt, first_pass_valid=(attempt == 0))
                self._save(audit)
                return plan
        audit.update(error=last_error, repair_count=max(0, len(audit["responses"]) - 1))
        self._save(audit)
        raise PlannerError(self.last_diagnostics["error"] or "Planner produced no usable response")
