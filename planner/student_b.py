# Owner: Student B
"""Qwen planner adapter. Offline contract tests do not establish model accuracy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from uuid import uuid4

from core.interfaces import Planner
from core.llm_client import APIError, ContentFiltered, LLMClient, SchemaError
from core.types import ExecutionContext, GroundedObject, GroundStatus, Plan, PlanStatus, SceneDescription
from planner.config import QwenPlannerConfig
from planner.contract import COMPILER_VERSION, PlanContractError, WIRE_SCHEMA, compile_plan
from planner.memory import EpisodeMemory
from planner.prompts import PROMPT_VERSION, SYSTEM_PROMPT, json_value, planning_input


class PlannerError(RuntimeError):
    """Service (HTTP/transport) or usage failure. The orchestrator stops; this is
    not INFEASIBLE. Output that fails the contract after repair is returned as a
    REJECTED plan instead (contract v4)."""


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
        # Optional proprioception hook (e.g. env.get_base_pose) for memory
        # expiry after base motion; None disables only that check.
        self.base_pose_source = None
        self.memory = EpisodeMemory()
        self.memory_max_age_frames = 40
        self.memory_max_base_move_m = 0.5
        self.reset()

    def reset(self):
        self._instruction = None
        self._goal = None
        self._known = {}
        self._clarification = None
        self._episode_id = uuid4().hex
        self._call_index = 0
        self.last_diagnostics = None
        self.rationales = []  # E3: one entry per planning call, for the trial record
        if hasattr(self, "memory"):
            self.memory.reset()

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

    def _rationale(self, audit, response, status):
        parsed = getattr(response, "parsed", None) if response is not None else None
        text = parsed.get("rationale", "") if isinstance(parsed, dict) else ""
        audit["rationale"] = text or None
        self.rationales.append({"call": self._call_index, "status": status, "rationale": text or None,
                                "used_memory_for": audit.get("used_memory_for")})

    def _save(self, audit):
        path = self.audit_root / self._episode_id / f"{self._call_index:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        audit["audit_path"] = str(path.resolve())
        self.last_diagnostics = self._safe(audit)
        path.write_text(json.dumps(self.last_diagnostics, indent=2, ensure_ascii=False) + "\n")

    def _base_pose(self):
        try:
            return tuple(float(v) for v in self.base_pose_source()) if self.base_pose_source else None
        except Exception:
            return None

    def _with_memory(self, scene, context):
        """Scene for planning: A's scene, plus the bound goal region at its
        remembered position when it is out of view now.  The object is never
        recalled (GRASP needs fresh 3D evidence), so a missing object is still
        searched for.  Returns (scene, memory audit or None)."""
        if not self.config.use_memory:
            return scene, None
        pose = self._base_pose()
        self.memory.update(scene, pose, context.held_instance_id, context.last_release_instance_id)
        goal = self._goal or {}
        region = goal.get("region_id")
        if not region:
            return scene, None
        current = scene.find(region)
        if current is not None and current.status is GroundStatus.LOCALIZED:
            return scene, None
        if current is not None and current.status is GroundStatus.AMBIGUOUS:
            return scene, None
        pos = self.memory.recall(region, self.memory_max_age_frames, self.memory_max_base_move_m,
                                 frame_id=scene.frame_id, base_pose=pose)
        entry = self.memory.entry(region)
        if pos is None or entry.kind != "region" or entry.name != goal.get("region_name"):
            return scene, None
        recalled = GroundedObject(region, entry.name, GroundStatus.LOCALIZED,
                                  bbox_xyxy=current.bbox_xyxy if current else None, pos_world=pos,
                                  confidence=current.confidence if current else 0.0, kind="region",
                                  frame_id=entry.frame_id, attributes=dict(entry.attributes),
                                  region_half_extents_xy=entry.region_half_extents_xy)
        regions = [g for g in scene.regions if g.instance_id != region] + [recalled]
        planned = SceneDescription(list(scene.objects), regions, scene.caption,
                                   list(scene.ambiguities), scene.frame_id, scene.sim_time)
        age = self.memory.age(region, scene.frame_id)
        return planned, {"used_memory_for": region, "memory_age_frames": age,
                         "memory_pos_world": list(pos), "memory_frame_id": entry.frame_id}

    def _make(self, instruction, scene, history, context, clarification):
        ids = [g.instance_id for g in scene.objects + scene.regions]
        if not instruction.strip() or len(ids) != len(set(ids)):
            raise PlannerError("Instruction must be nonempty and scene IDs must be unique")
        self._call_index += 1
        scene, memory_audit = self._with_memory(scene, context)
        context = ExecutionContext(scene, context.held_instance_id, context.last_release_instance_id)
        data = planning_input(instruction, scene, history, context, self._goal, clarification)
        audit = {"schema": "student-b-audit-v1", "prompt_version": PROMPT_VERSION,
                 "compiler_version": COMPILER_VERSION,
                 "settings": self.config.public_settings(), "input": data,
                 "system_prompt": SYSTEM_PROMPT, "responses": [], "accepted": False,
                 "response_source": self.client.response_source, "error": None,
                 "used_memory_for": None, "memory_age_frames": None}
        if memory_audit:
            audit.update(memory_audit)
        last_error = None
        service_error = False
        for attempt in range(self.config.max_repairs + 1):
            response = None
            normalizations = []
            try:
                response = self.client.call_llm(json.dumps(data, ensure_ascii=False, allow_nan=False),
                                                system=SYSTEM_PROMPT, json_schema=WIRE_SCHEMA)
                if isinstance(response, ContentFiltered):
                    raise APIError(response.detail)  # same prompt would be refused again
                plan, goal = compile_plan(response.parsed, context, self._goal, self._known,
                                          normalizations=normalizations)
            except (SchemaError, PlanContractError) as exc:
                response = response or getattr(exc, "response", None)
                last_error = str(exc)
                audit["responses"].append({"response": json_value(response), "validation_error": last_error,
                                           "normalizations": normalizations})
                data = {**data, "repair": {"validation_error": last_error,
                        "previous_output": response.text if response else "",
                        "instruction": "Return corrected JSON; keep the original task and goal."}}
                continue
            except APIError as exc:
                last_error = str(exc)
                service_error = True
                break  # HTTP/transport errors are retried only by the bounded client
            else:
                plan.raw_llm_output = response.text
                self._goal = goal
                self._known.update({g.instance_id: copy.deepcopy(g) for g in scene.objects + scene.regions})
                audit["responses"].append({"response": json_value(response), "validation_error": None,
                                           "normalizations": normalizations})
                audit.update(accepted=True, compiled_plan=json_value(plan), goal=goal,
                             repair_count=attempt, first_pass_valid=(attempt == 0),
                             normalization_count=len(normalizations))
                self._rationale(audit, response, plan.status.value)
                self._save(audit)
                return plan
        audit.update(error=last_error, repair_count=max(0, len(audit["responses"]) - 1))
        if not service_error:
            # The model answered, but no answer passed the contract: a
            # REJECTED plan lets the orchestrator replan instead of ERROR.
            plan = Plan(status=PlanStatus.REJECTED, reason=f"Planner output rejected by the contract: {last_error}")
            audit["compiled_plan"] = json_value(plan)
            self._rationale(audit, response, plan.status.value)
            self._save(audit)
            return plan
        self._save(audit)
        raise PlannerError(self.last_diagnostics["error"] or "Planner produced no usable response")
