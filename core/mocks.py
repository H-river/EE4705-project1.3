# Owner: backbone (ALL)
"""Ground-truth mocks for the three student interfaces, plus the scripted
clarification provider.

These are PRIVILEGED backbone components (they may use EvalOracle /
SimWorld); the real Student modules must not.  They exist so each student
can develop against working stand-ins, and so backbone infrastructure can
be exercised end-to-end.  Results obtained with mocks are infrastructure
checks, never system performance.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from core.interfaces import ClarificationProvider, Executor, Perception, Planner, RobotEnvProtocol
from core.oracle import EvalOracle
from core.types import (
    Action,
    ErrorCode,
    ExecutionContext,
    ExecutionResult,
    GroundedObject,
    GroundStatus,
    Observation,
    Plan,
    PlanStatus,
    SceneDescription,
    Skill,
)
from core.vocab import Vocab
from core.world import SimWorld

_REGION_GT = "red_region"


class ScriptedClarifier(ClarificationProvider):
    """Scripted user: each response is consumed exactly once, in order."""

    def __init__(self, responses: Optional[list[str]] = None) -> None:
        self._responses = list(responses or [])
        self.exchanges: list[tuple[str, Optional[str]]] = []

    def ask(self, question: str) -> Optional[str]:
        response = self._responses.pop(0) if self._responses else None
        self.exchanges.append((question, response))
        return response

    def remaining(self) -> int:
        return len(self._responses)


class GTPerception(Perception):
    """Ground-truth perception mock.

    Uses the oracle's segmentation-based visibility (field of view AND
    occlusion of the current camera) but maintains a PERCEPTION-SIDE
    instance ID lifecycle: opaque ids ("p0", "p1", ...) assigned in order
    of first sighting, stable within an episode, cleared on reset().
    Ground-truth ID strings are never copied into perceived IDs.
    """

    def __init__(self, oracle: EvalOracle, vocab: Optional[Vocab] = None,
                 mode: str = "visible") -> None:
        if mode not in ("visible", "omniscient"):
            raise ValueError(f"unknown GTPerception mode {mode!r}")
        self._oracle = oracle
        self._vocab = vocab or Vocab()
        self._mode = mode
        self._ids: dict[str, str] = {}  # internal gt id -> perceived id
        self._counter = 0

    def reset(self) -> None:
        self._ids.clear()
        self._counter = 0

    def _pid(self, gt_id: str) -> str:
        if gt_id not in self._ids:
            self._ids[gt_id] = f"p{self._counter}"
            self._counter += 1
        return self._ids[gt_id]

    def _grounded(self, gt_id: str, bbox, frame_id: int) -> GroundedObject:
        entry = self._vocab.entry_for_gt(gt_id)
        cls = entry.cls if entry else gt_id
        kind = entry.kind if entry else "object"
        pos = self._oracle.object_pos(gt_id) if kind == "object" else None
        if kind == "region":
            b = self._oracle.region_bounds(gt_id)
            pos = np.array([b.center_xy[0], b.center_xy[1], b.support_z])
        return GroundedObject(
            instance_id=self._pid(gt_id),
            name=cls,
            status=GroundStatus.LOCALIZED,
            bbox_xyxy=tuple(float(v) for v in bbox),
            pos_world=tuple(float(v) for v in pos),
            confidence=1.0,
            source="gt",
            kind=kind,
            frame_id=frame_id,
            attributes=dict(entry.attributes) if entry else {},
        )

    def describe(self, obs: Observation, query: Optional[str] = None) -> SceneDescription:
        bboxes = self._gt_bboxes(obs)
        objects, regions = [], []
        for gt_id, bbox in bboxes.items():
            g = self._grounded(gt_id, bbox, obs.frame_id)
            (regions if g.kind == "region" else objects).append(g)
        ambiguities = []
        by_class: dict[str, list[GroundedObject]] = {}
        for g in objects:
            by_class.setdefault(g.name, []).append(g)
        for cls, group in by_class.items():
            if len(group) > 1:
                desc = ", ".join(f"{g.instance_id} ({g.attributes.get('color', '?')})" for g in group)
                ambiguities.append(f"multiple {cls} instances visible: {desc}")
        return SceneDescription(
            objects=objects, regions=regions,
            caption=f"{len(objects)} objects, {len(regions)} regions visible",
            ambiguities=ambiguities, frame_id=obs.frame_id, sim_time=obs.sim_time,
        )

    def _gt_bboxes(self, obs: Observation) -> dict:
        if self._mode == "visible":
            return self._oracle.gt_bboxes(camera=obs.camera_name)
        # omniscient mode: all active objects regardless of view (debug only)
        bboxes = dict(self._oracle.gt_bboxes(camera=obs.camera_name))
        for gt_id in self._oracle.active_objects():
            bboxes.setdefault(gt_id, (0.0, 0.0, 0.0, 0.0))
        bboxes.setdefault(_REGION_GT, (0.0, 0.0, 0.0, 0.0))
        return bboxes

    def ground(self, obs: Observation, target: str) -> Optional[GroundedObject]:
        scene = self.describe(obs)
        kind = "region" if self._vocab.find_phrase(target, "region") else "object"
        found = self._vocab.find_phrase(target, kind)
        if found is None:
            return None
        cls, qualifier = found
        qualifier = qualifier or Vocab.color_in_text(target)
        pool = scene.regions if kind == "region" else scene.objects
        candidates = [g for g in pool if g.name == cls]
        if qualifier:
            candidates = [g for g in candidates if g.attributes.get("color") == qualifier]
        if not candidates:
            return None
        if len(candidates) > 1:
            first = candidates[0]
            return GroundedObject(
                instance_id=first.instance_id, name=cls, status=GroundStatus.AMBIGUOUS,
                bbox_xyxy=first.bbox_xyxy, pos_world=None, confidence=0.4, source="gt",
                kind=first.kind, frame_id=obs.frame_id, attributes=dict(first.attributes),
            )
        return candidates[0]


class RulePlanner(Planner):
    """Keyword/synonym rule planner producing the fixed skeleton
    APPROACH -> GRASP -> MOVE_TO -> PLACE -> VERIFY -> STOP.

    Emits NEEDS_SEARCH when the target class is not visible,
    NEEDS_CLARIFICATION when several candidates match, INFEASIBLE when the
    instruction names no known object/region.  Positions of referenced
    instances are copied into action params so downstream execution mocks
    can act without perceived-ID -> ground-truth shortcuts.
    """

    def __init__(self, vocab: Optional[Vocab] = None) -> None:
        self._vocab = vocab or Vocab()
        self._pending_ambiguity: Optional[dict] = None

    def reset(self) -> None:
        self._pending_ambiguity = None

    # -- helpers

    def _resolve(self, instruction: str, scene: SceneDescription,
                 qualifier_override: Optional[str] = None):
        obj = self._vocab.find_phrase(instruction, "object")
        region = self._vocab.find_phrase(instruction, "region")
        if obj is None or region is None:
            return None, None, None, None
        cls, qualifier = obj
        qualifier = qualifier_override or qualifier
        region_cls = region[0]
        candidates = [g for g in scene.objects if g.name == cls]
        if qualifier:
            candidates = [g for g in candidates if g.attributes.get("color") == qualifier]
        region_inst = next((r for r in scene.regions if r.name == region_cls), None)
        return cls, candidates, region_cls, region_inst

    _CARRY_HEIGHT = 0.18

    @classmethod
    def _grasp_actions(cls, target: GroundedObject) -> list[Action]:
        tp = list(target.pos_world) if target.pos_world else None
        return [
            Action(Skill.APPROACH, target=target.instance_id, params={"pos": tp}),
            Action(Skill.GRASP, target=target.instance_id, params={"pos": tp}),
        ]

    @classmethod
    def _transport_actions(cls, held_id: str, region: GroundedObject,
                           verify_object: Optional[GroundedObject]) -> list[Action]:
        rp = list(region.pos_world)
        actions = [
            Action(Skill.MOVE_TO, target=region.instance_id,
                   params={"pos": [rp[0], rp[1], rp[2] + cls._CARRY_HEIGHT]}),
            Action(Skill.PLACE, target=region.instance_id,
                   params={"object": held_id, "pos": rp}),
        ]
        if verify_object is not None:
            actions.append(Action(Skill.VERIFY, target=verify_object.instance_id,
                                  params={"condition": "object_in_region",
                                          "object": verify_object.instance_id,
                                          "region": region.instance_id}))
        actions.append(Action(Skill.STOP))
        return actions

    # -- interface

    def plan(self, instruction: str, scene: SceneDescription) -> Plan:
        return self._plan_impl(instruction, scene, held_id=None, clarification=None)

    def replan(self, instruction: str, scene: SceneDescription, history, context: ExecutionContext,
               clarification: Optional[str] = None) -> Plan:
        return self._plan_impl(instruction, scene,
                               held_id=context.held_instance_id, clarification=clarification)

    def _plan_impl(self, instruction: str, scene: SceneDescription,
                   held_id: Optional[str], clarification: Optional[str]) -> Plan:
        qualifier = Vocab.color_in_text(clarification) if clarification else None
        cls, candidates, region_cls, region_inst = self._resolve(instruction, scene, qualifier)
        if cls is None:
            return Plan(status=PlanStatus.INFEASIBLE,
                        reason="instruction does not name a known object and destination region",
                        raw_llm_output=instruction)

        # Already holding the object: transport it.  If the destination is
        # not in the current view, search for it first.
        if held_id is not None:
            if region_inst is None:
                return Plan(status=PlanStatus.NEEDS_SEARCH,
                            actions=[Action(Skill.SEARCH, target=region_cls)],
                            reason=f"destination {region_cls} not currently visible",
                            raw_llm_output=instruction)
            held_inst = scene.find(held_id)  # may be None (occluded by arm)
            return Plan(status=PlanStatus.READY,
                        actions=self._transport_actions(held_id, region_inst, held_inst),
                        raw_llm_output=instruction)

        # Not holding yet: get the target first.
        if not candidates:
            return Plan(status=PlanStatus.NEEDS_SEARCH,
                        actions=[Action(Skill.SEARCH, target=cls)],
                        reason=f"target {cls} not currently visible",
                        raw_llm_output=instruction)
        if len(candidates) > 1:
            colors = sorted({g.attributes.get("color", "?") for g in candidates})
            return Plan(status=PlanStatus.NEEDS_CLARIFICATION,
                        clarification_question=(
                            f"I can see {len(candidates)} {cls} instances "
                            f"({', '.join(colors)}). Which one do you mean?"),
                        raw_llm_output=instruction)
        target = candidates[0]
        actions = self._grasp_actions(target)
        if region_inst is not None:
            actions += self._transport_actions(target.instance_id, region_inst, target)
        else:
            # Region not visible from here: grasp now, then replan (the
            # orchestrator replans after the plan ends without a PLACE).
            actions.append(Action(Skill.STOP))
        return Plan(status=PlanStatus.READY, actions=actions, raw_llm_output=instruction)


class TeleportExecutor(Executor):
    """Instant-motion execution mock with seeded fault injection.

    PRIVILEGED: manipulates the SimWorld directly (teleporting the robot
    joints), but keeps coherent held/released state through the real weld
    attachment machinery and produces observable scene changes.  It never
    unconditionally returns success.

    Fault injection (seeded, deterministic per trial after reset()):
    * fail_prob — a GRASP attempt misses (no attachment).
    * wrong_object_prob — a GRASP actually attaches a DIFFERENT nearby
      simulated object (the injected error is physical, so the evaluator
      must detect it through oracle state, not through any flag).
    """

    def __init__(self, world: SimWorld, seed: int = 0,
                 fail_prob: float = 0.0, wrong_object_prob: float = 0.0) -> None:
        self._world = world
        self._seed = seed
        self.fail_prob = fail_prob
        self.wrong_object_prob = wrong_object_prob
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    # -- helpers (privileged)

    def _teleport_joints(self, targets: dict[str, float]) -> None:
        """Set robot joints instantly, carrying any held (welded) object
        along so its EE-relative placement is preserved."""
        w = self._world
        held = w.attached_body_name()
        offset = w.body_pos(held) - w.ee_pos() if held is not None else None
        for joint, value in targets.items():
            w._set_joint(joint, value)
        self._sync_ctrl_and_forward()
        if held is not None and offset is not None:
            w.teleport_body(held, w.ee_pos() + offset)

    def _teleport_base_near(self, pos, standoff: float = 0.45) -> None:
        w = self._world
        p = np.asarray(pos, dtype=float)
        base = w.base_pose()
        delta = p[:2] - base[:2]
        dist = float(np.linalg.norm(delta))
        yaw = float(np.arctan2(delta[1], delta[0])) if dist > 1e-6 else float(base[2])
        target_xy = p[:2] - standoff * delta / dist if dist > standoff else base[:2]
        self._teleport_joints({"base_x": float(target_xy[0]), "base_y": float(target_xy[1]),
                               "base_yaw": yaw})

    def _teleport_ee_to(self, pos) -> bool:
        w = self._world
        p = np.asarray(pos, dtype=float)
        base = w.base_pose()
        yaw = base[2]
        rot = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        local_xy = rot @ (p[:2] - base[:2])
        targets = {"arm_x": float(local_xy[0]), "arm_y": float(local_xy[1]), "arm_z": float(p[2] - 0.80)}
        ranges = w.joint_ranges()
        for joint, value in targets.items():
            lo, hi = ranges[joint]
            if not lo <= value <= hi:
                return False
        self._teleport_joints(targets)
        return True

    def _sync_ctrl_and_forward(self) -> None:
        w = self._world
        w.data.ctrl[:] = [w._get_joint(j) for j in ("base_x", "base_y", "base_yaw", "arm_x", "arm_y", "arm_z")]
        from core.rendering import mujoco

        mujoco.mj_forward(w.model, w.data)

    def _result(self, action: Action, env: RobotEnvProtocol, success: bool,
                error: ErrorCode = ErrorCode.NONE, **info) -> ExecutionResult:
        obs = env.get_obs()
        return ExecutionResult(action=action, success=success, error_code=error,
                               post_frame_id=obs.frame_id, info=info)

    # -- interface

    def execute(self, action: Action, env: RobotEnvProtocol, perception: Perception) -> ExecutionResult:
        skill = action.skill
        params = action.params or {}
        pos = params.get("pos")

        if skill is Skill.APPROACH:
            if pos is None:
                return self._result(action, env, False, ErrorCode.INVALID_ACTION)
            self._teleport_base_near(pos)
            return self._result(action, env, True)

        if skill is Skill.REACH:
            if pos is None:
                return self._result(action, env, False, ErrorCode.INVALID_ACTION)
            ok = self._teleport_ee_to(pos)
            return self._result(action, env, ok, ErrorCode.NONE if ok else ErrorCode.UNREACHABLE)

        if skill is Skill.GRASP:
            if env.is_attached():
                return self._result(action, env, False, ErrorCode.ALREADY_HOLDING)
            if pos is None:
                return self._result(action, env, False, ErrorCode.INVALID_ACTION)
            if self._rng.random() < self.fail_prob:
                return self._result(action, env, False, ErrorCode.GRASP_MISSED, injected=True)
            grasp_pos = np.asarray(pos, dtype=float)
            if self._rng.random() < self.wrong_object_prob:
                wrong = self._pick_wrong_object(grasp_pos)
                if wrong is not None:
                    grasp_pos = wrong  # physically go to the WRONG object
            if not self._teleport_ee_to(grasp_pos):
                return self._result(action, env, False, ErrorCode.UNREACHABLE)
            handle = self._world.try_attach_near_ee()
            if handle is None:
                return self._result(action, env, False, ErrorCode.GRASP_MISSED)
            return self._result(action, env, True, attachment=handle)

        if skill is Skill.MOVE_TO:
            if pos is None:
                return self._result(action, env, False, ErrorCode.INVALID_ACTION)
            # Turn toward the destination first (so subsequent VERIFY /
            # verification observations actually have it in view).
            base = self._world.base_pose()
            delta = np.asarray(pos, dtype=float)[:2] - base[:2]
            if np.linalg.norm(delta) > 1e-6:
                self._teleport_joints({"base_yaw": float(np.arctan2(delta[1], delta[0]))})
            if not self._teleport_ee_to(pos):
                self._teleport_base_near(pos)
                if not self._teleport_ee_to(pos):
                    return self._result(action, env, False, ErrorCode.UNREACHABLE)
            return self._result(action, env, True)

        if skill is Skill.PLACE:
            if not env.is_attached():
                return self._result(action, env, False, ErrorCode.NOT_HOLDING)
            held = self._world.attached_body_name()
            self._world.detach()
            if pos is not None and held is not None:
                drop = np.asarray(pos, dtype=float) + [0.0, 0.0, 0.03]
                self._world.teleport_body(held, drop)
            self._world.step(150)  # settle: observable, physical
            return self._result(action, env, True)

        if skill is Skill.SEARCH:
            return self._search(action, env, perception)

        if skill is Skill.VERIFY:
            return self._verify(action, env, perception)

        if skill is Skill.STOP:
            return self._result(action, env, True)

        return self._result(action, env, False, ErrorCode.INVALID_ACTION)

    def _pick_wrong_object(self, intended_pos: np.ndarray):
        """Deterministically pick a different active object's position."""
        others = []
        for name in self._world.active_objects():
            p = self._world.body_pos(name)
            if np.linalg.norm(p - intended_pos) > 0.10:
                others.append((name, p))
        if not others:
            return None
        others.sort(key=lambda item: item[0])
        return others[0][1]

    def _search(self, action: Action, env: RobotEnvProtocol, perception: Perception) -> ExecutionResult:
        """Rotate the base (teleport increments), re-observing through the
        NORMAL camera path until the target becomes visible.  Perception
        stays view-limited: no switch to full-scene perception."""
        w = self._world
        base = w.base_pose()
        for k in range(13):
            obs = env.get_obs()
            try:
                found = perception.ground(obs, action.target or "")
            except Exception as exc:
                return self._result(action, env, False, ErrorCode.SEARCH_FATAL,
                                    detail=f"{type(exc).__name__}: {exc}")
            if found is not None:
                return self._result(action, env, True, views=k + 1,
                                    grounded_instance=found.instance_id)
            yaw = float(base[2]) + 0.5 * (k + 1)
            yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
            self._teleport_joints({"base_yaw": yaw})
        return self._result(action, env, False, ErrorCode.SEARCH_NOT_FOUND, views=13)

    def _verify(self, action: Action, env: RobotEnvProtocol, perception: Perception) -> ExecutionResult:
        condition = (action.params or {}).get("condition")
        obs = env.get_obs()
        if condition == "holding":
            ok = env.is_attached()
            return self._result(action, env, ok, ErrorCode.NONE if ok else ErrorCode.VERIFY_FAILED)
        if condition == "object_visible":
            scene = perception.describe(obs)
            ok = scene.find(action.target or "") is not None
            return self._result(action, env, ok, ErrorCode.NONE if ok else ErrorCode.VERIFY_FAILED)
        if condition == "object_in_region":
            scene = perception.describe(obs)
            obj = scene.find((action.params or {}).get("object", ""))
            region = scene.find((action.params or {}).get("region", ""))
            if obj is None or region is None or obj.pos_world is None or region.pos_world is None:
                return self._result(action, env, False, ErrorCode.VERIFY_FAILED, detail="not grounded")
            dx = abs(obj.pos_world[0] - region.pos_world[0])
            dy = abs(obj.pos_world[1] - region.pos_world[1])
            ok = dx <= 0.11 and dy <= 0.11
            return self._result(action, env, ok, ErrorCode.NONE if ok else ErrorCode.VERIFY_FAILED,
                                dx=dx, dy=dy)
        return self._result(action, env, False, ErrorCode.INVALID_ACTION)
