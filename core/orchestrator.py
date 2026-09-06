# Owner: backbone (ALL)
"""Closed-loop orchestration: perceive -> plan -> validate -> execute ->
verify, with bounded search, clarification, replanning and retry.

Success policy: the orchestrator returns CLAIMED_SUCCESS only when the
task-level final verification passes.  Completing an action list is never
sufficient.  CLAIMED_SUCCESS remains the system's own (vision-based) claim;
ground-truth "actual success" is judged separately by eval.criteria.

Final-verification reuse rule: a successful in-plan VERIFY(object_in_region)
result may be reused as the final verification ONLY when no state-changing
action (GRASP/PLACE/MOVE_TO/APPROACH/REACH/SEARCH) executed after it;
otherwise a fresh observation is taken and re-verified.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from typing import Optional

from core.env import RobotEnv
from core.interfaces import ClarificationProvider, Executor, Perception, Planner
from core.obs_store import ObservationStore
from core.types import (
    Action,
    ClarificationExchange,
    ErrorCode,
    ExecutionContext,
    ExecutionResult,
    GroundedObject,
    GroundStatus,
    Plan,
    PlanStatus,
    SceneDescription,
    Skill,
    TrialOutcome,
    VerificationResult,
)
from core.validation import validate_plan

_STATE_CHANGING = {Skill.SEARCH, Skill.APPROACH, Skill.REACH, Skill.GRASP, Skill.MOVE_TO, Skill.PLACE}

# Camera routing (docs/DECISIONS.md §11): the orchestrator's own captures —
# describe/ground at plan time, SEARCH views (via the executor) and the
# FINAL VERIFICATION observation — use the HEAD camera (env.get_obs() default,
# alias "onboard").  The wrist cameras (left_wrist / right_wrist) are exposed
# through RobotEnv.get_obs / get_obs_multi for a future Executor's alignment
# and grasp/place checks; nothing here requests them, and GTPerception plus
# the grounding evaluation stay head-only.

# Region half-extent assumed by SYSTEM-side (vision) verification when 3D
# positions are available; matches assets/objects.yaml size_xyz/2 for
# red_region.  This is a perception-side constant, not oracle access.
_REGION_HALF_XY = 0.08
_VERIFY_XY_SLACK = 0.03


@dataclass
class OrchestratorConfig:
    max_search_attempts: int = 2
    max_clarifications: int = 2
    max_replans: int = 4
    max_action_attempts: int = 2  # attempts per action (1 initial + retries)
    max_total_actions: int = 40  # overall execution bound
    max_total_plans: int = 10  # overall planning bound (includes clarifications)


@dataclass
class EpisodeResult:
    outcome: TrialOutcome
    claimed_success: bool = False
    verification: Optional[VerificationResult] = None
    events: list[dict] = field(default_factory=list)
    clarifications: list[ClarificationExchange] = field(default_factory=list)
    instruction_history: list[str] = field(default_factory=list)
    error: Optional[str] = None


class Orchestrator:
    def __init__(
        self,
        perception: Perception,
        planner: Planner,
        executor: Executor,
        env: RobotEnv,
        clarifier: ClarificationProvider,
        config: Optional[OrchestratorConfig] = None,
        store: Optional[ObservationStore] = None,
    ) -> None:
        self.perception = perception
        self.planner = planner
        self.executor = executor
        self.env = env
        self.clarifier = clarifier
        self.config = config or OrchestratorConfig()
        self.store = store

    # ------------------------------------------------------------------

    def run(self, instruction: str) -> EpisodeResult:
        result = EpisodeResult(outcome=TrialOutcome.ERROR)
        result.instruction_history.append(instruction)
        try:
            return self._run_inner(instruction, result)
        except KeyboardInterrupt:
            self._safe_stop(result, "interrupted")
            result.outcome = TrialOutcome.INTERRUPTED
            return result
        except Exception:
            self._safe_stop(result, "exception")
            result.outcome = TrialOutcome.ERROR
            result.error = traceback.format_exc()
            return result

    def _safe_stop(self, result: EpisodeResult, why: str) -> None:
        """Stop safely: hold the current pose (no new motion commands) and
        record the event.  Never claims success."""
        try:
            base = self.env.get_base_pose()
            self.env.set_base_target(float(base[0]), float(base[1]), float(base[2]))
            self.env.step(10)
        except Exception:
            pass
        self._event(result, "safe_stop", reason=why)

    def _event(self, result: EpisodeResult, kind: str, **data) -> None:
        payload = {"type": kind, "sim_time": self._now(), **data}
        result.events.append(payload)

    def _now(self) -> float:
        try:
            return self.env.sim_time()
        except Exception:
            return -1.0

    def _mark(self, frame_id: int) -> None:
        if self.store is not None and frame_id >= 0:
            try:
                self.store.mark_persist(frame_id)
            except KeyError:
                pass

    # ------------------------------------------------------------------

    def _run_inner(self, instruction: str, result: EpisodeResult) -> EpisodeResult:
        cfg = self.config
        self.perception.reset()
        self.planner.reset()
        self.executor.reset()

        context = ExecutionContext(scene=SceneDescription())
        history: list[ExecutionResult] = []
        clarification: Optional[str] = None
        search_attempts = 0
        clarifications_used = 0
        replans = 0
        plans_made = 0
        actions_executed = 0
        # Task goal (for final verification), learned from executed actions.
        grasp_target: Optional[str] = None
        place_region: Optional[str] = None
        reusable_verify: Optional[VerificationResult] = None

        while True:
            if plans_made >= cfg.max_total_plans:
                self._event(result, "limit", which="max_total_plans")
                result.outcome = TrialOutcome.LIMIT_EXCEEDED
                return result

            obs = self.env.get_obs()
            self._mark(obs.frame_id)
            scene = self.perception.describe(obs)
            context.scene = scene
            self._event(result, "perceive", frame_id=obs.frame_id,
                        objects=[g.instance_id for g in scene.objects],
                        ambiguities=list(scene.ambiguities))

            if plans_made == 0 and clarification is None:
                plan = self.planner.plan(instruction, scene)
            else:
                plan = self.planner.replan(instruction, scene, history, context, clarification)
            clarification = None
            plans_made += 1
            self._event(result, "plan", status=plan.status.value,
                        actions=[a.skill.value for a in plan.actions], reason=plan.reason)

            if plan.status is PlanStatus.INFEASIBLE:
                result.outcome = TrialOutcome.REFUSED
                self._event(result, "refused", reason=plan.reason)
                return result

            if plan.status is PlanStatus.NEEDS_CLARIFICATION:
                if clarifications_used >= cfg.max_clarifications:
                    result.outcome = TrialOutcome.CLARIFICATION_EXHAUSTED
                    self._event(result, "clarification_exhausted")
                    return result
                question = plan.clarification_question or "please clarify"
                response = self.clarifier.ask(question)  # consumes one scripted response
                clarifications_used += 1
                exchange = ClarificationExchange(question=question, response=response)
                result.clarifications.append(exchange)
                self._event(result, "clarification", question=question, response=response)
                if response is None:
                    result.outcome = TrialOutcome.CLARIFICATION_EXHAUSTED
                    return result
                result.instruction_history.append(f"[clarification] {response}")
                clarification = response
                continue

            errors = validate_plan(plan, context)
            if errors:
                self._event(result, "plan_invalid",
                            errors=[{"code": e.code, "index": e.action_index, "msg": e.message} for e in errors])
                replans += 1
                if replans > cfg.max_replans:
                    result.outcome = TrialOutcome.LIMIT_EXCEEDED
                    return result
                history.append(ExecutionResult(
                    action=Action(Skill.STOP), success=False, error_code=ErrorCode.INVALID_ACTION,
                    info={"validation_errors": [e.message for e in errors]}))
                continue

            # ---------------- execute this plan's actions ----------------
            replan_needed = False
            for action in plan.actions:
                if actions_executed >= cfg.max_total_actions:
                    self._event(result, "limit", which="max_total_actions")
                    self._safe_stop(result, "action budget exhausted")
                    result.outcome = TrialOutcome.LIMIT_EXCEEDED
                    return result

                attempts = 0
                exec_result: Optional[ExecutionResult] = None
                while attempts < cfg.max_action_attempts:
                    attempts += 1
                    actions_executed += 1
                    exec_result = self.executor.execute(action, self.env, self.perception)
                    history.append(exec_result)
                    self._event(result, "action", skill=action.skill.value, target=action.target,
                                success=exec_result.success, error=exec_result.error_code.value,
                                attempt=attempts, frame_id=exec_result.post_frame_id)
                    self._mark(exec_result.post_frame_id)
                    if exec_result.success:
                        break
                    if exec_result.error_code in (
                        ErrorCode.SEARCH_NOT_FOUND, ErrorCode.SEARCH_FATAL,
                        ErrorCode.PERCEPTION_ERROR, ErrorCode.INTERNAL_ERROR, ErrorCode.TARGET_LOST,
                    ):
                        break  # not retryable at the action level

                assert exec_result is not None

                if exec_result.success:
                    if action.skill in _STATE_CHANGING:
                        reusable_verify = None  # state changed: old VERIFY is stale
                    if action.skill is Skill.GRASP:
                        context.held_instance_id = action.target
                        grasp_target = action.target
                    elif action.skill is Skill.PLACE:
                        context.last_release_instance_id = context.held_instance_id
                        context.held_instance_id = None
                        place_region = action.target
                    elif action.skill is Skill.SEARCH:
                        # Target became visible: the remaining plan was made
                        # blind and is now obsolete -> replan from the new view.
                        self._event(result, "search_found", target=action.target)
                        replan_needed = True
                        break
                    elif action.skill is Skill.VERIFY and action.params.get("condition") == "object_in_region":
                        reusable_verify = VerificationResult(
                            passed=True, condition="object_in_region",
                            detail="in-plan VERIFY", frame_id=exec_result.post_frame_id)
                    continue

                # ---- failure handling ----
                code = exec_result.error_code
                if code is ErrorCode.SEARCH_NOT_FOUND:
                    search_attempts += 1
                    if search_attempts >= cfg.max_search_attempts:
                        result.outcome = TrialOutcome.SEARCH_EXHAUSTED
                        self._event(result, "search_exhausted")
                        return result
                    replan_needed = True
                    break
                if code in (ErrorCode.SEARCH_FATAL, ErrorCode.PERCEPTION_ERROR, ErrorCode.INTERNAL_ERROR):
                    self._safe_stop(result, f"fatal error {code.value}")
                    result.outcome = TrialOutcome.ERROR
                    result.error = f"fatal executor error: {code.value}"
                    return result
                # Invalidated / missing / uncertain target, or exhausted
                # retries on an ordinary failure: replan from a fresh view.
                replans += 1
                if replans > cfg.max_replans:
                    self._safe_stop(result, "replan budget exhausted")
                    result.outcome = TrialOutcome.FAILED
                    return result
                replan_needed = True
                break

            if replan_needed:
                continue

            # Plan completed.  Final verification gates any success claim.
            verification = reusable_verify or self._verify_final(grasp_target, place_region, context)
            result.verification = verification
            self._event(result, "final_verification", passed=verification.passed,
                        condition=verification.condition, detail=verification.detail,
                        frame_id=verification.frame_id, reused=verification is reusable_verify)
            if verification.passed:
                result.claimed_success = True
                result.outcome = TrialOutcome.CLAIMED_SUCCESS
                return result
            # Verification failed: allow replanning within budget.
            replans += 1
            if replans > cfg.max_replans:
                result.outcome = TrialOutcome.FAILED
                return result
            history.append(ExecutionResult(
                action=Action(Skill.VERIFY, params={"condition": "object_in_region"}),
                success=False, error_code=ErrorCode.VERIFY_FAILED,
                info={"detail": verification.detail}))

    # ------------------------------------------------------------------

    def _verify_final(self, grasp_target: Optional[str], place_region: Optional[str],
                      context: ExecutionContext) -> VerificationResult:
        """Vision-based task verification from a FRESH observation: the
        manipulated object must be grounded inside the destination region.
        No oracle access."""
        if grasp_target is None or place_region is None:
            return VerificationResult(False, "object_in_region",
                                      detail="no completed grasp+place to verify")
        obs = self.env.get_obs()
        self._mark(obs.frame_id)
        scene = self.perception.describe(obs)
        obj = scene.find(grasp_target)
        region = scene.find(place_region)
        if region is None:
            # instance ids can change between frames for some perception
            # implementations; fall back to matching by kind/name
            prev = context.scene.find(place_region)
            name = prev.name if prev else "red_region"
            region = next((r for r in scene.regions if r.name == name), None)
        if obj is None:
            prev = context.scene.find(grasp_target)
            name = prev.name if prev else None
            candidates = [o for o in scene.objects if name and o.name == name]
            obj = candidates[0] if len(candidates) >= 1 else None
        if obj is None or region is None:
            return VerificationResult(False, "object_in_region", frame_id=obs.frame_id,
                                      detail="object or region not visible for verification")
        if obj.pos_world is not None and region.pos_world is not None:
            dx = abs(obj.pos_world[0] - region.pos_world[0])
            dy = abs(obj.pos_world[1] - region.pos_world[1])
            ok = dx <= _REGION_HALF_XY + _VERIFY_XY_SLACK and dy <= _REGION_HALF_XY + _VERIFY_XY_SLACK
            return VerificationResult(ok, "object_in_region", frame_id=obs.frame_id,
                                      detail=f"3D offset dx={dx:.3f} dy={dy:.3f}")
        if obj.bbox_xyxy is not None and region.bbox_xyxy is not None:
            cx = (obj.bbox_xyxy[0] + obj.bbox_xyxy[2]) / 2
            cy = (obj.bbox_xyxy[1] + obj.bbox_xyxy[3]) / 2
            x1, y1, x2, y2 = region.bbox_xyxy
            ok = x1 <= cx <= x2 and y1 <= cy <= y2
            return VerificationResult(ok, "object_in_region", frame_id=obs.frame_id,
                                      detail="2D bbox-center containment")
        return VerificationResult(False, "object_in_region", frame_id=obs.frame_id,
                                  detail="insufficient grounding for verification")
