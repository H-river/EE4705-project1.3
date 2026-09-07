# Owner: backbone (ALL)
"""Structural plan validation: validate_plan(plan, context) -> list[PlanError].

Validation simulates plan state (held object) starting from the *current*
execution context, checks per-skill reference and parameter requirements,
and returns all findings.  An empty list means the plan is structurally
valid to execute.

IMPORTANT: plan validity is permission to *attempt* the actions, nothing
more.  A valid (possibly partial) plan never bypasses final task
verification — only the orchestrator's final VERIFY gates CLAIMED_SUCCESS.

Error codes (PlanError.code):
  EMPTY_PLAN            plan has no actions but status requires some
  BAD_STATUS            plan status inconsistent with its content
  MISSING_REFERENCE     target references an instance_id not in the scene
  UNCERTAIN_REFERENCE   target references an AMBIGUOUS instance
  UNLOCATED_REFERENCE   metric skill references an instance without pos_world
  NOT_HOLDING           PLACE/MOVE_TO-with-held-object while nothing held
  ALREADY_HOLDING       GRASP while (simulated) already holding
  WRONG_OBJECT          PLACE references an object other than the held one
  SEARCH_IN_READY       SEARCH action inside a READY plan
  BAD_PARAM             missing/mistyped/out-of-bounds parameter
  BAD_TERMINATION       actions after STOP
  UNKNOWN_SKILL         action.skill is not a Skill enum member

Per-skill requirements (decision recorded in docs/DECISIONS.md; the prompt's
acceptance cases fix the semantics):

  SEARCH    target: class-name string (no grounded instance needed).
            Only allowed when plan.status is NEEDS_SEARCH.
  APPROACH  target: instance with pos_world (LOCALIZED), or params["pos"].
  REACH     target: LOCALIZED instance only (unlocated is an error).
  GRASP     target: LOCALIZED instance; requires simulated hand empty.
  MOVE_TO   params["pos"] finite 3-vector inside workspace bounds, OR a
            LOCALIZED region/instance target.  Allowed while holding.
  PLACE     target: LOCALIZED region instance, or region plus params["pos"]. Requires
            simulated holding.  Optional params["object"]: must equal the
            simulated held instance id.
  VERIFY    params["condition"] in {"object_visible", "object_in_region",
            "holding"}.  object_visible may reference an UNLOCALIZED
            instance (2D evidence suffices); object_in_region requires the
            region reference to exist.
  STOP      no params; must be the last action if present.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from core.types import (
    Action,
    ExecutionContext,
    GroundedObject,
    GroundStatus,
    Plan,
    PlanError,
    PlanStatus,
    Skill,
)

# Workspace bounds for positional parameters, world frame, meters.
WORKSPACE_MIN = (-3.0, -3.0, -0.5)
WORKSPACE_MAX = (3.0, 3.0, 2.5)

_METRIC_SKILLS = {Skill.APPROACH, Skill.REACH, Skill.GRASP}
_VERIFY_CONDITIONS = {"object_visible", "object_in_region", "holding"}


def _finite_vec3(value: Any) -> Optional[tuple[float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return (out[0], out[1], out[2])


def _in_workspace(p: tuple[float, float, float]) -> bool:
    return all(WORKSPACE_MIN[i] <= p[i] <= WORKSPACE_MAX[i] for i in range(3))


def _lookup(context: ExecutionContext, instance_id: Optional[str]) -> Optional[GroundedObject]:
    if instance_id is None:
        return None
    return context.scene.find(instance_id)


def validate_plan(plan: Plan, context: ExecutionContext) -> list[PlanError]:
    errors: list[PlanError] = []

    if plan.status in (PlanStatus.NEEDS_CLARIFICATION, PlanStatus.INFEASIBLE):
        if plan.actions:
            errors.append(PlanError("BAD_STATUS", -1, f"{plan.status.value} plan must carry no actions"))
        if plan.status is PlanStatus.NEEDS_CLARIFICATION and not plan.clarification_question:
            errors.append(PlanError("BAD_STATUS", -1, "NEEDS_CLARIFICATION plan requires clarification_question"))
        if plan.status is PlanStatus.INFEASIBLE and not plan.reason:
            errors.append(PlanError("BAD_STATUS", -1, "INFEASIBLE plan requires reason"))
        return errors

    if not plan.actions:
        errors.append(PlanError("EMPTY_PLAN", -1, "executable plan has no actions"))
        return errors

    # Simulated plan state, initialized from the current execution context.
    held: Optional[str] = context.held_instance_id
    stopped_at: Optional[int] = None

    for i, action in enumerate(plan.actions):
        if not isinstance(action.skill, Skill):
            errors.append(PlanError("UNKNOWN_SKILL", i, f"unknown skill {action.skill!r}"))
            continue
        if stopped_at is not None:
            errors.append(PlanError("BAD_TERMINATION", i, f"action after STOP (index {stopped_at})"))
            continue

        skill = action.skill
        target = action.target
        params = action.params or {}
        ref = _lookup(context, target)

        # All supported waypoint overrides share the executor's finite
        # coordinate contract; APPROACH/MOVE_TO validate theirs below.
        if skill in (Skill.REACH, Skill.GRASP, Skill.PLACE) and params.get("pos") is not None:
            p = _finite_vec3(params["pos"])
            if p is None or not _in_workspace(p):
                errors.append(PlanError("BAD_PARAM", i, f"{skill.value} pos must be a finite in-workspace 3-vector"))

        def ref_errors(*, need_located: bool, allow_unlocalized: bool = False) -> bool:
            """Append reference errors; return True when reference usable."""
            if target is None:
                errors.append(PlanError("MISSING_REFERENCE", i, f"{skill.value} requires a target"))
                return False
            if ref is None:
                errors.append(PlanError("MISSING_REFERENCE", i, f"{skill.value} target {target!r} not in scene"))
                return False
            if ref.status is GroundStatus.AMBIGUOUS:
                errors.append(PlanError("UNCERTAIN_REFERENCE", i, f"{skill.value} target {target!r} is ambiguous"))
                return False
            if ref.status is GroundStatus.NOT_FOUND:
                errors.append(PlanError("MISSING_REFERENCE", i, f"{skill.value} target {target!r} has no visual evidence"))
                return False
            if need_located and ref.status is not GroundStatus.LOCALIZED:
                if not allow_unlocalized:
                    errors.append(
                        PlanError("UNLOCATED_REFERENCE", i, f"{skill.value} target {target!r} has no 3D position")
                    )
                    return False
            return True

        if skill is Skill.SEARCH:
            if plan.status is PlanStatus.READY:
                errors.append(PlanError("SEARCH_IN_READY", i, "SEARCH is not allowed inside a READY plan"))
            if not target or not isinstance(target, str):
                errors.append(PlanError("BAD_PARAM", i, "SEARCH requires a class-name target string"))

        elif skill is Skill.APPROACH:
            pos = params.get("pos")
            if pos is not None:
                p = _finite_vec3(pos)
                if p is None:
                    errors.append(PlanError("BAD_PARAM", i, f"APPROACH params['pos'] must be a finite 3-vector, got {pos!r}"))
                elif not _in_workspace(p):
                    errors.append(PlanError("BAD_PARAM", i, f"APPROACH pos {p} outside workspace bounds"))
            else:
                ref_errors(need_located=True)

        elif skill is Skill.REACH:
            ref_errors(need_located=True)

        elif skill is Skill.GRASP:
            if ref_errors(need_located=True):
                if held is not None:
                    errors.append(PlanError("ALREADY_HOLDING", i, f"GRASP while already holding {held!r}"))
                else:
                    held = target

        elif skill is Skill.MOVE_TO:
            pos = params.get("pos")
            if pos is not None:
                p = _finite_vec3(pos)
                if p is None:
                    errors.append(PlanError("BAD_PARAM", i, f"MOVE_TO params['pos'] must be a finite 3-vector, got {pos!r}"))
                elif not _in_workspace(p):
                    errors.append(PlanError("BAD_PARAM", i, f"MOVE_TO pos {p} outside workspace bounds"))
            elif target is not None:
                ref_errors(need_located=True)
            else:
                errors.append(PlanError("BAD_PARAM", i, "MOVE_TO requires params['pos'] or a located target"))

        elif skill is Skill.PLACE:
            if held is None:
                errors.append(PlanError("NOT_HOLDING", i, "PLACE while not holding any object"))
            obj_ref = params.get("object")
            if obj_ref is not None:
                if not isinstance(obj_ref, str):
                    errors.append(PlanError("BAD_PARAM", i, "PLACE params['object'] must be an instance id string"))
                elif held is not None and obj_ref != held:
                    errors.append(
                        PlanError("WRONG_OBJECT", i, f"PLACE object {obj_ref!r} differs from held object {held!r}")
                    )
            if target is None:
                errors.append(PlanError("MISSING_REFERENCE", i, "PLACE requires a region target"))
            else:
                if ref is None:
                    errors.append(PlanError("MISSING_REFERENCE", i, f"PLACE region {target!r} not in scene"))
                elif ref.kind != "region":
                    errors.append(PlanError("BAD_PARAM", i, f"PLACE target {target!r} is not a region"))
                elif ref.status is GroundStatus.AMBIGUOUS:
                    errors.append(PlanError("UNCERTAIN_REFERENCE", i, f"PLACE region {target!r} is ambiguous"))
                elif ref.status is GroundStatus.NOT_FOUND:
                    errors.append(PlanError("MISSING_REFERENCE", i, f"PLACE region {target!r} has no visual evidence"))
                elif params.get("pos") is None and ref.status is not GroundStatus.LOCALIZED:
                    errors.append(PlanError("UNLOCATED_REFERENCE", i, f"PLACE region {target!r} has no 3D position"))
            if held is not None:
                held = None  # simulated release

        elif skill is Skill.VERIFY:
            condition = params.get("condition")
            if condition not in _VERIFY_CONDITIONS:
                errors.append(
                    PlanError("BAD_PARAM", i, f"VERIFY condition must be one of {sorted(_VERIFY_CONDITIONS)}, got {condition!r}")
                )
            elif condition == "object_visible":
                # 2D evidence suffices: an UNLOCALIZED instance is fine.
                ref_errors(need_located=False)
            elif condition == "object_in_region":
                obj = params.get("object")
                region = params.get("region")
                for key, val in (("object", obj), ("region", region)):
                    if not isinstance(val, str):
                        errors.append(PlanError("BAD_PARAM", i, f"VERIFY object_in_region requires string params[{key!r}]"))
                    elif (_lookup(context, val) is None and not
                          (key == "object" and val in
                           (context.held_instance_id, context.last_release_instance_id))):
                        # A held/recently released instance may be occluded in
                        # this planning frame. Its tracked identity permits a
                        # future VERIFY, never motion or a success claim.
                        errors.append(PlanError("MISSING_REFERENCE", i, f"VERIFY {key} {val!r} not in scene"))
            elif condition == "holding":
                if held is None:
                    errors.append(PlanError("NOT_HOLDING", i, "VERIFY holding while (simulated) not holding"))

        elif skill is Skill.STOP:
            if params:
                errors.append(PlanError("BAD_PARAM", i, "STOP takes no params"))
            stopped_at = i

    return errors
