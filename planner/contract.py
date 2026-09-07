"""Qwen wire format -> the existing backbone Plan. No model-written coordinates."""
from __future__ import annotations

from core.action_targets import resolve_action_position
from core.llm_client import validate_schema
from core.types import Action, GroundStatus, Plan, PlanStatus, Skill
from core.validation import validate_plan
from core.vocab import Vocab

WIRE_VERSION = "student-b-plan-v1"


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_TEXT = {"type": "string", "maxLength": 120}
GOAL_SCHEMA = _object({key: _TEXT for key in
                      ("object_id", "object_name", "object_color", "region_id", "region_name")})
WIRE_SCHEMA = _object({
    "schema_version": {"type": "string", "enum": [WIRE_VERSION]},
    "status": {"type": "string", "enum": [s.value for s in PlanStatus]},
    "goal": GOAL_SCHEMA,
    "actions": {"type": "array", "maxItems": 16, "items": _object({
        "skill": {"type": "string", "enum": [s.value for s in Skill]},
        "target": _TEXT, "object": _TEXT, "region": _TEXT,
        "condition": {"type": "string", "enum": ["", "holding", "object_visible", "object_in_region"]},
    })},
    "reason": {"type": "string", "maxLength": 1000},
    "clarification_question": {"type": "string", "maxLength": 500},
})


class PlanContractError(ValueError):
    pass


def public_vocabulary():
    """Only public class names and phrases; no simulator IDs or poses."""
    rows = []
    for entry in Vocab().entries:
        row = {"name": entry.cls, "kind": entry.kind,
               "synonyms": entry.synonyms, "attributes": entry.attributes}
        if row not in rows:
            rows.append(row)
    return rows


def compile_plan(wire, context, locked_goal=None, known=None):
    """Return Plan + validated goal, or reject. This cannot prove language accuracy or IK."""
    validate_schema(wire, WIRE_SCHEMA)
    scene, goal = context.scene, dict(wire["goal"])
    known = known or {}
    status = PlanStatus(wire["status"])

    def require(ok, message):
        if not ok:
            raise PlanContractError(message)

    if locked_goal:
        for key, value in locked_goal.items():
            require(not value or goal[key] == value, f"Original goal must keep {key}={value!r}")
    vocab = public_vocabulary()
    for role in ("object", "region"):
        name, ident = goal[role + "_name"], goal[role + "_id"]
        require(not name or any(v["name"] == name and v["kind"] == role for v in vocab),
                f"Unknown {role} class {name!r}")
        ref = scene.find(ident) if ident else None
        if ref is None and ident and locked_goal and ident == locked_goal[role + "_id"]:
            ref = known.get(ident)  # identity only; motion always uses the new scene
        require(not ident or ref is not None, f"Unknown perceived {role} ID {ident!r}")
        if ref:
            require(ref.kind == role and name == ref.name, f"Goal {role} ID/class/role mismatch")
            if role == "object" and goal["object_color"]:
                require(ref.attributes.get("color") == goal["object_color"], "Goal object color mismatch")
    if status is PlanStatus.READY:
        require(bool(goal["object_id"] and goal["region_id"]), "READY needs both grounded goal IDs")
        require(not context.held_instance_id or context.held_instance_id == goal["object_id"],
                "Held object differs from original goal; do not place it as the requested object")
        require(not wire["clarification_question"], "READY cannot ask a clarification question")

    plan = Plan(status=status, reason=wire["reason"] or None,
                clarification_question=wire["clarification_question"] or None)
    held = context.held_instance_id
    released = bool(not held and context.last_release_instance_id == goal["object_id"])
    approached = transported = verified = False
    for index, item in enumerate(wire["actions"]):
        skill, target = Skill(item["skill"]), item["target"]
        require(not verified or skill is Skill.STOP, "Only STOP may follow final placement VERIFY")
        used = {"skill", "target"}
        params = {}
        if skill is Skill.PLACE:
            used.add("object")
            params["object"] = item["object"]
        elif skill is Skill.VERIFY:
            used.add("condition")
            params["condition"] = item["condition"]
            if item["condition"] == "object_in_region":
                used.update(("object", "region"))
                params.update(object=item["object"], region=item["region"])
                require(not target, "Placement VERIFY uses object/region fields, with empty target")
        require(all(not value for key, value in item.items() if key not in used),
                f"Action {index}: unused fields must be empty strings")
        require(status in (PlanStatus.READY, PlanStatus.NEEDS_SEARCH), "Non-action status must have no actions")
        if status is PlanStatus.NEEDS_SEARCH:
            require(skill is Skill.SEARCH, "NEEDS_SEARCH may only contain SEARCH actions")
            require(bool(target) and target in (goal["object_name"], goal["region_name"]),
                    "SEARCH must name a goal class; preserve color and other constraints in the goal")
        else:
            require(skill is not Skill.SEARCH, "READY cannot SEARCH")
            if skill in (Skill.APPROACH, Skill.REACH, Skill.GRASP):
                require(target == goal["object_id"] and not held, "Object motion must target the goal with an empty hand")
                if skill is Skill.GRASP:
                    require(approached, "GRASP needs APPROACH or REACH first")
                    held, released = target, False
                    approached = transported = False
                else:
                    approached = True
            elif skill is Skill.MOVE_TO:
                require(held == goal["object_id"] and target == goal["region_id"],
                        "MOVE_TO must carry the goal object to the goal region")
                transported = True
            elif skill is Skill.PLACE:
                require(held == goal["object_id"] and target == goal["region_id"]
                        and item["object"] == held and transported,
                        "PLACE requires the goal object, goal region, and a preceding MOVE_TO")
                held, released = None, True
                approached = transported = False
            elif skill is Skill.VERIFY:
                if item["condition"] == "object_in_region":
                    require(released and item["object"] == goal["object_id"]
                            and item["region"] == goal["region_id"],
                            "Final VERIFY must check the released goal object in the goal region")
                    verified = True
                else:
                    require(target == goal["object_id"], "VERIFY must reference the goal object")
                    require(item["condition"] != "holding" or held == target, "VERIFY holding needs the goal in hand")
            elif skill is Skill.STOP:
                require(not target and verified and index == len(wire["actions"]) - 1,
                        "STOP must be last, after final placement VERIFY")

        action = Action(skill, target or None, params)
        if skill in (Skill.APPROACH, Skill.REACH, Skill.GRASP, Skill.MOVE_TO, Skill.PLACE):
            ref = scene.find(target)
            require(ref is not None and ref.status is GroundStatus.LOCALIZED,
                    f"Action {index}: target needs a current, unambiguous 3D position")
            params["pos"] = resolve_action_position(action, scene).tolist()
        plan.actions.append(action)
    if status is PlanStatus.READY:
        require(verified and bool(plan.actions) and plan.actions[-1].skill is Skill.STOP,
                "READY must finish with placement VERIFY and STOP")
    errors = validate_plan(plan, context)
    require(not errors, "; ".join(f"{e.code} at action {e.action_index}: {e.message}" for e in errors))
    return plan, goal
