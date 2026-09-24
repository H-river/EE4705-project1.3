# Owner: Student B
"""Final2 2.4: history-aware replanning.

If the same action on the same instance has already failed twice in this
episode, B does not emit it unchanged again. In order: a SEARCH first (when
none ran since that action last failed), then an APPROACH from a standoff
rotated by 90 degrees (params ``standoff_rotation_deg``, +90 then -90),
then REJECTED with a reason that says so.
"""
from __future__ import annotations

from core.types import Action, Plan, PlanStatus, Skill

REPEAT_LIMIT = 2
ROTATIONS = (90.0, -90.0)
_IGNORED = (Skill.STOP, Skill.VERIFY)  # the orchestrator's synthetic history entries


def failure_counts(history):
    counts, last = {}, {}
    for index, result in enumerate(history):
        action = result.action
        if result.success or action.skill in _IGNORED or not action.target:
            continue
        key = (action.skill, action.target)
        counts[key] = counts.get(key, 0) + 1
        last[key] = index
    return counts, last


def _searched_since(history, index):
    return any(r.action.skill is Skill.SEARCH for r in history[index + 1:])


def guard(plan, history, goal, search_target):
    """Return (plan, audit or None). ``search_target(cls)`` builds a SEARCH
    target for a goal class."""
    if plan.status is not PlanStatus.READY or not history:
        return plan, None
    counts, last = failure_counts(history)
    repeated = [(i, a) for i, a in enumerate(plan.actions)
                if counts.get((a.skill, a.target), 0) >= REPEAT_LIMIT]
    if not repeated:
        return plan, None
    index, action = repeated[0]
    key = (action.skill, action.target)
    n = counts[key]
    audit = {"repeated": f"{action.skill.value}({action.target})", "failures": n}
    goal = goal or {}
    role = ("object" if action.target == goal.get("object_id")
            else "region" if action.target == goal.get("region_id") else None)
    if role and not _searched_since(history, last[key]):
        cls = goal.get(role + "_name")
        audit["decision"] = "SEARCH first"
        return Plan(status=PlanStatus.NEEDS_SEARCH, actions=[Action(Skill.SEARCH, search_target(cls))],
                    reason=f"{action.skill.value} on {action.target} failed {n} times; "
                           f"re-localizing {cls} before trying again"), audit
    approach = next((i for i, a in enumerate(plan.actions[:index + 1]) if a.skill is Skill.APPROACH), None)
    if (role == "object" and action.skill in (Skill.APPROACH, Skill.GRASP) and approach is not None
            and n - REPEAT_LIMIT < len(ROTATIONS)):
        rotation = ROTATIONS[n - REPEAT_LIMIT]
        old = plan.actions[approach]
        plan.actions[approach] = Action(old.skill, old.target, {**old.params, "standoff_rotation_deg": rotation})
        audit["decision"] = f"APPROACH standoff rotated {rotation:+.0f} deg"
        return plan, audit
    audit["decision"] = "REJECTED"
    tried = ("after re-localizing and changing the approach direction"
             if role == "object" and action.skill in (Skill.APPROACH, Skill.GRASP) else
             "after re-localizing" if role else "")
    return Plan(status=PlanStatus.REJECTED,
                reason=f"{action.skill.value} on {action.target} already failed {n} times in this episode"
                       f"{' (' + tried + ')' if tried else ''}; not repeating it"), audit
