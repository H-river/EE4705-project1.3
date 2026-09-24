# Owner: Student B
"""Final2 6.1: one user-facing sentence about how the episode ended.

Composed from B's bound goal, the orchestrator outcome, the planner's reasons
and the final (vision) verification. No model call, no oracle data: it says
what the system itself believes, e.g. "Placed the grey stone on the red area,
confirmed visually." / "Could not confirm - the stone is out of view."
"""
from __future__ import annotations

import re

_REGION_WORDS = {"red_region": "red area"}


def _colour_word(colour):
    return (colour or "").replace("_", " ").replace("gray", "grey").strip()


def _object_phrase(goal):
    goal = goal or {}
    name = goal.get("object_name") or "object"
    colour = _colour_word(goal.get("object_color"))
    return f"the {colour + ' ' if colour else ''}{name}"


def _region_phrase(goal):
    region = (goal or {}).get("region_name") or ""
    return "the " + _REGION_WORDS.get(region, region.replace("_", " ") or "target area")


def _offset_cm(detail):
    m = re.search(r"offset xyz=\[([-\d.e]+), ([-\d.e]+)", detail or "")
    if not m:
        return None
    return 100 * (float(m.group(1)) ** 2 + float(m.group(2)) ** 2) ** 0.5


def outcome_sentence(outcome, goal=None, verification=None, events=(), clarifications=(), rationales=()):
    obj, region = _object_phrase(goal), _region_phrase(goal)
    detail = (verification or {}).get("detail") or ""
    plans = [e for e in events if e.get("type") == "plan"]
    last_reason = next((e.get("reason") for e in reversed(plans) if e.get("reason")), "") or ""
    if outcome == "CLAIMED_SUCCESS":
        off = _offset_cm(detail)
        where = f" ({off:.0f} cm from its centre)" if off is not None else ""
        return f"Placed {obj} on {region}{where}, confirmed visually."
    if outcome == "REFUSED":
        reason = last_reason or next((r.get("rationale") for r in reversed(list(rationales)) if r.get("rationale")), "")
        return f"Did not start: {reason.rstrip('.')}." if reason else "Did not start: the request is not supported."
    if outcome == "CLARIFICATION_EXHAUSTED":
        question = next((c.get("question") for c in reversed(list(clarifications)) if c.get("question")), "")
        return (f"Stopped: I needed an answer to \"{question}\" and did not get one." if question
                else "Stopped: I needed a clarification and did not get one.")
    if outcome == "SEARCH_EXHAUSTED":
        return f"Could not find {obj} after searching the table."
    if outcome in ("FAILED", "LIMIT_EXCEEDED"):
        if "not visible" in detail:
            return f"Could not confirm - {obj} is out of view."
        if "3D grounding" in detail:
            return f"Could not confirm - I see {obj} but cannot tell exactly where it is."
        if detail and "offset" in detail:
            return f"Could not confirm - {obj} is not inside {region} ({detail.split(';')[0]})."
        failed = [e for e in events if e.get("type") == "action" and not e.get("success")]
        if failed:
            last = failed[-1]
            return (f"Gave up after repeated failures; the last was {last.get('skill')} "
                    f"({str(last.get('error', '')).lower().replace('_', ' ')}).")
        if last_reason:
            return f"Gave up: {last_reason.rstrip('.')}."
        return f"Gave up before moving {obj}."
    if outcome == "INTERRUPTED":
        return "Stopped: the run was interrupted."
    return "Stopped on an internal error; nothing is confirmed."
