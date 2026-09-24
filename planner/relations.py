# Owner: Student B
"""Final2 2.3: deterministic binding of relational object references.

The model names the relation and the anchor's class (``reference`` in the
wire output); code decides which perceived instance satisfies it, from A's
``pos_world`` only. Robot frame: x ahead of the base, y to its left.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from core.types import GroundStatus

RELATIONS = ("", "next_to", "left_of", "right_of", "closest_to", "farthest_from")
MARGIN_M = 0.03  # a relation must separate the best candidate by this much


def _colour_ok(colour, attribute):
    from planner.contract import colour_matches
    return not colour or colour_matches(colour, attribute)


def _located(scene, name, colour, kind="object"):
    return [g for g in scene.objects + scene.regions
            if g.kind == kind and g.name == name and g.status is GroundStatus.LOCALIZED
            and g.pos_world is not None and _colour_ok(colour, g.attributes.get("color"))]


def resolve(scene, object_name, object_color, reference, base_yaw=0.0):
    """Return (instance_id or None, detail)."""
    relation = (reference or {}).get("relation") or ""
    anchor_class = (reference or {}).get("anchor_class") or ""
    if relation not in RELATIONS or not relation or not anchor_class:
        return None, "no relation"
    candidates = _located(scene, object_name, object_color)
    anchor_kind = "region" if anchor_class.endswith("_region") else "object"
    anchors = [a for a in _located(scene, anchor_class, reference.get("anchor_color") or "", anchor_kind)
               if a.instance_id not in {c.instance_id for c in candidates}]
    if len(anchors) != 1:
        return None, f"{len(anchors)} anchors of class {anchor_class!r}"
    if not candidates:
        return None, f"no localized {object_name!r}"
    anchor = np.asarray(anchors[0].pos_world[:2], dtype=float)
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    scored = []
    for g in candidates:
        d = np.asarray(g.pos_world[:2], dtype=float) - anchor
        lateral = -s * d[0] + c * d[1]  # + = left of the anchor, seen from the robot
        scored.append((g.instance_id, float(np.linalg.norm(d)), float(lateral)))
    if relation in ("next_to", "closest_to", "farthest_from"):
        far = relation == "farthest_from"
        ranked = sorted(scored, key=lambda t: -t[1] if far else t[1])
        if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < MARGIN_M:
            return None, "distances too close to separate"
        return ranked[0][0], f"{relation} {anchors[0].instance_id}: {ranked[0][1]:.3f} m"
    sign = 1.0 if relation == "left_of" else -1.0
    side = [t for t in scored if sign * t[2] > MARGIN_M]
    if not side:
        return None, f"no {object_name!r} {relation} the anchor"
    best = min(side, key=lambda t: t[1])
    return best[0], f"{relation} {anchors[0].instance_id}: lateral {best[2]:+.3f} m"


def bind_reference(wire, scene, base_yaw=0.0, locked_goal=None, normalizations=None):
    """Return the wire with the goal object replaced by the instance the
    relation selects, when the relation is decisive and the goal is not
    locked yet. The model's other choices are kept."""
    reference = wire.get("reference") if isinstance(wire, dict) else None
    goal = (wire or {}).get("goal") or {}
    if not reference or not reference.get("relation") or wire.get("status") != "READY":
        return wire
    if locked_goal and locked_goal.get("object_id"):
        return wire
    chosen, detail = resolve(scene, goal.get("object_name"), goal.get("object_color"), reference, base_yaw)
    old = goal.get("object_id")
    if chosen is None or chosen == old:
        if normalizations is not None and chosen is None:
            normalizations.append({"field": "reference", "change": "not decisive", "detail": detail})
        return wire
    out = copy.deepcopy(wire)
    out["goal"]["object_id"] = chosen
    for action in out["actions"]:
        for field in ("target", "object"):
            if old and action.get(field) == old:
                action[field] = chosen
    if normalizations is not None:
        normalizations.append({"field": "goal.object_id", "change": "relational binding",
                               "from": old, "to": chosen, "detail": detail})
    return out
