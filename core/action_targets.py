# Owner: backbone (ALL)
"""Resolve public action coordinates without simulator identities or oracle access."""

from __future__ import annotations

import numpy as np

from core.types import Action, ErrorCode, GroundStatus, SceneDescription, Skill

TRANSPORT_CLEARANCE_M = 0.18
_POSITION_SKILLS = {Skill.APPROACH, Skill.REACH, Skill.GRASP, Skill.MOVE_TO, Skill.PLACE}


class TargetResolutionError(ValueError):
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.TARGET_LOST) -> None:
        super().__init__(message)
        self.error_code = error_code


def resolve_action_position(action: Action, scene: SceneDescription) -> np.ndarray:
    """Return world meters. Explicit ``params.pos`` remains an override.

    Otherwise resolve the exact perceived instance ID in a fresh scene.
    MOVE_TO a region means its support point plus 0.18 m transport clearance;
    PLACE means its support point. Object targets mean estimated centers.
    Missing, ambiguous or unlocalized references never fall back by name.
    This helper does not replace validate_plan's precondition checks.
    """
    if action.skill not in _POSITION_SKILLS:
        raise TargetResolutionError("skill has no positional target", ErrorCode.INVALID_ACTION)
    explicit = (action.params or {}).get("pos")
    if explicit is not None:
        if (not isinstance(explicit, (list, tuple)) or len(explicit) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in explicit)):
            raise TargetResolutionError("pos must be a finite 3-vector", ErrorCode.INVALID_ACTION)
        pos = np.asarray(explicit, dtype=float)
        if not np.all(np.isfinite(pos)):
            raise TargetResolutionError("pos must be finite", ErrorCode.INVALID_ACTION)
        return pos.copy()
    ref = scene.find(action.target or "")
    if ref is None or ref.status is not GroundStatus.LOCALIZED or ref.pos_world is None:
        raise TargetResolutionError(f"target {action.target!r} is missing, ambiguous or unlocalized")
    if action.skill is Skill.PLACE and ref.kind != "region":
        raise TargetResolutionError("PLACE target must be a region", ErrorCode.INVALID_ACTION)
    pos = np.asarray(ref.pos_world, dtype=float).copy()
    if action.skill is Skill.MOVE_TO and ref.kind == "region":
        pos[2] += TRANSPORT_CLEARANCE_M
    return pos
