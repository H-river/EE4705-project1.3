# Owner: backbone (ALL)
"""Conservative placement verification from current perception and proprioception."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from core.interfaces import Perception, RobotEnvProtocol
from core.placement import STABILITY_DRIFT_M, position_in_region
from core.types import GroundStatus, SceneDescription, VerificationResult
from core.vocab import Vocab

VISUAL_STABILITY_S = 0.2


@lru_cache(maxsize=1)
def _known_region_sizes() -> dict[str, tuple[float, float]]:
    # Class-level nominal geometry, not simulator body IDs or positions.
    return {entry.cls: (entry.size_xyz[0] / 2, entry.size_xyz[1] / 2)
            for entry in Vocab().entries if entry.kind == "region"}


def check_placement(scene: SceneDescription, object_id: str, region_id: str,
                    *, attached: bool) -> VerificationResult:
    """Check one frame. Unknown evidence fails closed; IDs must remain stable.

    Region pos_world denotes the support point. A supplied region extent
    overrides class-level nominal sizes. Bounding boxes alone cannot prove
    that an object has been released onto a support surface.
    """
    def verdict(ok, detail):
        return VerificationResult(ok, "object_in_region", detail, scene.frame_id)
    if attached:
        return verdict(False, "object is still attached")
    obj, region = scene.find(object_id), scene.find(region_id)
    if obj is None or region is None:
        return verdict(False, "exact object or region instance is not visible")
    if obj.kind != "object" or region.kind != "region":
        return verdict(False, "incorrect object/region kind")
    if any(g.status is not GroundStatus.LOCALIZED or g.pos_world is None for g in (obj, region)):
        return verdict(False, "reliable 3D grounding is required")
    if any(g.frame_id >= 0 and g.frame_id != scene.frame_id for g in (obj, region)):
        return verdict(False, "grounding is from a different observation")
    half = region.region_half_extents_xy
    if half is None:
        half = _known_region_sizes().get(region.name)
    if half is None:
        return verdict(False, "region extent is unknown")
    delta = np.asarray(obj.pos_world) - region.pos_world
    ok = position_in_region(obj.pos_world, region.pos_world, half)
    return verdict(ok, f"offset xyz={delta.round(4).tolist()}, half extents={list(half)}")


def verify_placement(env: RobotEnvProtocol, perception: Perception,
                     object_id: str, region_id: str) -> VerificationResult:
    """Two fresh visual checks 0.2 simulated seconds apart, with release,
    height, footprint and <=2 cm object drift required at both checks.
    The evaluator separately performs its longer ground-truth stability test.
    """
    first = None
    for sample in range(2):
        if sample:
            env.step(max(1, int(round(VISUAL_STABILITY_S / env.timestep()))))
        obs = env.get_obs()
        scene = perception.describe(obs)
        if scene.frame_id != obs.frame_id or abs(scene.sim_time - obs.sim_time) > 1e-9:
            return VerificationResult(False, "object_in_region", "stale scene observation", obs.frame_id)
        result = check_placement(scene, object_id, region_id, attached=env.is_attached())
        if not result.passed:
            return result
        obj, region = scene.find(object_id), scene.find(region_id)
        if first is not None:
            old_name, old_pos, old_region_name = first
            if obj.name != old_name or region.name != old_region_name:
                return VerificationResult(False, "object_in_region", "instance identity changed", obs.frame_id)
            drift = np.linalg.norm(np.asarray(obj.pos_world) - old_pos)
            if drift > STABILITY_DRIFT_M:
                return VerificationResult(False, "object_in_region", f"object drift {drift:.4f} m", obs.frame_id)
        first = obj.name, np.asarray(obj.pos_world).copy(), region.name
    result.detail += f"; released and stable across {VISUAL_STABILITY_S}s"
    return result
