# Owner: backbone (ALL)
"""EvalOracle: privileged ground-truth access for evaluation and mocks.

The oracle is the ONLY component allowed to associate perceived instances
with ground-truth objects, and it does so geometrically (projected-bbox
IoU on segmentation renders from the same camera and simulation state as
the perception result) — NEVER by comparing perceived ID strings with
ground-truth ID strings.

Architectural rule (enforced by tests/test_architecture.py): the real
Student module directories (perception/, planner/, executor/) must not
import this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.rendering import mujoco
from core.types import SceneDescription
from core.world import MANIPULABLE_BODIES, RegionBounds, SimWorld

# Visibility: a ground-truth object is visible from a camera when at least
# MIN_VISIBLE_PIXELS of its geoms survive in the segmentation render
# (this accounts for both field of view and occlusion).
MIN_VISIBLE_PIXELS = 30

# Association: greedy one-to-one matching on bbox IoU.  Pairs below
# IOU_THRESHOLD never match.  A perceived box whose best two candidate IoUs
# are both above threshold and closer than AMBIGUITY_MARGIN is reported as
# ambiguous and left unmatched.
IOU_THRESHOLD = 0.30
AMBIGUITY_MARGIN = 0.10

# Placement: object center must lie inside the region's x/y bounds and rest
# within [support_z - 0.005, support_z + REGION_MAX_HEIGHT] (e.g. on top of
# another object still counts as "in the region" only up to this height).
REGION_MAX_HEIGHT = 0.12

# Stability: maintained over the whole interval, sampled every SAMPLE_DT.
STABILITY_DURATION_S = 2.0
STABILITY_SAMPLE_DT = 0.1
STABILITY_POS_TOL = 0.02  # meters of drift from the first sample
STABILITY_VEL_TOL = 0.05  # m/s linear speed at every sample


@dataclass
class AssociationResult:
    """Outcome of perceived <-> ground-truth association for one scene."""

    matches: dict[str, str] = field(default_factory=dict)  # perceived id -> gt id
    ious: dict[str, float] = field(default_factory=dict)  # perceived id -> matched IoU
    ambiguous: list[str] = field(default_factory=list)  # perceived ids
    unmatched_perceived: list[str] = field(default_factory=list)
    unmatched_gt: list[str] = field(default_factory=list)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class EvalOracle:
    def __init__(self, world: SimWorld) -> None:
        self._world = world

    # ---------------------------------------------------------- ground truth state

    def object_pos(self, gt_id: str) -> np.ndarray:
        return self._world.body_pos(gt_id)

    def object_speed(self, gt_id: str) -> float:
        return float(np.linalg.norm(self._world.body_linvel(gt_id)))

    def held_gt_id(self) -> Optional[str]:
        return self._world.attached_body_name()

    def active_objects(self) -> list[str]:
        return self._world.active_objects()

    def region_bounds(self, region: str = "red_region") -> RegionBounds:
        return self._world.region_bounds(region)

    def object_in_region(self, gt_id: str, region: str = "red_region") -> bool:
        b = self.region_bounds(region)
        p = self.object_pos(gt_id)
        return (
            abs(p[0] - b.center_xy[0]) <= b.half_extents_xy[0]
            and abs(p[1] - b.center_xy[1]) <= b.half_extents_xy[1]
            and (b.support_z - 0.005) <= p[2] <= (b.support_z + REGION_MAX_HEIGHT)
        )

    # ---------------------------------------------------------- visibility / gt bboxes

    def gt_bboxes(self, camera: str = "onboard") -> dict[str, tuple[float, float, float, float]]:
        """Pixel bboxes (x1, y1, x2, y2) of currently visible ground-truth
        objects and the red region, from a segmentation render of the
        CURRENT simulation state.  Visibility = >= MIN_VISIBLE_PIXELS
        unoccluded pixels."""
        seg = self._world.render_segmentation(camera)
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        geom_ids = seg[..., 0]
        model = self._world.model
        out: dict[str, tuple[float, float, float, float]] = {}
        names = list(MANIPULABLE_BODIES) + ["red_region"]
        for name in names:
            mask = np.zeros(geom_ids.shape, dtype=bool)
            body_id = model.body(name).id
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    mask |= is_geom & (geom_ids == geom_id)
            # The segmentation renderer emits stray mislabeled single pixels
            # along silhouette edges of other objects; keep only pixels with
            # at least 4 of 8 neighbors in the mask (cheap denoise).
            m = mask.astype(np.int8)
            neighbors = (
                np.roll(m, 1, 0) + np.roll(m, -1, 0) + np.roll(m, 1, 1) + np.roll(m, -1, 1)
                + np.roll(np.roll(m, 1, 0), 1, 1) + np.roll(np.roll(m, 1, 0), -1, 1)
                + np.roll(np.roll(m, -1, 0), 1, 1) + np.roll(np.roll(m, -1, 0), -1, 1)
            )
            clean = mask & (neighbors >= 4)
            count = int(clean.sum())
            if count >= MIN_VISIBLE_PIXELS:
                ys, xs = np.nonzero(clean)
                out[name] = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        return out

    def visible_objects(self, camera: str = "onboard") -> list[str]:
        return [n for n in self.gt_bboxes(camera) if n != "red_region"]

    # ---------------------------------------------------------- association

    def associate(self, scene: SceneDescription, camera: str = "onboard") -> AssociationResult:
        """Associate a SceneDescription's perceived objects with ground
        truth via projected-bbox IoU.  Must be called while the simulation
        still holds the state the scene was perceived from (guarded via
        sim_time)."""
        if abs(self._world.sim_time - scene.sim_time) > 1e-9:
            raise ValueError(
                f"associate() requires the perception-time sim state "
                f"(scene sim_time {scene.sim_time}, world {self._world.sim_time})"
            )
        gt = {k: v for k, v in self.gt_bboxes(camera).items() if k != "red_region"}
        result = AssociationResult(unmatched_gt=list(gt.keys()))

        perceived = [g for g in scene.objects if g.bbox_xyxy is not None]
        # Ambiguity detection per perceived box.
        candidate_ious: dict[str, list[tuple[float, str]]] = {}
        for g in perceived:
            pairs = sorted(
                ((_bbox_iou(g.bbox_xyxy, box), gt_id) for gt_id, box in gt.items()),
                reverse=True,
            )
            candidate_ious[g.instance_id] = pairs
            if (
                len(pairs) >= 2
                and pairs[0][0] >= IOU_THRESHOLD
                and pairs[1][0] >= IOU_THRESHOLD
                and (pairs[0][0] - pairs[1][0]) < AMBIGUITY_MARGIN
            ):
                result.ambiguous.append(g.instance_id)

        # Greedy one-to-one matching over unambiguous perceived boxes.
        candidates = [
            (iou, g.instance_id, gt_id)
            for g in perceived
            if g.instance_id not in result.ambiguous
            for iou, gt_id in candidate_ious[g.instance_id]
            if iou >= IOU_THRESHOLD
        ]
        candidates.sort(reverse=True)
        used_p: set[str] = set()
        used_gt: set[str] = set()
        for iou, pid, gt_id in candidates:
            if pid in used_p or gt_id in used_gt:
                continue
            result.matches[pid] = gt_id
            result.ious[pid] = iou
            used_p.add(pid)
            used_gt.add(gt_id)
        result.unmatched_perceived = [
            g.instance_id for g in perceived
            if g.instance_id not in used_p and g.instance_id not in result.ambiguous
        ]
        result.unmatched_gt = [gt_id for gt_id in gt if gt_id not in used_gt]
        return result

    # ---------------------------------------------------------- stability

    def check_stability(
        self,
        gt_id: str,
        duration_s: float = STABILITY_DURATION_S,
        sample_dt: float = STABILITY_SAMPLE_DT,
        pos_tol: float = STABILITY_POS_TOL,
        vel_tol: float = STABILITY_VEL_TOL,
    ) -> tuple[bool, str]:
        """Advance the simulation ``duration_s`` seconds, sampling every
        ``sample_dt``; the object must satisfy BOTH tolerances at EVERY
        sample (the condition is maintained over the interval, not just at
        an endpoint).  Intended for end-of-trial evaluation (it steps the
        world)."""
        steps_per_sample = max(1, int(round(sample_dt / self._world.timestep)))
        n_samples = max(1, int(round(duration_s / sample_dt)))
        ref = self.object_pos(gt_id)
        for k in range(n_samples):
            self._world.step(steps_per_sample)
            drift = float(np.linalg.norm(self.object_pos(gt_id) - ref))
            speed = self.object_speed(gt_id)
            if drift > pos_tol:
                return False, f"drift {drift:.3f} m > {pos_tol} at sample {k + 1}"
            if speed > vel_tol:
                return False, f"speed {speed:.3f} m/s > {vel_tol} at sample {k + 1}"
        return True, f"stable for {duration_s}s ({n_samples} samples)"
