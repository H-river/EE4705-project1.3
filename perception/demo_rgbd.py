# Owner: backbone demo (ALL), replaceable by Student A
"""Offline RGB-D baseline for the known colored tabletop objects.

Uses rendered RGB/depth only. Color rules and class dimensions are public
priors, not simulator segmentation, body IDs or true object positions.
This is a teaching demo, not the assignment's completed visual-AI module.
"""
from __future__ import annotations

import numpy as np

from core.interfaces import Perception
from core.types import GroundedObject, GroundStatus, Observation, SceneDescription
from core.vocab import Vocab


def _components(mask):
    """Small 4-connected components on a downsampled mask; NumPy only."""
    remaining = mask.copy()
    h, w = remaining.shape
    for y, x in zip(*np.nonzero(mask)):
        if not remaining[y, x]:
            continue
        stack, points = [(int(y), int(x))], []
        remaining[y, x] = False
        while stack:
            yy, xx = stack.pop()
            points.append((yy, xx))
            for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                if 0 <= ny < h and 0 <= nx < w and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    stack.append((ny, nx))
        if len(points) >= 16:
            yield np.asarray(points)


class RGBDPerception(Perception):
    LABEL = "A: RGB-D color/geometry rules (offline demo)"

    def __init__(self, on_scene=None):
        self.on_scene = on_scene
        self.vocab = Vocab()
        self.reset()

    def reset(self):
        self._ids = {}
        self._region_center = None

    def describe(self, obs: Observation, query=None):
        rgb = obs.rgb[::2, ::2].astype(float)
        depth = obs.depth[::2, ::2]
        yy, xx = np.indices(depth.shape)
        k, t = obs.intrinsics, obs.t_world_camera
        pc = np.stack(((2*xx-k[0, 2])*depth/k[0, 0],
                       (2*yy-k[1, 2])*depth/k[1, 1], depth), axis=-1)
        points = pc @ t[:3, :3].T + t[:3, 3]
        valid = (np.isfinite(depth) & (points[..., 2] > .848) & (points[..., 2] < 1.35)
                 & (points[..., 0] > .20) & (points[..., 0] < .75)
                 & (np.abs(points[..., 1]) < .55))
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        masks = [
            ("stone", "gray", (np.maximum.reduce([r, g, b])-np.minimum.reduce([r, g, b]) < 12) & (r > 40)),
            ("stone", "dark_red", (r > 1.7*g) & (r > 1.8*b) & (r > 55) & (points[..., 2] > .865)),
            ("cube", "blue", (b > 1.4*r) & (b > 1.25*g) & (b > 65)),
            ("bottle", "green", (g > 1.4*r) & (g > 1.25*b) & (g > 55)),
            ("red_region", "red", (r > 160) & (g < 110) & (b < 110) & (r-g > 100) & (points[..., 2] < .865)),
        ]
        objects, regions = [], []
        for name, color, mask in masks:
            candidates = []
            for pixels in _components(mask & valid):
                cloud = points[pixels[:, 0], pixels[:, 1]]
                span = np.ptp(cloud, axis=0)
                if name != "red_region" and (np.max(span) > .15 or span[2] < .008):
                    continue
                if name == "red_region" and len(pixels) < 80:
                    continue
                candidates.append((pixels, cloud))
            # The demo supports one object per class/color, including two differently colored stones.
            if not candidates:
                continue
            pixels, cloud = max(candidates, key=lambda item: len(item[0]))
            bbox = (int(pixels[:, 1].min()*2), int(pixels[:, 0].min()*2),
                    int(pixels[:, 1].max()*2+2), int(pixels[:, 0].max()*2+2))
            status, confidence = GroundStatus.LOCALIZED, .8
            center = (cloud.min(axis=0) + cloud.max(axis=0))/2
            if name == "stone":
                # Fit a known-size ellipsoid to visible surface points.
                mid = np.median(cloud, axis=0)
                v, scale = cloud-mid, np.array([.030, .025, .025])
                design = np.column_stack((2*v/(scale*scale), np.ones(len(v))))
                coef, *_ = np.linalg.lstsq(design, np.sum((v/scale)**2, axis=1), rcond=None)
                estimate = mid + coef[:3]
                if np.linalg.norm(estimate-center) < .06:
                    center = estimate
                else:
                    status, confidence = GroundStatus.UNLOCALIZED, .3
            elif name in ("cube", "bottle"):
                height = .05 if name == "cube" else .12
                top_z = np.percentile(cloud[:, 2], 99)
                top = cloud[cloud[:, 2] > top_z-.006]
                center[:2] = (top[:, :2].min(axis=0) + top[:, :2].max(axis=0))/2
                center[2] = top_z-height/2
            else:
                center[2] = np.median(cloud[:, 2])
                if np.all(np.ptp(cloud[:, :2], axis=0) > .135):
                    self._region_center = center.copy()
                elif self._region_center is not None:
                    # Visible but partly clipped/occluded static region: reuse its last full fit.
                    center = self._region_center.copy()
                    confidence = .65
                else:
                    status, confidence = GroundStatus.UNLOCALIZED, .4
            key = (name, color)
            if key not in self._ids:
                self._ids[key] = f"p{len(self._ids)}"
            is_region = name == "red_region"
            grounded = GroundedObject(
                self._ids[key], name, status, bbox_xyxy=bbox,
                pos_world=tuple(center) if status is GroundStatus.LOCALIZED else None,
                confidence=confidence, source="rgbd_demo", kind="region" if is_region else "object",
                frame_id=obs.frame_id, attributes={"color": color},
                region_half_extents_xy=(.08, .08) if is_region else None,
            )
            (regions if is_region else objects).append(grounded)
        ambiguities = [f"multiple {name} objects: specify a color" for name in {o.name for o in objects}
                       if sum(o.name == name for o in objects) > 1]
        scene = SceneDescription(objects, regions, caption="Visible: " + ", ".join(
            f"{g.attributes['color']} {g.name} ({g.instance_id})" for g in objects+regions),
            ambiguities=ambiguities, frame_id=obs.frame_id, sim_time=obs.sim_time)
        if self.on_scene:
            self.on_scene(obs, scene)
        return scene

    def ground(self, obs, target):
        scene = self.describe(obs, target)
        ref = scene.find(target)
        if ref:
            return ref
        phrase = self.vocab.find_phrase(target, "object") or self.vocab.find_phrase(target, "region")
        if phrase is None:
            return None
        name, color = phrase
        color = color or self.vocab.color_in_text(target)
        found = [g for g in scene.objects+scene.regions if g.name == name
                 and (not color or g.attributes.get("color") == color)]
        if len(found) > 1:
            return GroundedObject("ambiguous", name, GroundStatus.AMBIGUOUS,
                                  frame_id=obs.frame_id, source="rgbd_demo")
        return found[0] if found else None
