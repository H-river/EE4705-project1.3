# Owner: Student B
"""Episode memory of where A last LOCALIZED each perceived instance.

B owns cross-frame memory (A reports only the current frame; C keeps no
memory between actions).  Memory never changes what A reported: B uses a
recalled position only to plan motion toward a goal member that is out of
view now, and says so in its audit.  Held and released objects are never
recalled: their positions changed because the robot moved them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from core.types import GroundStatus, SceneDescription


@dataclass
class MemoryEntry:
    instance_id: str
    name: str
    kind: str
    pos_world: tuple[float, float, float]
    frame_id: int
    base_pose: Optional[tuple[float, float, float]]
    attributes: dict
    region_half_extents_xy: Optional[tuple[float, float]] = None
    held: bool = False
    released: bool = False


class EpisodeMemory:
    def __init__(self):
        self.entries: dict[str, MemoryEntry] = {}
        self.last_frame_id = -1

    def reset(self):
        self.entries.clear()
        self.last_frame_id = -1

    def update(self, scene: SceneDescription, base_pose=None, held_instance_id=None,
               last_release_instance_id=None):
        """Record every LOCALIZED instance of ``scene``; flag held/released ones."""
        if scene.frame_id >= 0:
            self.last_frame_id = max(self.last_frame_id, scene.frame_id)
        pose = tuple(float(v) for v in base_pose) if base_pose is not None else None
        for g in scene.objects + scene.regions:
            if g.status is not GroundStatus.LOCALIZED or g.pos_world is None:
                continue
            frame = g.frame_id if g.frame_id >= 0 else scene.frame_id
            self.entries[g.instance_id] = MemoryEntry(
                g.instance_id, g.name, g.kind, tuple(g.pos_world), frame, pose,
                {k: v for k, v in g.attributes.items() if k == "color"}, g.region_half_extents_xy)
        for ident, flag in ((held_instance_id, "held"), (last_release_instance_id, "released")):
            if ident and ident in self.entries:
                setattr(self.entries[ident], flag, True)

    def entry(self, instance_id) -> Optional[MemoryEntry]:
        return self.entries.get(instance_id)

    def age(self, instance_id, frame_id=None) -> Optional[int]:
        e = self.entries.get(instance_id)
        if e is None:
            return None
        now = self.last_frame_id if frame_id is None else frame_id
        return max(0, now - e.frame_id)

    def recall(self, instance_id, max_age_frames=40, max_base_move_m=0.5, *,
               frame_id=None, base_pose=None):
        """Last LOCALIZED pos_world, or None when unknown, too old, taken
        from a base pose more than ``max_base_move_m`` away, or held/released."""
        e = self.entries.get(instance_id)
        if e is None or e.held or e.released:
            return None
        if self.age(instance_id, frame_id) > max_age_frames:
            return None
        if base_pose is not None and e.base_pose is not None:
            if math.dist(e.base_pose[:2], tuple(base_pose)[:2]) > max_base_move_m:
                return None
        return e.pos_world
