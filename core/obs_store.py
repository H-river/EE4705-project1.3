# Owner: backbone (ALL)
"""ObservationStore: bounded in-memory window of recent observations.

Keeps the most recent CAPACITY (=32) frames.  Observations marked for
persistence are written to disk (RGB as PNG, depth as .npy, metadata as
JSON) no later than eviction time, so logs never lose frames they need.
Arrays stored here are the Observation's own copies (producers already copy
out of renderer buffers; see core.world.render_rgbd), so nothing in the
store aliases a mutable renderer buffer.
"""

from __future__ import annotations

import json
import pathlib
from collections import OrderedDict
from typing import Optional

import numpy as np
from PIL import Image

from core.types import Observation

CAPACITY = 32


def _json_safe(value: float) -> object:
    """Encode possibly non-finite floats explicitly (strict JSON has no
    NaN/Infinity literals)."""
    if isinstance(value, float) and not np.isfinite(value):
        return {"__nonfinite__": repr(value)}
    return value


def persist_observation(obs: Observation, directory: pathlib.Path) -> dict[str, str]:
    """Write one observation to ``directory`` and return the file map."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{obs.frame_id:06d}"
    rgb_path = directory / f"{stem}_rgb.png"
    depth_path = directory / f"{stem}_depth.npy"
    meta_path = directory / f"{stem}_meta.json"
    Image.fromarray(obs.rgb).save(rgb_path)
    np.save(depth_path, obs.depth)
    meta = {
        "frame_id": obs.frame_id,
        "camera_name": obs.camera_name,
        "sim_time": _json_safe(float(obs.sim_time)),
        "intrinsics": obs.intrinsics.tolist(),
        "t_world_camera": obs.t_world_camera.tolist(),
        "rgb_file": rgb_path.name,
        "depth_file": depth_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2, allow_nan=False))
    return {"rgb": str(rgb_path), "depth": str(depth_path), "meta": str(meta_path)}


class ObservationStore:
    def __init__(self, capacity: int = CAPACITY, persist_dir: Optional[pathlib.Path] = None) -> None:
        self.capacity = capacity
        self.persist_dir = pathlib.Path(persist_dir) if persist_dir is not None else None
        self._frames: OrderedDict[int, Observation] = OrderedDict()
        self._persist_marks: set[int] = set()
        self._persisted: dict[int, dict[str, str]] = {}

    def put(self, obs: Observation) -> None:
        if obs.frame_id in self._frames:
            raise ValueError(f"duplicate frame_id {obs.frame_id}")
        self._frames[obs.frame_id] = obs
        while len(self._frames) > self.capacity:
            frame_id, evicted = self._frames.popitem(last=False)
            self._flush_if_marked(frame_id, evicted)

    def get(self, frame_id: int) -> Optional[Observation]:
        return self._frames.get(frame_id)

    def latest(self) -> Optional[Observation]:
        if not self._frames:
            return None
        return next(reversed(self._frames.values()))

    def mark_persist(self, frame_id: int) -> None:
        """Ensure this frame reaches disk before (or at) eviction."""
        if frame_id not in self._frames and frame_id not in self._persisted:
            raise KeyError(f"frame {frame_id} is not in the store")
        self._persist_marks.add(frame_id)

    def _flush_if_marked(self, frame_id: int, obs: Observation) -> None:
        if frame_id in self._persist_marks and frame_id not in self._persisted:
            if self.persist_dir is None:
                raise RuntimeError("frame marked for persistence but no persist_dir configured")
            self._persisted[frame_id] = persist_observation(obs, self.persist_dir)

    def flush(self) -> dict[int, dict[str, str]]:
        """Persist all marked, still-resident frames now.  Returns the map
        of frame_id -> written files (cumulative)."""
        for frame_id in sorted(self._persist_marks):
            obs = self._frames.get(frame_id)
            if obs is not None and frame_id not in self._persisted:
                if self.persist_dir is None:
                    raise RuntimeError("frame marked for persistence but no persist_dir configured")
                self._persisted[frame_id] = persist_observation(obs, self.persist_dir)
        return dict(self._persisted)

    def clear(self) -> None:
        """Drop all frames and marks (per-episode reset).  Marked frames are
        flushed first so nothing needed by logs is lost."""
        self.flush()
        self._frames.clear()
        self._persist_marks.clear()

    def __len__(self) -> int:
        return len(self._frames)
