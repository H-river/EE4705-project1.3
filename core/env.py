# Owner: backbone (ALL)
"""RobotEnv: the PUBLIC environment surface for executor code.

Wraps an injected SimWorld but exposes no ground truth: no object names,
no ground-truth IDs, no object poses, no oracle access, and no MjModel /
MjData handles.  Attachment returns an opaque handle only.

Every capture is an owned-copy Observation; if an ObservationStore is
provided, captures are recorded there automatically.

Cameras (see assets/README.md):
* ``head``        — describe, ground, SEARCH and final verification (the
                    default; ``"onboard"`` is an accepted alias).
* ``left_wrist``, ``right_wrist`` — available for future Executor
                    alignment and grasp/place checks; not used by the
                    current mock workflow.

Control contract (unchanged from the pre-G1 backbone): ``set_base_target``
and ``set_arm_target`` only update targets and return immediately; motion
happens exclusively inside ``step``.  ``set_arm_target`` is position-only;
its orientation policy is documented in core.g1.G1Controller
(default tilted top-down approach, relaxed to position-only when that
orientation is unreachable).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.obs_store import ObservationStore
from core.types import Observation
from core.world import DEFAULT_CAMERA, SimWorld, resolve_camera


class RobotEnv:
    def __init__(self, world: SimWorld, store: Optional[ObservationStore] = None) -> None:
        self._world = world
        self._store = store
        self._frame_counter = 0
        self._capture_counter = 0

    # -------------------------------------------------- observation

    def get_obs(self, camera: str = "onboard") -> Observation:
        """One RGB-D observation (``"onboard"`` == head camera)."""
        return self.get_obs_multi([camera])[resolve_camera(camera)]

    def get_obs_multi(self, camera_ids: list[str]) -> dict[str, Observation]:
        """Atomic multi-camera capture: every requested camera is rendered
        (RGB and depth) from ONE unchanged simulation state, with its own
        calibration from that same state.  All observations share
        ``sim_time`` and a common capture identifier
        ``ep<episode>_capture<n>_<camera>``; the capture counter increments
        on every call, so repeated captures without stepping still get
        unique IDs.  Keys are the canonical camera names."""
        if not camera_ids:
            raise ValueError("camera_ids must not be empty")
        names = [resolve_camera(c) for c in camera_ids]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate camera in request: {camera_ids}")
        cap = self._world.capture(tuple(names))
        capture_index = self._capture_counter
        self._capture_counter += 1
        base_id = f"ep{self._world.episode:02d}_capture{capture_index:04d}"
        out: dict[str, Observation] = {}
        for cam in names:
            rgb, depth, k, t = cap.frames[cam]
            obs = Observation(
                frame_id=self._frame_counter,
                rgb=rgb,
                depth=depth,
                intrinsics=k,
                t_world_camera=t,
                sim_time=cap.sim_time,
                camera_name=cam,
                capture_id=f"{base_id}_{cam}",
            )
            self._frame_counter += 1
            if self._store is not None:
                self._store.put(obs)
            out[cam] = obs
        return out

    # -------------------------------------------------- proprioception

    def get_ee_pos(self) -> np.ndarray:
        return self._world.ee_pos()

    def get_base_pose(self) -> np.ndarray:
        return self._world.base_pose()

    def sim_time(self) -> float:
        return self._world.sim_time

    def timestep(self) -> float:
        return self._world.timestep

    # -------------------------------------------------- control

    def set_base_target(self, x: float, y: float, yaw: float) -> None:
        """Non-blocking planar base target (x, y, yaw); the base slides
        toward it with speed limits during ``step``."""
        self._world.set_base_target(float(x), float(y), float(yaw))

    def set_arm_target(self, pos_world: np.ndarray) -> None:
        """Non-blocking: command the right arm so the end effector converges
        toward ``pos_world`` (world frame, meters) during ``step``.  Raises
        ValueError when no IK solution exists for the current base pose."""
        p = np.asarray(pos_world, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError(f"pos_world must be a finite 3-vector, got {pos_world!r}")
        self._world.set_arm_target(p)

    def step(self, n: int = 1) -> None:
        self._world.step(n)

    # -------------------------------------------------- attachment

    def try_attach_near_ee(self) -> Optional[str]:
        """Attempt to attach the nearest eligible object within the
        configured distance of the end effector.  Returns an OPAQUE handle
        (no ground-truth identity), or None."""
        return self._world.try_attach_near_ee()

    def detach(self) -> bool:
        return self._world.detach()

    def is_attached(self) -> bool:
        return self._world.is_attached()

    # -------------------------------------------------- lifecycle

    def close(self) -> None:
        self._world.close()


__all__ = ["RobotEnv", "DEFAULT_CAMERA"]
