# Owner: backbone (ALL)
"""RobotEnv: the PUBLIC environment surface for executor code.

Wraps an injected SimWorld but exposes no ground truth: no object names,
no ground-truth IDs, no object poses, no oracle access, and no MjModel /
MjData handles.  Attachment returns an opaque handle only.

Every capture is an owned-copy Observation; if an ObservationStore is
provided, captures are recorded there automatically.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.obs_store import ObservationStore
from core.types import Observation
from core.world import SimWorld


class RobotEnv:
    def __init__(self, world: SimWorld, store: Optional[ObservationStore] = None) -> None:
        self._world = world
        self._store = store
        self._frame_counter = 0

    # -------------------------------------------------- observation

    def get_obs(self, camera: str = "onboard") -> Observation:
        rgb, depth = self._world.render_rgbd(camera)
        obs = Observation(
            frame_id=self._frame_counter,
            rgb=rgb,
            depth=depth,
            intrinsics=self._world.camera_intrinsics(camera),
            t_world_camera=self._world.camera_extrinsics(camera),
            sim_time=self._world.sim_time,
            camera_name=camera,
        )
        self._frame_counter += 1
        if self._store is not None:
            self._store.put(obs)
        return obs

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
        ctrl = self._world.get_ctrl()
        self._world.set_ctrl(x, y, yaw, ctrl[3], ctrl[4], ctrl[5])

    def set_arm_target(self, pos_world: np.ndarray) -> None:
        """Command the arm slides so the EE converges toward ``pos_world``
        (world frame, meters).  Raises ValueError when the point is outside
        the reachable envelope for the current base pose."""
        p = np.asarray(pos_world, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError(f"pos_world must be a finite 3-vector, got {pos_world!r}")
        base = self._world.base_pose()
        yaw = base[2]
        rot = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        local_xy = rot @ (p[:2] - base[:2])
        # Arm slide targets in the base frame; arm_base sits 0.80 m above base.
        targets = {"arm_x": local_xy[0], "arm_y": local_xy[1], "arm_z": p[2] - 0.80}
        ranges = self._world.joint_ranges()
        for joint, value in targets.items():
            lo, hi = ranges[joint]
            if not lo <= value <= hi:
                raise ValueError(
                    f"target {p.tolist()} unreachable: {joint}={value:.3f} outside [{lo:.2f}, {hi:.2f}]"
                )
        ctrl = self._world.get_ctrl()
        self._world.set_ctrl(ctrl[0], ctrl[1], ctrl[2], targets["arm_x"], targets["arm_y"], targets["arm_z"])

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
