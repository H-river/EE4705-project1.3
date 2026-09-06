# Owner: backbone (ALL)
"""SimWorld: owns the MuJoCo model/data for the simplified platform.

SimWorld is internal backbone infrastructure.  Student modules never touch
it: the executor sees only RobotEnv (core.env), evaluation sees only
EvalOracle (core.oracle).  Both wrap one injected SimWorld and may share it.

Attachment ("grasping") is weld-based: the model predefines one inactive
weld equality per manipulable body; attaching activates exactly one weld
with the object's CURRENT pose relative to the end-effector body, so the
object is never snapped.  Ordinary stepping never teleports objects; only
privileged mock code (TeleportExecutor) may call the explicit
``teleport_body`` method.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.rendering import mujoco
from core.types import IMAGE_HEIGHT, IMAGE_WIDTH, SceneConfig

ASSETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"
SCENE_XML = ASSETS_DIR / "scene.xml"

MANIPULABLE_BODIES = ("stone", "stone2", "cube", "bottle")
_WELD_FOR_BODY = {name: f"grasp_{name}" for name in MANIPULABLE_BODIES}
_PARK_POS = (10.0, 10.0)  # parked (unused) objects go far outside the workspace

# Home pose of the arm slides (qpos), chosen inside joint ranges and OFF to
# the side so the parked arm does not occlude the onboard camera's view.
_ARM_HOME = {"arm_x": 0.15, "arm_y": -0.45, "arm_z": 0.14}

ATTACH_RADIUS = 0.08  # meters; distance predicate for try_attach_near_ee
MAX_ATTACHMENTS = 1

_INVALID_DEPTH_FRACTION = 0.98  # depth >= this fraction of zfar -> NaN


@dataclass(frozen=True)
class RegionBounds:
    """Finite bounds of a support region."""

    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    support_z: float


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


class SimWorld:
    def __init__(self, xml_path: pathlib.Path = SCENE_XML) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._renderer: Optional[mujoco.Renderer] = None
        self._attached_body: Optional[str] = None
        self._attach_handles: dict[str, str] = {}  # opaque handle -> body name
        self._attach_counter = 0
        self._episode = 0
        self._active_objects: list[str] = []
        self.config: Optional[SceneConfig] = None
        self._joint_qposadr = {
            self.model.joint(i).name: int(self.model.jnt_qposadr[i]) for i in range(self.model.njnt)
        }
        self._eq_id = {
            self.model.equality(i).name: i for i in range(self.model.neq)
        }
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ setup

    def reset(self, config: SceneConfig) -> None:
        """Restore a reproducible episode state: poses, velocities, controls,
        sim time, attachments, and episode-local bookkeeping."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.eq_active[:] = 0
        self._attached_body = None
        self._attach_handles.clear()
        self._attach_counter = 0
        self._episode += 1
        self.config = config

        # Robot base + arm home.
        init = {"x": 0.0, "y": 0.0, "yaw": 0.0, **(config.robot_init or {})}
        self._set_joint("base_x", init["x"])
        self._set_joint("base_y", init["y"])
        self._set_joint("base_yaw", init["yaw"])
        for joint, q in _ARM_HOME.items():
            self._set_joint(joint, q)

        # Objects: configured ones at their spec pose, the rest parked.
        placed = {}
        for spec in config.objects:
            if spec.name not in _WELD_FOR_BODY:
                raise ValueError(f"unknown manipulable object {spec.name!r} in SceneConfig")
            placed[spec.name] = spec
        self._active_objects = [s.name for s in config.objects]
        for i, name in enumerate(MANIPULABLE_BODIES):
            adr = self._joint_qposadr[f"{name}_free"]
            if name in placed:
                spec = placed[name]
                pos = np.asarray(spec.pos, dtype=float)
                quat = _yaw_quat(spec.yaw)
            else:
                pos = np.array([_PARK_POS[0] + 0.5 * i, _PARK_POS[1], 0.05])
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            self.data.qpos[adr : adr + 3] = pos
            self.data.qpos[adr + 3 : adr + 7] = quat

        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        # Hold current joint targets (actuator order matches joint order here).
        self.data.ctrl[:] = [
            self._get_joint(j) for j in ("base_x", "base_y", "base_yaw", "arm_x", "arm_y", "arm_z")
        ]
        mujoco.mj_forward(self.model, self.data)

    def _set_joint(self, name: str, value: float) -> None:
        self.data.qpos[self._joint_qposadr[name]] = value

    def _get_joint(self, name: str) -> float:
        return float(self.data.qpos[self._joint_qposadr[name]])

    # ------------------------------------------------------------------ stepping

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    # ------------------------------------------------------------------ robot state

    def base_pose(self) -> np.ndarray:
        return np.array([self._get_joint("base_x"), self._get_joint("base_y"), self._get_joint("base_yaw")])

    def ee_pos(self) -> np.ndarray:
        return np.array(self.data.site("ee_site").xpos, dtype=float)

    def set_ctrl(self, base_x: float, base_y: float, base_yaw: float,
                 arm_x: float, arm_y: float, arm_z: float) -> None:
        self.data.ctrl[:] = [base_x, base_y, base_yaw, arm_x, arm_y, arm_z]

    def get_ctrl(self) -> np.ndarray:
        return np.array(self.data.ctrl, dtype=float)

    def joint_ranges(self) -> dict[str, tuple[float, float]]:
        out = {}
        for name in ("base_x", "base_y", "base_yaw", "arm_x", "arm_y", "arm_z"):
            j = self.model.joint(name)
            out[name] = (float(j.range[0]), float(j.range[1]))
        return out

    # ------------------------------------------------------------------ attachment

    def try_attach_near_ee(self) -> Optional[str]:
        """Attach the nearest eligible manipulable free body within
        ATTACH_RADIUS of the end-effector site, at its CURRENT relative
        pose.  Returns an opaque handle string, or None (nothing in range,
        or the single supported attachment is already in use).  Ties are
        resolved deterministically by (distance, body id).  Selection uses
        actual simulated geometry only — never perceived IDs."""
        if len(self._attach_handles) >= MAX_ATTACHMENTS:
            return None
        ee = self.ee_pos()
        candidates: list[tuple[float, int, str]] = []
        for name in self._active_objects:
            body = self.data.body(name)
            dist = float(np.linalg.norm(np.array(body.xpos) - ee))
            if dist <= ATTACH_RADIUS:
                candidates.append((dist, int(self.model.body(name).id), name))
        if not candidates:
            return None
        candidates.sort()
        name = candidates[0][2]
        self._activate_weld(name)
        self._attach_counter += 1
        handle = f"att_{self._episode}_{self._attach_counter}"
        self._attach_handles[handle] = name
        self._attached_body = name
        return handle

    def _activate_weld(self, body_name: str) -> None:
        eq = self._eq_id[_WELD_FOR_BODY[body_name]]
        ee = self.data.body("ee_body")
        obj = self.data.body(body_name)
        r1 = np.array(ee.xmat).reshape(3, 3)
        r2 = np.array(obj.xmat).reshape(3, 3)
        relpos = r1.T @ (np.array(obj.xpos) - np.array(ee.xpos))
        relmat = r1.T @ r2
        relquat = np.empty(4)
        mujoco.mju_mat2Quat(relquat, relmat.reshape(9))
        data = np.zeros(self.model.eq_data.shape[1])
        data[0:3] = 0.0  # anchor
        data[3:6] = relpos
        data[6:10] = relquat
        data[10] = 1.0  # torquescale
        self.model.eq_data[eq, :] = data
        self.data.eq_active[eq] = 1

    def detach(self) -> bool:
        if not self._attach_handles:
            return False
        for name in self._attach_handles.values():
            self.data.eq_active[self._eq_id[_WELD_FOR_BODY[name]]] = 0
        self._attach_handles.clear()
        self._attached_body = None
        return True

    def is_attached(self) -> bool:
        return bool(self._attach_handles)

    def attached_body_name(self) -> Optional[str]:
        """Ground-truth identity of the attached body.  PRIVILEGED: for
        EvalOracle and mocks only; never exposed through RobotEnv."""
        return self._attached_body

    # ------------------------------------------------------------------ privileged state access (oracle / mocks)

    def body_pos(self, name: str) -> np.ndarray:
        return np.array(self.data.body(name).xpos, dtype=float)

    def body_linvel(self, name: str) -> np.ndarray:
        body_id = self.model.body(name).id
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
        return vel[3:6].copy()  # linear part (world frame)

    def teleport_body(self, name: str, pos: np.ndarray, yaw: float = 0.0) -> None:
        """PRIVILEGED: instantaneous re-placement, for TeleportExecutor and
        trial setup only.  Never called by RobotEnv paths."""
        adr = self._joint_qposadr[f"{name}_free"]
        self.data.qpos[adr : adr + 3] = np.asarray(pos, dtype=float)
        self.data.qpos[adr + 3 : adr + 7] = _yaw_quat(yaw)
        dofadr = self.model.joint(f"{name}_free").dofadr[0]
        self.data.qvel[dofadr : dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def active_objects(self) -> list[str]:
        return list(self._active_objects)

    def region_bounds(self, name: str = "red_region") -> RegionBounds:
        geom = self.model.geom(f"{name}_geom")
        body = self.model.body(name)
        center = np.array(body.pos) + np.array(geom.pos)
        half = np.array(geom.size)
        return RegionBounds(
            center_xy=(float(center[0]), float(center[1])),
            half_extents_xy=(float(half[0]), float(half[1])),
            support_z=float(center[2] - half[2]),
        )

    # ------------------------------------------------------------------ rendering

    def _get_renderer(self) -> "mujoco.Renderer":
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        return self._renderer

    def render_rgbd(self, camera: str = "onboard") -> tuple[np.ndarray, np.ndarray]:
        """Aligned (rgb, depth) captured from the SAME simulation state: the
        two render passes share one MjData with no stepping in between.
        Returns owned copies (no aliasing of renderer buffers).  Depth is
        metric optical-axis z in meters (the installed mujoco Renderer
        already converts the z-buffer; see docs/DECISIONS.md), with NaN for
        background/invalid samples."""
        r = self._get_renderer()
        r.update_scene(self.data, camera=camera)
        rgb = np.array(r.render(), dtype=np.uint8, copy=True)
        r.enable_depth_rendering()
        try:
            r.update_scene(self.data, camera=camera)
            depth = np.array(r.render(), dtype=np.float32, copy=True)
        finally:
            r.disable_depth_rendering()
        zfar = float(self.model.vis.map.zfar * self.model.stat.extent)
        depth[depth >= _INVALID_DEPTH_FRACTION * zfar] = np.nan
        return rgb, depth

    def render_segmentation(self, camera: str = "onboard") -> np.ndarray:
        """(H, W, 2) int32 segmentation: [..., 0] object id, [..., 1] object
        type.  PRIVILEGED: used by EvalOracle/mocks for visibility and
        ground-truth bboxes; never exposed through RobotEnv."""
        r = self._get_renderer()
        r.enable_segmentation_rendering()
        try:
            r.update_scene(self.data, camera=camera)
            seg = np.array(r.render(), copy=True)
        finally:
            r.disable_segmentation_rendering()
        return seg

    def camera_intrinsics(self, camera: str = "onboard") -> np.ndarray:
        """Pinhole K for the public camera frame (+x right, +y down, +z
        forward), pixel centers at integer coordinates."""
        fovy = float(self.model.camera(camera).fovy[0])
        f = (IMAGE_HEIGHT / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        return np.array(
            [
                [f, 0.0, IMAGE_WIDTH / 2.0 - 0.5],
                [0.0, f, IMAGE_HEIGHT / 2.0 - 0.5],
                [0.0, 0.0, 1.0],
            ]
        )

    def camera_extrinsics(self, camera: str = "onboard") -> np.ndarray:
        """T_world_camera for the public camera frame: p_world = T @ [p_cam, 1].
        MuJoCo's camera frame is (+x right, +y up, -z forward); the public
        convention is (+x right, +y down, +z forward), i.e. a 180-degree
        rotation about x."""
        cam_id = self.model.camera(camera).id
        r_gl = np.array(self.data.cam_xmat[cam_id]).reshape(3, 3)
        pos = np.array(self.data.cam_xpos[cam_id])
        r_cv = r_gl @ np.diag([1.0, -1.0, -1.0])
        t = np.eye(4)
        t[:3, :3] = r_cv
        t[:3, 3] = pos
        return t

    # ------------------------------------------------------------------ cleanup

    def close(self) -> None:
        """Release rendering resources.  Safe to call more than once."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "SimWorld":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
