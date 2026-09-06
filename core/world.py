# Owner: backbone (ALL)
"""SimWorld: owns the MuJoCo model/data for the Unitree G1 tabletop scene.

SimWorld is internal backbone infrastructure.  Student modules never touch
it: the executor sees only RobotEnv (core.env), evaluation sees only
EvalOracle (core.oracle).  Both wrap one injected SimWorld and may share it.

Robot control is delegated to core.g1.G1Controller (sliding mocap-welded
base, position-controlled upper body, right-arm IK).  ``step`` is the ONLY
place where control and physics advance: every physics step is preceded by
one controller update.  Target setters never step.

Attachment ("grasping") is weld-based: the model predefines one inactive
weld equality per manipulable body between the right end-effector body and
the object; attaching activates exactly one weld with the object's CURRENT
pose relative to that body, so the object is never snapped.  Ordinary
stepping never teleports objects; only privileged mock code
(TeleportExecutor) may call the explicit ``teleport_*`` methods.

Rendering: all captures (single or multi-camera) happen under one lock,
from one unchanged simulation state, with owned array copies.
"""

from __future__ import annotations

import pathlib
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.g1 import (
    CAMERA_ALIASES,
    CAMERA_HEAD,
    CAMERAS,
    RIGHT_EE_BODY,
    G1Controller,
    G1Limits,
    yaw_quat,
)
from core.ik import IKConfig, IKResult
from core.rendering import mujoco
from core.types import IMAGE_HEIGHT, IMAGE_WIDTH, SceneConfig

ASSETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"
SCENE_XML = ASSETS_DIR / "scene.xml"
SCENE_CAM_SYNC_XML = ASSETS_DIR / "scene_cam_sync.xml"  # diagnostic variant (+ falling ball)

MANIPULABLE_BODIES = ("stone", "stone2", "cube", "bottle")
_WELD_FOR_BODY = {name: f"grasp_{name}" for name in MANIPULABLE_BODIES}
_PARK_POS = (10.0, 10.0)  # parked (unused) objects go far outside the workspace

ATTACH_RADIUS = 0.05  # meters; distance predicate for try_attach_near_ee (EE site to object center)
MAX_ATTACHMENTS = 1

_INVALID_DEPTH_FRACTION = 0.98  # depth >= this fraction of zfar -> NaN

DEFAULT_CAMERA = CAMERA_HEAD


@dataclass(frozen=True)
class RegionBounds:
    """Finite bounds of a support region."""

    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    support_z: float


@dataclass
class CameraCapture:
    """Raw multi-camera capture of ONE simulation state (internal)."""

    sim_time: float
    frames: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]  # name -> rgb, depth, K, T


def resolve_camera(name: str) -> str:
    """Map public/alias camera names to MuJoCo camera names."""
    return CAMERA_ALIASES.get(name, name)


class SimWorld:
    def __init__(self, xml_path: pathlib.Path = SCENE_XML, limits: Optional[G1Limits] = None,
                 ik_config: Optional[IKConfig] = None) -> None:
        self.xml_path = pathlib.Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()  # guards stepping vs. capture
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
        self._eq_id = {self.model.equality(i).name: i for i in range(self.model.neq)}
        self._grasp_eq_ids = {self._eq_id[_WELD_FOR_BODY[n]] for n in MANIPULABLE_BODIES}
        self.robot = G1Controller(self.model, self.data, limits=limits, ik_config=ik_config)
        self.robot.reset(0.0, 0.0, 0.0)
        self._park_all_objects()
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ setup

    def _park_all_objects(self) -> None:
        for i, name in enumerate(MANIPULABLE_BODIES):
            adr = self._joint_qposadr[f"{name}_free"]
            self.data.qpos[adr:adr + 3] = (_PARK_POS[0] + 0.5 * i, _PARK_POS[1], 0.05)
            self.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)

    def reset(self, config: SceneConfig) -> None:
        """Restore a reproducible episode state: poses, velocities, controls,
        sim time, attachments, and episode-local bookkeeping."""
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            # grasp welds inactive; the base weld stays active
            for eq in self._grasp_eq_ids:
                self.data.eq_active[eq] = 0
            self._attached_body = None
            self._attach_handles.clear()
            self._attach_counter = 0
            self._episode += 1
            self.config = config

            init = {"x": 0.0, "y": 0.0, "yaw": 0.0, **(config.robot_init or {})}
            self.robot.reset(init["x"], init["y"], init["yaw"])

            placed = {}
            for spec in config.objects:
                if spec.name not in _WELD_FOR_BODY:
                    raise ValueError(f"unknown manipulable object {spec.name!r} in SceneConfig")
                placed[spec.name] = spec
            self._active_objects = [s.name for s in config.objects]
            self._park_all_objects()
            for name, spec in placed.items():
                adr = self._joint_qposadr[f"{name}_free"]
                self.data.qpos[adr:adr + 3] = np.asarray(spec.pos, dtype=float)
                self.data.qpos[adr + 3:adr + 7] = yaw_quat(spec.yaw)

            self.data.qvel[:] = 0.0
            self.data.time = 0.0
            mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ stepping

    def step(self, n: int = 1) -> None:
        """Advance control AND physics by ``n`` steps.  This is the only
        path that moves time forward."""
        with self.lock:
            for _ in range(n):
                self.robot.update()
                mujoco.mj_step(self.model, self.data)

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def episode(self) -> int:
        return self._episode

    # ------------------------------------------------------------------ robot state / control

    def base_pose(self) -> np.ndarray:
        return self.robot.base_pose()

    def ee_pos(self) -> np.ndarray:
        self._refresh_kinematics()
        return self.robot.ee_pos()

    def ee_rot(self) -> np.ndarray:
        self._refresh_kinematics()
        return self.robot.ee_rot()

    def set_base_target(self, x: float, y: float, yaw: float) -> None:
        self.robot.set_base_target(x, y, yaw)

    def set_arm_target(self, pos_world: np.ndarray, rot_world: Optional[np.ndarray] = None) -> IKResult:
        """Non-blocking IK command for the right arm (see G1Controller)."""
        with self.lock:
            self._refresh_kinematics()
            return self.robot.set_arm_target(pos_world, rot_world)

    def set_waist_target(self, yaw: float, roll: float, pitch: float) -> None:
        self.robot.set_waist_target(yaw, roll, pitch)

    def _refresh_kinematics(self) -> None:
        """Make derived kinematic quantities (site/cam/geom poses) consistent
        with the CURRENT qpos.  After mj_step they lag one integration step;
        this recomputes them without touching qpos/qvel/time/act."""
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    # ------------------------------------------------------------------ attachment

    def try_attach_near_ee(self) -> Optional[str]:
        """Attach the nearest eligible manipulable free body within
        ATTACH_RADIUS of the end-effector site, at its CURRENT relative
        pose.  Returns an opaque handle string, or None (nothing in range,
        or the single supported attachment is already in use).  Ties are
        resolved deterministically by (distance, body id).  Selection uses
        actual simulated geometry only — never perceived IDs."""
        with self.lock:
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
        ee = self.data.body(RIGHT_EE_BODY)
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
        with self.lock:
            if not self._attach_handles:
                return False
            for name in self._attach_handles.values():
                self.data.eq_active[self._eq_id[_WELD_FOR_BODY[name]]] = 0
            self._attach_handles.clear()
            self._attached_body = None
            return True

    def is_attached(self) -> bool:
        return bool(self._attach_handles)

    def grasp_weld_active(self) -> bool:
        """True when any grasp weld equality is active in MjData (physical
        attachment state, independent of the handle bookkeeping)."""
        return any(bool(self.data.eq_active[eq]) for eq in self._grasp_eq_ids)

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
        with self.lock:
            adr = self._joint_qposadr[f"{name}_free"]
            self.data.qpos[adr:adr + 3] = np.asarray(pos, dtype=float)
            self.data.qpos[adr + 3:adr + 7] = yaw_quat(yaw)
            dofadr = self.model.joint(f"{name}_free").dofadr[0]
            self.data.qvel[dofadr:dofadr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def teleport_base(self, x: float, y: float, yaw: float) -> None:
        """PRIVILEGED (TeleportExecutor): instantaneous base placement,
        carrying any held object along with the end effector."""
        with self.lock:
            held = self._attached_body
            offset = self.body_pos(held) - self.ee_pos() if held is not None else None
            self.robot.teleport_base(x, y, yaw)
            mujoco.mj_forward(self.model, self.data)
            if held is not None and offset is not None:
                self.teleport_body(held, self.ee_pos() + offset)

    def teleport_arm_to(self, pos_world: np.ndarray) -> bool:
        """PRIVILEGED (TeleportExecutor): solve IK (default orientation
        policy, position-only fallback) and place the right arm there
        instantly, carrying any held object.  False when unreachable."""
        with self.lock:
            self._refresh_kinematics()
            res, _policy = self.robot.solve_arm_target(pos_world)
            if not res.success:
                return False
            held = self._attached_body
            offset = self.body_pos(held) - self.ee_pos() if held is not None else None
            self.robot.teleport_arm(res.q)
            mujoco.mj_forward(self.model, self.data)
            if held is not None and offset is not None:
                self.teleport_body(held, self.ee_pos() + offset)
            return True

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

    def robot_contacts(self, penetration: float = 1e-4) -> list[tuple[str, str, float]]:
        """PRIVILEGED diagnostics: penetrating contacts involving any robot
        body (self-collision or robot/environment), as (body1, body2, dist)."""
        mujoco.mj_forward(self.model, self.data)
        out = []
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if con.dist >= -penetration:
                continue
            b1 = self.model.body(self.model.geom_bodyid[con.geom1]).name
            b2 = self.model.body(self.model.geom_bodyid[con.geom2]).name
            if self._is_robot_body(b1) or self._is_robot_body(b2):
                out.append((b1, b2, float(con.dist)))
        return out

    def robot_contacts_for_arm_q(self, q: np.ndarray, penetration: float = 1e-4) -> list[tuple[str, str, float]]:
        """PRIVILEGED diagnostics: penetrating contacts the robot WOULD have
        with the right arm at joint values ``q`` (scratch copy; live state
        untouched)."""
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.robot.ik.qpos_ids] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self.model, scratch)
        out = []
        for c in range(scratch.ncon):
            con = scratch.contact[c]
            if con.dist >= -penetration:
                continue
            b1 = self.model.body(self.model.geom_bodyid[con.geom1]).name
            b2 = self.model.body(self.model.geom_bodyid[con.geom2]).name
            if self._is_robot_body(b1) or self._is_robot_body(b2):
                out.append((b1, b2, float(con.dist)))
        return out

    def _is_robot_body(self, name: str) -> bool:
        return name not in ("world", "table", "red_region", "diag_ball", *MANIPULABLE_BODIES)

    # ------------------------------------------------------------------ rendering

    def _get_renderer(self) -> "mujoco.Renderer":
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        return self._renderer

    def capture(self, cameras: tuple[str, ...] | list[str] = (DEFAULT_CAMERA,)) -> CameraCapture:
        """Capture RGB, depth, K and T_world_camera for every requested
        camera from ONE unchanged simulation state: the lock is held for
        the whole batch, nothing steps, no controller update runs, and all
        arrays are owned copies.  Depth is metric optical-axis z in meters
        (the installed mujoco Renderer already converts the z-buffer; see
        docs/DECISIONS.md), NaN for background/invalid samples."""
        names = [resolve_camera(c) for c in cameras]
        for n in names:
            if n not in CAMERAS and n != "overhead":
                raise ValueError(f"unknown camera {n!r}; known: {CAMERAS + ('overhead',)} (+ alias 'onboard')")
        with self.lock:
            self._refresh_kinematics()
            r = self._get_renderer()
            zfar = float(self.model.vis.map.zfar * self.model.stat.extent)
            frames = {}
            for cam in names:
                r.update_scene(self.data, camera=cam)
                rgb = np.array(r.render(), dtype=np.uint8, copy=True)
                r.enable_depth_rendering()
                try:
                    r.update_scene(self.data, camera=cam)
                    depth = np.array(r.render(), dtype=np.float32, copy=True)
                finally:
                    r.disable_depth_rendering()
                depth[depth >= _INVALID_DEPTH_FRACTION * zfar] = np.nan
                frames[cam] = (rgb, depth, self.camera_intrinsics(cam), self.camera_extrinsics(cam))
            return CameraCapture(sim_time=self.sim_time, frames=frames)

    def render_rgbd(self, camera: str = DEFAULT_CAMERA) -> tuple[np.ndarray, np.ndarray]:
        """Aligned (rgb, depth) of one camera from the same state (see capture)."""
        cap = self.capture((camera,))
        rgb, depth, _k, _t = cap.frames[resolve_camera(camera)]
        return rgb, depth

    def render_segmentation(self, camera: str = DEFAULT_CAMERA) -> np.ndarray:
        """(H, W, 2) int32 segmentation: [..., 0] object id, [..., 1] object
        type.  PRIVILEGED: used by EvalOracle/mocks for visibility and
        ground-truth bboxes; never exposed through RobotEnv."""
        cam = resolve_camera(camera)
        with self.lock:
            self._refresh_kinematics()
            r = self._get_renderer()
            r.enable_segmentation_rendering()
            try:
                r.update_scene(self.data, camera=cam)
                seg = np.array(r.render(), copy=True)
            finally:
                r.disable_segmentation_rendering()
            return seg

    def camera_intrinsics(self, camera: str = DEFAULT_CAMERA) -> np.ndarray:
        """Pinhole K for the public camera frame (+x right, +y down, +z
        forward), pixel centers at integer coordinates."""
        fovy = float(self.model.camera(resolve_camera(camera)).fovy[0])
        f = (IMAGE_HEIGHT / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        return np.array(
            [
                [f, 0.0, IMAGE_WIDTH / 2.0 - 0.5],
                [0.0, f, IMAGE_HEIGHT / 2.0 - 0.5],
                [0.0, 0.0, 1.0],
            ]
        )

    def camera_extrinsics(self, camera: str = DEFAULT_CAMERA) -> np.ndarray:
        """T_world_camera for the public camera frame: p_world = T @ [p_cam, 1].
        MuJoCo's camera frame is (+x right, +y up, -z forward); the public
        convention is (+x right, +y down, +z forward), i.e. a 180-degree
        rotation about x.  Uses the current kinematic state (call after
        _refresh_kinematics or inside capture)."""
        cam_id = self.model.camera(resolve_camera(camera)).id
        r_gl = np.array(self.data.cam_xmat[cam_id]).reshape(3, 3)
        pos = np.array(self.data.cam_xpos[cam_id])
        r_cv = r_gl @ np.diag([1.0, -1.0, -1.0])
        t = np.eye(4)
        t[:3, :3] = r_cv
        t[:3, 3] = pos
        return t

    def effective_clip_distances(self) -> tuple[float, float]:
        """(znear, zfar) in meters actually used by the renderer: MJCF
        map.znear/zfar are FRACTIONS of the compiled model extent."""
        ext = float(self.model.stat.extent)
        return float(self.model.vis.map.znear * ext), float(self.model.vis.map.zfar * ext)

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
