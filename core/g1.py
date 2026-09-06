# Owner: backbone (ALL)
"""Unitree G1 platform description and low-level controller.

Everything model-specific (joint names, rest posture, finger posture, base
mocap weld, gains, rate limits, IK chain, camera names) lives here or in
``assets/g1_with_hands_ee4705.xml`` — never in module code.  See
``assets/README.md`` ("Unitree G1") for the documented parameters.

Sliding-base approximation
--------------------------
The pelvis keeps its free joint and is welded to a world-child mocap body
(``base_target``).  Planar base motion (x, y, yaw) is commanded by moving
that mocap body with translational/angular speed limits, at a fixed standing
height and upright orientation.  The legs are position-held in the standing
posture and the feet are excluded from colliding with the floor.  This is
deliberately NOT a walking controller.

Control flow (non-blocking)
---------------------------
``set_*`` methods only update *targets*.  ``update()`` — called by
``SimWorld.step`` once per physics step — moves the mocap body and the
actuator setpoints toward those targets with rate limits, then MuJoCo
integrates.  Live joint positions are never overwritten to fake control.

Right-arm commands are executed as CARTESIAN setpoint streaming: the
end-effector setpoint travels in a straight line from the current EE
position to the target at ``ee_speed`` and IK is re-solved (warm-started)
along the way, so the hand follows a straight path instead of an arbitrary
joint-space arc (which swept under the table slab).  A joint-rate limit is
applied on top as a safety layer.

IK
--
``solve_arm_target`` runs damped least-squares IK (core.ik) for the right
arm on a scratch ``MjData`` copy; it never mutates live state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.ik import ChainIK, IKConfig, IKResult, interpolate_rotation, rotation_error_rad
from core.rendering import mujoco

# ----------------------------------------------------------------- names

LEG_JOINTS = [
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
RIGHT_ARM_JOINTS = [f"right_{j}_joint" for j in _ARM]
LEFT_ARM_JOINTS = [f"left_{j}_joint" for j in _ARM]
_HAND = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
RIGHT_HAND_JOINTS = [f"right_hand_{j}_joint" for j in _HAND]
LEFT_HAND_JOINTS = [f"left_hand_{j}_joint" for j in _HAND]

PELVIS_BODY = "pelvis"
BASE_MOCAP_BODY = "base_target"
BASE_WELD = "base_weld"
RIGHT_EE_SITE = "right_ee_site"
LEFT_EE_SITE = "left_ee_site"
RIGHT_EE_BODY = "right_wrist_yaw_link"  # body carrying right_ee_site (weld anchor)

# Public camera names.  "onboard" is kept as an ALIAS of the head camera so
# the pre-G1 default argument of RobotEnv.get_obs keeps working.
CAMERA_HEAD = "head"
CAMERA_LEFT_WRIST = "left_wrist"
CAMERA_RIGHT_WRIST = "right_wrist"
CAMERAS = (CAMERA_HEAD, CAMERA_LEFT_WRIST, CAMERA_RIGHT_WRIST)
CAMERA_ALIASES = {"onboard": CAMERA_HEAD}

# ----------------------------------------------------------------- posture

STAND_HEIGHT = 0.79  # m, pelvis origin above the floor (upstream "stand" keyframe)
WAIST_REST = {"waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.30}
# LEFT arm rest: upstream "stand" keyframe values, except shoulder roll
# widened from 0.2 to 0.4 rad so the hand clears the hip at rest.
LEFT_ARM_REST = {"left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.4, "left_shoulder_yaw_joint": 0.0,
                 "left_elbow_joint": 1.28, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0,
                 "left_wrist_yaw_joint": 0.0}
# RIGHT arm rest ("ready" posture): elbow bent and arm swung out to the
# right so the hand is raised beside the body (EE ~(-0.11, -0.48, 0.81) at
# base pose 0), out of the head camera's view and above the hip.  Chosen so
# joint-space paths to tabletop targets do not sweep under the table slab
# (see docs/DECISIONS.md).  Also the IK rest-posture bias.
RIGHT_ARM_REST = {"right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": -0.9, "right_shoulder_yaw_joint": 0.3,
                  "right_elbow_joint": 1.9, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0,
                  "right_wrist_yaw_joint": 0.0}
# Fixed, collision-safe semi-closed finger posture (see assets/README.md):
# index/middle curled 0.35 rad at both joints; thumb abducted and flexed to
# the side so it stays 1.4 cm BEHIND the palm reference point along the
# approach axis.  Left hand mirrors the sign convention of the model.
RIGHT_HAND_POSTURE = {
    "right_hand_thumb_0_joint": -1.0, "right_hand_thumb_1_joint": -1.0, "right_hand_thumb_2_joint": -1.0,
    "right_hand_index_0_joint": 0.35, "right_hand_index_1_joint": 0.35,
    "right_hand_middle_0_joint": 0.35, "right_hand_middle_1_joint": 0.35,
}
LEFT_HAND_POSTURE = {
    "left_hand_thumb_0_joint": 1.0, "left_hand_thumb_1_joint": 1.0, "left_hand_thumb_2_joint": 1.0,
    "left_hand_index_0_joint": -0.35, "left_hand_index_1_joint": -0.35,
    "left_hand_middle_0_joint": -0.35, "left_hand_middle_1_joint": -0.35,
}
LEG_REST = {j: 0.0 for j in LEG_JOINTS}
# IK null-space bias posture (NOT a physical target): upstream keyframe arm.
IK_REST_BIAS = {"right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.4, "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 1.28, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0}


def rest_posture() -> dict[str, float]:
    out: dict[str, float] = {}
    for d in (LEG_REST, WAIST_REST, LEFT_ARM_REST, RIGHT_ARM_REST, LEFT_HAND_POSTURE, RIGHT_HAND_POSTURE):
        out.update(d)
    return out


# ----------------------------------------------------------------- limits / gains

@dataclass(frozen=True)
class G1Limits:
    base_lin_speed: float = 0.5  # m/s, mocap target translation
    base_ang_speed: float = 1.0  # rad/s, mocap target yaw
    arm_joint_speed: float = 2.5  # rad/s, right-arm setpoint rate limit (safety layer)
    ee_speed: float = 0.25  # m/s, right-arm Cartesian setpoint streaming speed
    ee_lead: float = 0.03  # m, the streamed setpoint never runs farther than this ahead of the live EE
    waist_joint_speed: float = 0.8  # rad/s
    gravity_compensation: bool = True  # feed-forward qfrc_bias on the right-arm DoFs
    # position actuator gains (upstream: kp=500, critically damped) — kept
    # for the upper body; see assets/README.md for the rationale.


# Default end-effector orientation for POSITION-ONLY commands, expressed in
# the base (pelvis-yaw) frame: approach axis (site +z) pointing down and
# tilted APPROACH_TILT toward the finger direction; fingers (site +x) at
# FINGER_AZIMUTH left of the base heading.  Chosen from the reach sweep in
# docs/DECISIONS.md (widest collision-free band for the right arm).
APPROACH_TILT = math.radians(20.0)
FINGER_AZIMUTH = math.radians(30.0)


def approach_rotation(base_yaw: float, tilt: float = APPROACH_TILT,
                      azimuth: float = FINGER_AZIMUTH) -> np.ndarray:
    """World-frame rotation matrix of the EE site for a tilted top-down
    approach whose finger azimuth is measured from ``base_yaw``."""
    az = base_yaw + azimuth
    fdir = np.array([math.cos(az), math.sin(az), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    a = -math.cos(tilt) * up + math.sin(tilt) * fdir  # approach axis (site z)
    f = math.cos(tilt) * fdir + math.sin(tilt) * up  # finger axis (site x)
    y = np.cross(a, f)
    return np.column_stack([f, y, a])


# Where the base parks relative to a manipulation target so that the target
# lands in the right arm's collision-free workspace (base frame, meters).
ARM_WORKSPACE_OFFSET = np.array([0.36, -0.20])


def approach_base_pose(target_xy: np.ndarray, base_xy: np.ndarray) -> tuple[float, float, float]:
    """Base (x, y, yaw) that puts ``target_xy`` at ARM_WORKSPACE_OFFSET in
    the base frame, heading chosen from the current base position."""
    t = np.asarray(target_xy, dtype=float)[:2]
    b = np.asarray(base_xy, dtype=float)[:2]
    delta = t - b
    phi = math.atan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-6 else 0.0
    off = ARM_WORKSPACE_OFFSET
    yaw = phi + math.atan2(-off[1], off[0])  # bearing of the offset vector
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    base = t - rot @ off
    return float(base[0]), float(base[1]), float(math.atan2(math.sin(yaw), math.cos(yaw)))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ----------------------------------------------------------------- controller

@dataclass
class ArmCommand:
    """Last accepted right-arm command (for diagnostics)."""

    target_pos: np.ndarray
    target_rot: Optional[np.ndarray]
    orientation_policy: str  # "explicit" | "default" | "position_only"
    ik: IKResult
    q_des: np.ndarray = field(default_factory=lambda: np.zeros(7))


class G1Controller:
    def __init__(self, model: "mujoco.MjModel", data: "mujoco.MjData",
                 limits: Optional[G1Limits] = None, ik_config: Optional[IKConfig] = None) -> None:
        self.model = model
        self.data = data
        self.limits = limits or G1Limits()
        self.dt = float(model.opt.timestep)
        self._qpos_adr = {model.joint(j).name: int(model.jnt_qposadr[j]) for j in range(model.njnt)}
        self._act_for_joint: dict[str, int] = {}
        for a in range(model.nu):
            jid = int(model.actuator_trnid[a, 0])
            self._act_for_joint[model.joint(jid).name] = a
        for j in rest_posture():
            if j not in self._act_for_joint:
                raise ValueError(f"G1 model has no actuator for joint {j!r}")
        self.pelvis_id = int(model.body(PELVIS_BODY).id)
        self.mocap_id = int(model.body(BASE_MOCAP_BODY).mocapid[0])
        self.ee_site_id = int(model.site(RIGHT_EE_SITE).id)
        self.ik = ChainIK(model, RIGHT_EE_SITE, RIGHT_ARM_JOINTS, ik_config or IKConfig())
        # warm-started solver for the streaming setpoints (few iterations)
        self.ik_stream = ChainIK(model, RIGHT_EE_SITE, RIGHT_ARM_JOINTS,
                                 IKConfig(max_iters=40, stall_iters=10))
        self.ik_every = 2  # physics steps between streaming IK solves
        self.max_branch_jump = 0.6  # rad; larger setpoint jumps are IK-branch changes
        self.stuck_steps_allow_jump = 150  # steps (0.3 s) stuck before a branch change is accepted
        self._ik_countdown = 0
        self._cart_target: Optional[np.ndarray] = None  # active Cartesian command (None: hold q_des)
        self._cart_rot: Optional[np.ndarray] = None  # goal orientation (None: position-only)
        self._cart_rot_start: Optional[np.ndarray] = None  # EE orientation when the command started
        self._cart_start = np.zeros(3)  # EE position when the command started
        self._cart_cmd = np.zeros(3)  # streaming setpoint
        self._full_solve_countdown = 0
        self._stuck_steps = 0  # steps the setpoint could not advance (EE not keeping up)
        self._scratch = mujoco.MjData(model)
        self._arm_acts = np.array([self._act_for_joint[j] for j in RIGHT_ARM_JOINTS])
        self._waist_acts = np.array([self._act_for_joint[j] for j in WAIST_JOINTS])
        # Physical rest ("ready") posture of the right arm.
        self.q_rest = np.array([RIGHT_ARM_REST[j] for j in RIGHT_ARM_JOINTS])
        # Null-space bias posture for IK (neutral elbow-bent arm from the
        # upstream keyframe): keeps solutions in the "elbow down/back"
        # family without pulling toward the swung-out ready posture.
        self.q_ik_rest = np.array([IK_REST_BIAS[j] for j in RIGHT_ARM_JOINTS])
        # targets
        self._base_target = np.zeros(3)  # x, y, yaw
        self._base_cmd = np.zeros(3)  # rate-limited command currently on the mocap body
        self._arm_q_des = self.q_rest.copy()
        self._waist_des = np.array([WAIST_REST[j] for j in WAIST_JOINTS])
        self.last_arm_command: Optional[ArmCommand] = None

    # ------------------------------------------------------------ reset

    def reset(self, x: float, y: float, yaw: float) -> None:
        """Place the robot at base pose (x, y, yaw) in the rest posture with
        zero velocity, mocap target consistent with the pelvis (no startup
        jump), and every actuator holding its joint's current value."""
        d = self.data
        adr = self._qpos_adr["floating_base_joint"]
        d.qpos[adr:adr + 3] = (x, y, STAND_HEIGHT)
        d.qpos[adr + 3:adr + 7] = yaw_quat(yaw)
        for j, q in rest_posture().items():
            d.qpos[self._qpos_adr[j]] = q
        d.mocap_pos[self.mocap_id] = (x, y, STAND_HEIGHT)
        d.mocap_quat[self.mocap_id] = yaw_quat(yaw)
        self._base_target[:] = (x, y, yaw)
        self._base_cmd[:] = (x, y, yaw)
        self._arm_q_des = self.q_rest.copy()
        self._cart_target = None
        self._cart_rot = None
        self._waist_des = np.array([WAIST_REST[j] for j in WAIST_JOINTS])
        self.last_arm_command = None
        self._write_hold_ctrl()

    def _write_hold_ctrl(self) -> None:
        for j, a in self._act_for_joint.items():
            self.data.ctrl[a] = self.data.qpos[self._qpos_adr[j]]

    # ------------------------------------------------------------ targets (non-blocking)

    def set_base_target(self, x: float, y: float, yaw: float) -> None:
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError("base target must be finite")
        self._base_target[:] = (x, y, _wrap(yaw))

    def set_waist_target(self, yaw: float, roll: float, pitch: float) -> None:
        vals = np.array([yaw, roll, pitch], dtype=float)
        for j, v in zip(WAIST_JOINTS, vals):
            lo, hi = self.model.joint(j).range
            if not lo <= v <= hi:
                raise ValueError(f"{j} target {v:.3f} outside [{lo:.2f}, {hi:.2f}]")
        self._waist_des = vals

    def set_arm_joint_target(self, q: np.ndarray) -> None:
        """Joint-space setpoint (cancels any Cartesian command)."""
        self._arm_q_des = self.ik.clamp(np.asarray(q, dtype=float))
        self._cart_target = None
        self._cart_rot = None

    def solve_arm_target(self, pos_world: np.ndarray, rot_world: Optional[np.ndarray] = None,
                         allow_position_only_fallback: bool = True,
                         seeds: Optional[list[np.ndarray]] = None) -> tuple[IKResult, str]:
        """IK for the right EE site on a scratch copy of the LIVE state (base
        + waist as they are now).  Orientation policy:

        * ``rot_world`` given  -> "explicit": pose-constrained solve only.
        * ``rot_world`` None   -> "default": request the tilted top-down
          approach orientation (approach_rotation of the current base yaw);
          if that fails and fallback is allowed, re-solve position-only
          ("position_only") — the orientation is then whatever the
          rest-posture bias yields.

        Never mutates live data or time.  Returns (best result, policy)."""
        p = np.asarray(pos_world, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError(f"pos_world must be a finite 3-vector, got {pos_world!r}")
        base_yaw = self.base_pose()[2]
        if rot_world is not None:
            attempts = [(np.asarray(rot_world, dtype=float), "explicit")]
        else:
            attempts = [(approach_rotation(base_yaw), "default")]
            if allow_position_only_fallback:
                attempts.append((None, "position_only"))
        seed_list = seeds or [np.array(self.data.ctrl[self._arm_acts]), self.q_ik_rest,
                              np.array([-0.6, -0.4, 0.0, 1.0, 0.0, 0.0, 0.0]),
                              np.array([-1.0, -0.8, 0.5, 1.5, 0.0, 0.0, 0.0]),
                              np.array([-0.3, -1.0, -0.5, 0.8, 0.0, 0.5, 0.0])]
        best: Optional[IKResult] = None
        best_policy = attempts[0][1]
        for rot, policy in attempts:
            for seed in seed_list:
                self._scratch.qpos[:] = self.data.qpos
                res = self.ik.solve(self._scratch, seed, p, rot, self.q_ik_rest)
                if res.success:
                    return res, policy
                if best is None or res.pos_err < best.pos_err:
                    best, best_policy = res, policy
        assert best is not None
        return best, best_policy

    def set_arm_target(self, pos_world: np.ndarray, rot_world: Optional[np.ndarray] = None) -> IKResult:
        """Non-blocking: validate the target with IK and start Cartesian
        streaming toward it.  Raises ValueError when no solution exists.
        Re-issuing the SAME target while it is still being executed is a
        no-op (the stream keeps going; the live base pose is used by every
        streaming solve anyway)."""
        p_req = np.asarray(pos_world, dtype=float)
        cmd = self.last_arm_command
        if (self._cart_target is not None and cmd is not None and p_req.shape == (3,)
                and np.allclose(p_req, self._cart_target, atol=1e-9)
                and ((rot_world is None and cmd.target_rot is None)
                     or (rot_world is not None and cmd.target_rot is not None
                         and np.allclose(np.asarray(rot_world, float), cmd.target_rot, atol=1e-9)))):
            return cmd.ik
        res, policy = self.solve_arm_target(pos_world, rot_world)
        if not res.success:
            raise ValueError(
                f"target {np.asarray(pos_world, float).round(3).tolist()} unreachable for the right arm "
                f"({policy}: best pos_err {res.pos_err * 1000:.1f} mm, rot_err "
                f"{math.degrees(res.rot_err):.1f} deg, {res.reason})"
            )
        p = np.asarray(pos_world, dtype=float).copy()
        if policy == "explicit":
            rot = np.asarray(rot_world, dtype=float).copy()
        elif policy == "default":
            rot = approach_rotation(self.base_pose()[2])
        else:
            rot = None
        # Start (or restart) Cartesian streaming from the CURRENT EE pose;
        # position moves along the straight line, orientation is slerped
        # from the current EE orientation to the goal along the same path.
        self._cart_target = p
        self._cart_rot = rot
        self._cart_rot_start = self.ee_rot().copy()
        self._cart_start = self.ee_pos().copy()
        self._cart_cmd = self._cart_start.copy()
        self._ik_countdown = 0
        self._full_solve_countdown = 0
        self.last_arm_command = ArmCommand(p, None if rot_world is None else np.asarray(rot_world, float).copy(),
                                           policy, res, res.q.copy())
        return res

    # ------------------------------------------------------------ per-step update

    def update(self) -> None:
        """Advance targets toward setpoints with rate limits and write the
        mocap pose and actuator controls.  Call once before each mj_step."""
        d, dt, lim = self.data, self.dt, self.limits
        # base (mocap) with translational and angular speed limits
        delta = self._base_target[:2] - self._base_cmd[:2]
        dist = float(np.linalg.norm(delta))
        if dist > 1e-12:
            step = min(dist, lim.base_lin_speed * dt)
            self._base_cmd[:2] += delta / dist * step
        dyaw = _wrap(self._base_target[2] - self._base_cmd[2])
        self._base_cmd[2] = _wrap(self._base_cmd[2] + float(np.clip(dyaw, -lim.base_ang_speed * dt, lim.base_ang_speed * dt)))
        d.mocap_pos[self.mocap_id] = (self._base_cmd[0], self._base_cmd[1], STAND_HEIGHT)
        d.mocap_quat[self.mocap_id] = yaw_quat(self._base_cmd[2])
        # right arm: Cartesian setpoint streaming + warm-started IK
        if self._cart_target is not None:
            self._stream_arm(d, dt, lim)
        # feed-forward gravity/Coriolis compensation on the right-arm DoFs
        # (qfrc_bias of the current state); the position actuators then only
        # correct tracking errors instead of holding the arm's weight.
        if lim.gravity_compensation:
            d.qfrc_applied[self.ik.dof_ids] = d.qfrc_bias[self.ik.dof_ids]
        # right arm / waist setpoints: SYNCHRONIZED joint-rate limit (all
        # joints scaled by the slowest one, so the setpoint moves along a
        # straight line in joint space and finishes simultaneously)
        d.ctrl[self._arm_acts] = self._rate_limited(d.ctrl[self._arm_acts], self._arm_q_des, lim.arm_joint_speed * dt)
        d.ctrl[self._waist_acts] = self._rate_limited(d.ctrl[self._waist_acts], self._waist_des, lim.waist_joint_speed * dt)
        # legs, left arm, fingers: held at their reset values (unchanged ctrl)

    def _stream_arm(self, d: "mujoco.MjData", dt: float, lim: G1Limits) -> None:
        """Advance the Cartesian setpoint ("pull-along": never more than
        ``ee_lead`` ahead of the live EE) and re-solve IK for it."""
        assert self._cart_target is not None
        delta = self._cart_target - self._cart_cmd
        dist = float(np.linalg.norm(delta))
        step = min(dist, lim.ee_speed * dt)
        lead = float(np.linalg.norm(self._cart_cmd - self.ee_pos()))
        if dist > 1e-12 and lead <= lim.ee_lead:
            self._cart_cmd = self._cart_cmd + delta / dist * step
            dist -= step
            self._stuck_steps = 0
        elif dist > 1e-12:
            self._stuck_steps += 1
        arrived = dist <= 1e-9
        self._ik_countdown -= 1
        self._full_solve_countdown -= 1
        if self._ik_countdown > 0 and not arrived:
            return
        self._ik_countdown = self.ik_every
        # orientation along the path (fraction of the straight line covered)
        rot_cmd = None
        if self._cart_rot is not None:
            total = float(np.linalg.norm(self._cart_target - self._cart_start))
            frac = 1.0 if total < 1e-9 else float(np.linalg.norm(self._cart_cmd - self._cart_start)) / total
            rot_cmd = self._cart_rot if arrived else interpolate_rotation(self._cart_rot_start, self._cart_rot, frac)
        self._scratch.qpos[:] = d.qpos  # live base/waist configuration
        res = self.ik_stream.solve(self._scratch, self._arm_q_des, self._cart_cmd, rot_cmd, self.q_ik_rest)
        if not res.success and rot_cmd is not None and not arrived:
            # intermediate orientation infeasible here: keep the position on
            # the line, let the orientation catch up later
            alt = self.ik_stream.solve(self._scratch, self._arm_q_des, self._cart_cmd, None, self.q_ik_rest)
            if alt.pos_err < res.pos_err:
                res = alt
        if not res.success and self._full_solve_countdown <= 0:
            # warm start stuck (joint limit / local minimum): bounded-rate
            # full multi-seed solve
            self._full_solve_countdown = 25
            full, _ = self.solve_arm_target(self._cart_cmd, rot_cmd, allow_position_only_fallback=True)
            if full.success or full.pos_err < res.pos_err:
                res = full
        # continuity guard: a solution far from the current setpoint means a
        # different IK branch (e.g. elbow flip) — only accept it when the
        # stream has been stuck for a while (otherwise keep the current
        # branch and let the pull-along wait for the arm)
        jump = float(np.max(np.abs(res.q - self._arm_q_des)))
        if res.pos_err < 0.05 and (jump <= self.max_branch_jump or self._stuck_steps > self.stuck_steps_allow_jump):
            self._arm_q_des = res.q
        if arrived and res.success and jump <= self.max_branch_jump:
            self._cart_target = None  # reached: hold the final joint setpoint

    @staticmethod
    def _rate_limited(cur: np.ndarray, des: np.ndarray, max_step: float) -> np.ndarray:
        delta = des - cur
        worst = float(np.max(np.abs(delta))) if delta.size else 0.0
        if worst <= max_step:
            return des.copy()
        return cur + delta * (max_step / worst)

    # ------------------------------------------------------------ state

    def base_pose(self) -> np.ndarray:
        adr = self._qpos_adr["floating_base_joint"]
        return np.array([self.data.qpos[adr], self.data.qpos[adr + 1], quat_yaw(self.data.qpos[adr + 3:adr + 7])])

    def base_target(self) -> np.ndarray:
        return self._base_target.copy()

    def base_command(self) -> np.ndarray:
        return self._base_cmd.copy()

    def ee_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.ee_site_id], dtype=float)

    def ee_rot(self) -> np.ndarray:
        return np.array(self.data.site_xmat[self.ee_site_id], dtype=float).reshape(3, 3)

    def arm_q(self) -> np.ndarray:
        return np.array(self.data.qpos[self.ik.qpos_ids], dtype=float)

    def arm_qvel(self) -> np.ndarray:
        return np.array(self.data.qvel[self.ik.dof_ids], dtype=float)

    def arm_q_des(self) -> np.ndarray:
        return self._arm_q_des.copy()

    def arm_ctrl(self) -> np.ndarray:
        return np.array(self.data.ctrl[self._arm_acts], dtype=float)

    def waist_q(self) -> np.ndarray:
        return np.array([self.data.qpos[self._qpos_adr[j]] for j in WAIST_JOINTS])

    def tracking_error(self) -> tuple[float, float]:
        """(position m, orientation rad) error of the live EE site w.r.t.
        the last accepted arm command (inf when none)."""
        cmd = self.last_arm_command
        if cmd is None:
            return math.inf, math.inf
        pe = float(np.linalg.norm(self.ee_pos() - cmd.target_pos))
        if cmd.orientation_policy == "position_only":
            return pe, 0.0
        rot = cmd.target_rot if cmd.target_rot is not None else approach_rotation(self.base_pose()[2])
        return pe, rotation_error_rad(rot, self.ee_rot())

    # ------------------------------------------------------------ privileged (mocks/tests)

    def teleport_base(self, x: float, y: float, yaw: float) -> None:
        """Instantaneous base placement (mocks only): pelvis qpos, mocap
        body and controller targets are all set consistently."""
        d = self.data
        adr = self._qpos_adr["floating_base_joint"]
        d.qpos[adr:adr + 3] = (x, y, STAND_HEIGHT)
        d.qpos[adr + 3:adr + 7] = yaw_quat(yaw)
        dof = int(self.model.jnt_dofadr[self.model.joint("floating_base_joint").id])
        d.qvel[dof:dof + 6] = 0.0
        d.mocap_pos[self.mocap_id] = (x, y, STAND_HEIGHT)
        d.mocap_quat[self.mocap_id] = yaw_quat(yaw)
        self._base_target[:] = (x, y, _wrap(yaw))
        self._base_cmd[:] = self._base_target

    def teleport_arm(self, q: np.ndarray) -> None:
        """Instantaneous right-arm placement (mocks only)."""
        q = self.ik.clamp(np.asarray(q, dtype=float))
        self.data.qpos[self.ik.qpos_ids] = q
        self.data.qvel[self.ik.dof_ids] = 0.0
        self.data.ctrl[self._arm_acts] = q
        self._arm_q_des = q.copy()
        self._cart_target = None
        self._cart_rot = None

    def joint_limit_violations(self, tol: float = 1e-3) -> list[str]:
        out = []
        m = self.model
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j]:
                continue
            q = float(self.data.qpos[m.jnt_qposadr[j]])
            lo, hi = m.jnt_range[j]
            if q < lo - tol or q > hi + tol:
                out.append(m.joint(j).name)
        return out
