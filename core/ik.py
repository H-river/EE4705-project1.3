# Owner: backbone (ALL)
"""Damped least-squares inverse kinematics for one kinematic chain.

Solves for the joint values of a declared set of hinge joints (e.g. the
seven right-arm joints of the G1) that bring a MuJoCo *site* to a target
world pose.  Design points:

* Works on a caller-provided **scratch** ``MjData``: the solver only runs
  ``mj_kinematics``/``mj_comPos`` on that copy and never touches the live
  simulation state or advances time.
* World-frame position error and SO(3) orientation error (rotation vector
  of ``R_target @ R_current^T``; never Euler subtraction), with explicit
  position/orientation weights.
* Damping ``lambda`` (default 0.05), per-iteration step bound, joint-limit
  clamping, a rest-posture bias projected into the task null space (so it
  cannot override task convergence), bounded iterations, and a
  deterministic ``reason`` string on failure.

IK convergence (this module) is a *kinematic* statement.  Whether the
actuated arm actually tracks the solution, and whether the resulting
configuration is collision-free, are separate checks made by the caller
(see core.g1 / scripts/ik_reach_test.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.rendering import mujoco


@dataclass(frozen=True)
class IKConfig:
    damping: float = 0.05  # lambda in J^T (J J^T + lambda^2 I)^-1
    pos_weight: float = 1.0  # weight of position error (per meter)
    rot_weight: float = 0.3  # weight of orientation error (per radian)
    max_iters: int = 200
    max_step_rad: float = 0.25  # bound on ||dq||_inf per iteration
    pos_tol: float = 1e-3  # m; solver-internal convergence tolerance
    rot_tol: float = np.deg2rad(0.5)  # rad
    rest_gain: float = 0.02  # null-space pull toward the rest posture
    limit_margin: float = 0.01  # rad kept away from hard joint limits
    stall_iters: int = 25  # stop when no improvement for this many iterations


@dataclass
class IKResult:
    success: bool
    q: np.ndarray  # joint values (len(dof_ids),) — best iterate, even on failure
    pos_err: float  # m, at the returned q
    rot_err: float  # rad, at the returned q (0 when orientation unconstrained)
    iters: int
    reason: str  # "converged" | "max_iters" | "stalled" | "invalid_target"
    info: dict = field(default_factory=dict)


class ChainIK:
    """IK for a fixed set of hinge joints driving one site."""

    def __init__(self, model: "mujoco.MjModel", site_name: str, joint_names: list[str],
                 config: Optional[IKConfig] = None,
                 range_overrides: Optional[dict[str, tuple[float, float]]] = None) -> None:
        """``range_overrides`` narrows the IK search range of named joints
        (must lie inside the model's joint range) — used to exclude
        undesirable solution families such as a hyper-extended elbow."""
        self.model = model
        self.cfg = config or IKConfig()
        self.site_id = int(model.site(site_name).id)
        self.joint_ids = [int(model.joint(n).id) for n in joint_names]
        for j in self.joint_ids:
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"IK chain joint {model.joint(j).name} is not a hinge")
        self.qpos_ids = np.array([int(model.jnt_qposadr[j]) for j in self.joint_ids])
        self.dof_ids = np.array([int(model.jnt_dofadr[j]) for j in self.joint_ids])
        self.lower = np.array([float(model.jnt_range[j][0]) for j in self.joint_ids])
        self.upper = np.array([float(model.jnt_range[j][1]) for j in self.joint_ids])
        for name, (lo, hi) in (range_overrides or {}).items():
            k = joint_names.index(name)
            if lo < self.lower[k] - 1e-9 or hi > self.upper[k] + 1e-9 or lo >= hi:
                raise ValueError(f"IK range override for {name} must lie inside the joint range")
            self.lower[k], self.upper[k] = float(lo), float(hi)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ------------------------------------------------------------ kinematics

    def _fk(self, data: "mujoco.MjData", q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[self.qpos_ids] = q
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)  # needed for correct Jacobians
        pos = np.array(data.site_xpos[self.site_id], dtype=float)
        rot = np.array(data.site_xmat[self.site_id], dtype=float).reshape(3, 3)
        return pos, rot

    def _jacobian(self, data: "mujoco.MjData") -> np.ndarray:
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.site_id)
        return np.vstack([self._jacp[:, self.dof_ids], self._jacr[:, self.dof_ids]])

    def _dls_step(self, Jw: np.ndarray, ew: np.ndarray, q: np.ndarray, q_rest: np.ndarray) -> np.ndarray:
        """One damped-least-squares update plus a rest-posture bias projected
        into the EXACT task null space (pseudo-inverse projector, so the bias
        cannot fight task convergence the way a damped projector would)."""
        cfg = self.cfg
        m = Jw.shape[0]
        JJt = Jw @ Jw.T + (cfg.damping ** 2) * np.eye(m)
        J_dls = Jw.T @ np.linalg.solve(JJt, np.eye(m))
        dq = J_dls @ ew
        if cfg.rest_gain > 0.0:
            null = np.eye(len(q)) - np.linalg.pinv(Jw, rcond=1e-3) @ Jw
            dq = dq + null @ (cfg.rest_gain * (q_rest - q))
        return dq

    def site_pose(self, data: "mujoco.MjData") -> tuple[np.ndarray, np.ndarray]:
        """Current (pos, R) of the site in ``data`` (no recomputation)."""
        return (np.array(data.site_xpos[self.site_id], dtype=float),
                np.array(data.site_xmat[self.site_id], dtype=float).reshape(3, 3))

    def clamp(self, q: np.ndarray) -> np.ndarray:
        m = self.cfg.limit_margin
        return np.clip(q, self.lower + m, self.upper - m)

    # ------------------------------------------------------------ solve

    def solve(self, scratch: "mujoco.MjData", q_init: np.ndarray, target_pos: np.ndarray,
              target_rot: Optional[np.ndarray] = None, q_rest: Optional[np.ndarray] = None,
              rot_weight: Optional[float] = None) -> IKResult:
        """Solve on ``scratch`` (a copy of the live state; its other joints
        — base, waist — define the chain's root pose).  ``target_rot`` is a
        3x3 world-frame rotation for the site, or None for a position-only
        target.  Returns the best iterate found."""
        cfg = self.cfg
        p_t = np.asarray(target_pos, dtype=float)
        if p_t.shape != (3,) or not np.all(np.isfinite(p_t)):
            return IKResult(False, self.clamp(np.asarray(q_init, float)), np.inf, np.inf, 0, "invalid_target")
        R_t = None if target_rot is None else np.asarray(target_rot, dtype=float)
        if R_t is not None and (R_t.shape != (3, 3) or not np.all(np.isfinite(R_t))):
            return IKResult(False, self.clamp(np.asarray(q_init, float)), np.inf, np.inf, 0, "invalid_target")
        w_rot = cfg.rot_weight if rot_weight is None else float(rot_weight)
        q = self.clamp(np.asarray(q_init, dtype=float).copy())
        q_rest = q.copy() if q_rest is None else np.asarray(q_rest, dtype=float)
        n = len(q)
        best_q, best_cost, best_pe, best_re = q.copy(), np.inf, np.inf, np.inf
        since_improvement = 0
        reason = "max_iters"
        iters = 0
        for iters in range(1, cfg.max_iters + 1):
            pos, R = self._fk(scratch, q)
            e_pos = p_t - pos
            pe = float(np.linalg.norm(e_pos))
            if R_t is not None:
                e_rot = _rotvec(R_t @ R.T)
                re = float(np.linalg.norm(e_rot))
            else:
                e_rot = None
                re = 0.0
            cost = cfg.pos_weight * pe + w_rot * re
            if cost < best_cost - 1e-9:
                best_cost, best_q, best_pe, best_re = cost, q.copy(), pe, re
                since_improvement = 0
            else:
                since_improvement += 1
            if pe <= cfg.pos_tol and (R_t is None or re <= cfg.rot_tol):
                reason = "converged"
                break
            if since_improvement >= cfg.stall_iters:
                reason = "stalled"
                break
            J = self._jacobian(scratch)
            if R_t is None:
                Jw = cfg.pos_weight * J[:3]
                ew = cfg.pos_weight * e_pos
            else:
                Jw = np.vstack([cfg.pos_weight * J[:3], w_rot * J[3:]])
                ew = np.concatenate([cfg.pos_weight * e_pos, w_rot * e_rot])
            dq = self._dls_step(Jw, ew, q, q_rest)
            # active set: a joint sitting on a limit must not be pushed
            # further into it — drop its column and re-solve once
            lo, hi = self.lower + cfg.limit_margin, self.upper - cfg.limit_margin
            blocked = ((q <= lo + 1e-9) & (dq < 0)) | ((q >= hi - 1e-9) & (dq > 0))
            if np.any(blocked):
                Jb = Jw.copy()
                Jb[:, blocked] = 0.0
                dq = self._dls_step(Jb, ew, q, q_rest)
                dq[blocked] = 0.0
            step = float(np.max(np.abs(dq)))
            if step > cfg.max_step_rad:
                dq *= cfg.max_step_rad / step
            q = self.clamp(q + dq)
        success = reason == "converged"
        if not success:
            # report the best iterate, re-evaluated
            q = best_q
            pe, re = best_pe, best_re
        return IKResult(success, q.copy(), float(pe), float(re), iters, reason,
                        info={"orientation_constrained": R_t is not None, "rot_weight": w_rot})


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix via MuJoCo."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R, dtype=float).reshape(9))
    vel = np.empty(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


def rotvec_to_mat(rotvec: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (axis * angle)."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(rotvec, float) / angle, angle)
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def interpolate_rotation(R_start: np.ndarray, R_goal: np.ndarray, s: float) -> np.ndarray:
    """Geodesic interpolation on SO(3): R(s) = exp(s * log(R_goal R_start^T)) R_start."""
    s = float(np.clip(s, 0.0, 1.0))
    return rotvec_to_mat(s * rotvec_between(R_goal, R_start)) @ np.asarray(R_start, float)


def rotvec_between(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking R_current to R_target."""
    return _rotvec(np.asarray(R_target, float) @ np.asarray(R_current, float).T)


def rotation_error_rad(R_target: np.ndarray, R_current: np.ndarray) -> float:
    return float(np.linalg.norm(rotvec_between(R_target, R_current)))
