# Owner: backbone (ALL)
"""Basic reference motion adapters for the simplified platform.

REPLACEABLE BACKBONE REFERENCE IMPLEMENTATIONS — these exist so the
backbone (mocks, smoke trials, environment tests) can move the robot; they
are NOT Student C's completed executor module.  Student C's real executor
may reuse, extend, or replace them behind the Executor interface.

All primitives:
* operate only on the public RobotEnvProtocol (no ground truth),
* use bounded step loops with explicit tolerances and sim-time timeouts,
* never teleport objects,
* return core.types.SkillResult.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from core.g1 import ARM_WORKSPACE_OFFSET, approach_base_pose
from core.interfaces import Perception, RobotEnvProtocol
from core.types import ErrorCode, GroundedObject, SkillResult

BASE_POS_TOL = 0.02  # m
BASE_YAW_TOL = 0.05  # rad
EE_POS_TOL = 0.012  # m
DEFAULT_TIMEOUT_S = 12.0  # sim seconds per primitive
# APPROACH parks the base so the target lands at ARM_WORKSPACE_OFFSET in the
# base frame (right-arm workspace, see core.g1); this is the xy distance.
APPROACH_STANDOFF = float(np.linalg.norm(ARM_WORKSPACE_OFFSET))
# GRASP reaches with the palm reference point this far ABOVE the object
# center (still inside ATTACH_RADIUS) so the fingers/thumb stay clear of
# the support surface.
GRASP_DESCEND_OFFSET = 0.02
SETTLE_STEPS = 375  # 0.75 s at the 2 ms timestep


def _step_until(env: RobotEnvProtocol, done, timeout_s: float = DEFAULT_TIMEOUT_S,
                chunk: int = 10) -> bool:
    """Bounded stepping loop: advances sim until ``done()`` or timeout.
    Returns True when the predicate was met."""
    deadline = env.sim_time() + timeout_s
    while env.sim_time() < deadline:
        if done():
            return True
        env.step(chunk)
    return done()


def approach(env: RobotEnvProtocol, pos_world: np.ndarray,
             standoff: float = APPROACH_STANDOFF,
             timeout_s: float = DEFAULT_TIMEOUT_S) -> SkillResult:
    """Drive the base to the parking pose that puts pos_world in the right
    arm's workspace (about ``standoff`` meters away, target ahead-right)."""
    p = np.asarray(pos_world, dtype=float)
    base = env.get_base_pose()
    bx, by, yaw = approach_base_pose(p[:2], base[:2])
    target_xy = np.array([bx, by])
    env.set_base_target(float(target_xy[0]), float(target_xy[1]), yaw)

    def done() -> bool:
        b = env.get_base_pose()
        yaw_err = abs(math.atan2(math.sin(b[2] - yaw), math.cos(b[2] - yaw)))
        return bool(np.linalg.norm(b[:2] - target_xy) < BASE_POS_TOL and yaw_err < BASE_YAW_TOL)

    if not _step_until(env, done, timeout_s):
        return SkillResult(False, ErrorCode.TIMEOUT, {"primitive": "approach"})
    return SkillResult(True, ErrorCode.NONE, {"primitive": "approach"})


PRE_REACH_HEIGHT = 0.10  # m: via-point above the target for long reaches
PRE_REACH_XY_TRIGGER = 0.08  # m: use the via-point when farther than this (xy)


def _reach_point(env: RobotEnvProtocol, p: np.ndarray, timeout_s: float, tol: float) -> SkillResult:
    try:
        env.set_arm_target(p)
    except ValueError as exc:
        return SkillResult(False, ErrorCode.UNREACHABLE, {"primitive": "reach", "detail": str(exc)})

    def done() -> bool:
        if bool(np.linalg.norm(env.get_ee_pos() - p) < tol):
            return True
        # Closed loop: the base may still be creeping toward its own target,
        # so re-resolve the arm command against the CURRENT base pose.
        try:
            env.set_arm_target(p)
        except ValueError:
            pass
        return False

    if not _step_until(env, done, timeout_s):
        return SkillResult(False, ErrorCode.TIMEOUT, {
            "primitive": "reach",
            "ee_error": float(np.linalg.norm(env.get_ee_pos() - p)),
        })
    return SkillResult(True, ErrorCode.NONE, {"primitive": "reach"})


def reach(env: RobotEnvProtocol, pos_world: np.ndarray,
          timeout_s: float = DEFAULT_TIMEOUT_S) -> SkillResult:
    """Move the end effector to pos_world (world frame).  Long reaches go
    through a via-point PRE_REACH_HEIGHT above the target first so the
    hand descends vertically instead of sweeping across the support
    surface (the humanoid arm's joint-space path is otherwise unconstrained)."""
    p = np.asarray(pos_world, dtype=float)
    ee = env.get_ee_pos()
    if np.linalg.norm(ee[:2] - p[:2]) > PRE_REACH_XY_TRIGGER:
        via = p + np.array([0.0, 0.0, PRE_REACH_HEIGHT])
        pre = _reach_point(env, via, timeout_s, tol=3.0 * EE_POS_TOL)
        if not pre.success and pre.error_code is ErrorCode.UNREACHABLE:
            pass  # via-point outside the envelope: go straight to the target
        elif not pre.success:
            return pre
    return _reach_point(env, p, timeout_s, EE_POS_TOL)


def grasp(env: RobotEnvProtocol, pos_world: np.ndarray,
          timeout_s: float = DEFAULT_TIMEOUT_S) -> SkillResult:
    """Reach to the target position and attempt attachment there."""
    if env.is_attached():
        return SkillResult(False, ErrorCode.ALREADY_HOLDING, {"primitive": "grasp"})
    r = reach(env, np.asarray(pos_world, dtype=float) + [0.0, 0.0, GRASP_DESCEND_OFFSET], timeout_s)
    if not r.success:
        return SkillResult(False, r.error_code, {"primitive": "grasp", **r.info})
    handle = env.try_attach_near_ee()
    if handle is None:
        return SkillResult(False, ErrorCode.GRASP_MISSED, {"primitive": "grasp"})
    env.step(50)  # let the attachment stabilize
    return SkillResult(True, ErrorCode.NONE, {"primitive": "grasp", "attachment": handle})


def move_to(env: RobotEnvProtocol, pos_world: np.ndarray,
            timeout_s: float = DEFAULT_TIMEOUT_S) -> SkillResult:
    """Transport the (held or empty) end effector to pos_world; drives the
    base first when the point is outside the arm envelope."""
    p = np.asarray(pos_world, dtype=float)
    try:
        env.set_arm_target(p)
    except ValueError:
        a = approach(env, p, timeout_s=timeout_s)
        if not a.success:
            return SkillResult(False, a.error_code, {"primitive": "move_to", **a.info})
        try:
            env.set_arm_target(p)
        except ValueError as exc:
            return SkillResult(False, ErrorCode.UNREACHABLE, {"primitive": "move_to", "detail": str(exc)})
    r = reach(env, p, timeout_s)
    if not r.success:
        return SkillResult(False, r.error_code, {"primitive": "move_to", **r.info})
    return SkillResult(True, ErrorCode.NONE, {"primitive": "move_to"})


def place(env: RobotEnvProtocol, settle_steps: int = SETTLE_STEPS) -> SkillResult:
    """Release the attachment and let the object settle."""
    if not env.is_attached():
        return SkillResult(False, ErrorCode.NOT_HOLDING, {"primitive": "place"})
    env.detach()
    env.step(settle_steps)
    if env.is_attached():
        return SkillResult(False, ErrorCode.PLACE_FAILED, {"primitive": "place"})
    return SkillResult(True, ErrorCode.NONE, {"primitive": "place"})


def search(env: RobotEnvProtocol, perception: Perception, target: str,
           step_rad: float = 0.5, max_views: int = 13,
           timeout_s: float = 60.0) -> SkillResult:
    """Rotate the base in increments, re-observing until ``perception``
    grounds ``target``.  Bounded by max_views and a global sim timeout.
    Not-found is the recoverable SEARCH_NOT_FOUND outcome; perception
    exceptions surface as SEARCH_FATAL."""
    deadline = env.sim_time() + timeout_s
    base = env.get_base_pose()
    found: Optional[GroundedObject] = None
    for k in range(max_views):
        if env.sim_time() >= deadline:
            return SkillResult(False, ErrorCode.TIMEOUT, {"primitive": "search", "views": k})
        obs = env.get_obs()
        try:
            found = perception.ground(obs, target)
        except Exception as exc:  # fatal: perception itself failed
            return SkillResult(False, ErrorCode.SEARCH_FATAL,
                               {"primitive": "search", "detail": f"{type(exc).__name__}: {exc}"})
        if found is not None:
            return SkillResult(True, ErrorCode.NONE,
                               {"primitive": "search", "grounded": found, "frame_id": obs.frame_id, "views": k + 1})
        # rotate to the next view (wrap yaw into the joint range)
        yaw = base[2] + step_rad * (k + 1)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        env.set_base_target(float(base[0]), float(base[1]), yaw)

        def turned() -> bool:
            b = env.get_base_pose()
            return abs(math.atan2(math.sin(b[2] - yaw), math.cos(b[2] - yaw))) < BASE_YAW_TOL

        _step_until(env, turned, timeout_s=4.0)
    return SkillResult(False, ErrorCode.SEARCH_NOT_FOUND, {"primitive": "search", "views": max_views})
