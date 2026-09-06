# Owner: backbone (ALL)
"""Focused control tests for the Unitree G1 platform: settling/stability,
base motion through the mocap weld, arm reaching, unreachable targets,
attachment and reset."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core import skills
from core.g1 import CAMERAS, RIGHT_ARM_REST, WAIST_REST, approach_rotation
from core.rendering import mujoco
from core.types import SceneConfig
from core.world import ATTACH_RADIUS

TABLE_TOP_Z = 0.85


def _finite_state(world) -> bool:
    d = world.data
    return bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)) and np.all(np.isfinite(d.ctrl)))


def test_settling_and_stability(standard_world):
    """Held posture for 3 s: finite state, no drift, no joint-limit
    violations, no persistent oscillation, no penetrating robot contacts
    (except none expected)."""
    w = standard_world
    r = w.robot
    base0 = w.base_pose().copy()
    ee0 = w.ee_pos().copy()
    q_arm0 = r.arm_q().copy()
    speeds = []
    for _ in range(30):
        w.step(int(round(0.1 / w.timestep)))
        assert _finite_state(w)
        speeds.append(float(np.max(np.abs(w.data.qvel[6:]))))
    assert np.linalg.norm(w.base_pose()[:2] - base0[:2]) < 0.002
    assert abs(math.atan2(math.sin(w.base_pose()[2] - base0[2]), math.cos(w.base_pose()[2] - base0[2]))) < 0.01
    assert np.linalg.norm(w.ee_pos() - ee0) < 0.01, "arm drifted while holding"
    assert np.max(np.abs(r.arm_q() - q_arm0)) < 0.05
    assert r.joint_limit_violations() == []
    # persistent oscillation: joint speeds must have decayed to ~0 in the last second
    assert max(speeds[-10:]) < 0.02, speeds[-10:]
    contacts = [c for c in w.robot_contacts() if not any(o in c[:2] for o in ("stone", "cube", "bottle", "stone2"))]
    assert contacts == [], contacts


def test_base_moves_through_mocap_with_speed_limits(standard_world, env):
    w = standard_world
    start = w.base_pose().copy()
    env.set_base_target(start[0] + 0.3, start[1] - 0.2, start[2] + 0.8)
    # after 0.2 s the base cannot have moved more than v_max * t (+ weld compliance)
    w.step(int(round(0.2 / w.timestep)))
    moved = np.linalg.norm(w.base_pose()[:2] - start[:2])
    assert moved <= w.robot.limits.base_lin_speed * 0.2 + 0.01, moved
    assert abs(w.base_pose()[2] - start[2]) <= w.robot.limits.base_ang_speed * 0.2 + 0.02
    # eventually converges
    w.step(int(round(2.5 / w.timestep)))
    b = w.base_pose()
    assert np.linalg.norm(b[:2] - (start[:2] + [0.3, -0.2])) < 0.01
    assert abs(math.atan2(math.sin(b[2] - start[2] - 0.8), math.cos(b[2] - start[2] - 0.8))) < 0.02
    # standing height and upright orientation are preserved
    adr = w.model.joint("floating_base_joint").qposadr[0]
    assert abs(w.data.qpos[adr + 2] - 0.79) < 0.01
    quat = w.data.qpos[adr + 3:adr + 7]
    up = np.empty(3)
    mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), quat)
    assert up[2] > 0.999


def test_mocap_initialised_consistently_no_startup_jump(world):
    from tests.conftest import standard_scene

    world.reset(standard_scene())
    p0 = world.base_pose().copy()
    ee0 = world.ee_pos().copy()
    world.step(25)  # 50 ms
    assert np.linalg.norm(world.base_pose()[:2] - p0[:2]) < 0.002
    assert np.linalg.norm(world.ee_pos() - ee0) < 0.01


def test_set_arm_target_is_non_blocking(standard_world, env):
    t = standard_world.sim_time
    env.set_arm_target(np.array([0.36, -0.20, 0.95]))
    assert standard_world.sim_time == t  # no time advanced inside the target update


@pytest.mark.parametrize("target", [
    (0.36, -0.20, TABLE_TOP_Z + 0.05),
    (0.40, -0.26, TABLE_TOP_Z + 0.05),
    (0.32, -0.26, TABLE_TOP_Z + 0.12),
])
def test_arm_reaches_feasible_targets_with_orientation(standard_world, target):
    """Actual simulated EE (not just IK) converges: < 0.01 m and < 5 deg for
    a pose-constrained command (explicit default approach orientation)."""
    w = standard_world
    rot = approach_rotation(w.base_pose()[2])
    p = np.array(target)
    res = w.set_arm_target(p, rot)
    assert res.success
    deadline = w.sim_time + 6.0
    while w.sim_time < deadline:
        w.step(10)
        pe, re = w.robot.tracking_error()
        if pe < 0.01 and re < math.radians(5) and np.max(np.abs(w.robot.arm_qvel())) < 0.05:
            break
    pe, re = w.robot.tracking_error()
    assert pe < 0.01, pe
    assert math.degrees(re) < 5.0, math.degrees(re)
    assert np.linalg.norm(w.ee_pos() - p) < 0.01
    assert w.robot.joint_limit_violations() == []
    assert _finite_state(w)


def test_small_motion_settles_quickly(standard_world):
    """A 5 cm step from a settled reach settles within ~0.5 s (tuning objective)."""
    w = standard_world
    p = np.array([0.36, -0.20, TABLE_TOP_Z + 0.10])
    w.set_arm_target(p, approach_rotation(w.base_pose()[2]))
    w.step(int(round(5.0 / w.timestep)))
    pe, _ = w.robot.tracking_error()
    assert pe < 0.01
    p2 = p + np.array([0.0, 0.0, -0.05])
    w.set_arm_target(p2, approach_rotation(w.base_pose()[2]))
    t0 = w.sim_time
    settled = None
    while w.sim_time - t0 < 1.5:
        w.step(5)
        pe, re = w.robot.tracking_error()
        if pe < 0.01 and re < math.radians(5) and np.max(np.abs(w.robot.arm_qvel())) < 0.05:
            settled = w.sim_time - t0
            break
    assert settled is not None and settled <= 0.75, settled


def test_unreachable_target_terminates_cleanly(standard_world, env):
    """Far targets are rejected by IK deterministically (bounded iterations,
    no NaNs) and the simulation state stays finite."""
    with pytest.raises(ValueError):
        env.set_arm_target(np.array([1.5, -0.2, 0.9]))
    with pytest.raises(ValueError):
        env.set_arm_target(np.array([0.4, 0.6, 0.9]))  # far left: right arm cannot reach
    with pytest.raises(ValueError):
        env.set_arm_target(np.array([np.nan, 0.0, 0.9]))
    res, policy = standard_world.robot.solve_arm_target(np.array([2.0, 0.0, 0.9]))
    assert not res.success and np.all(np.isfinite(res.q)) and res.iters <= 200 * 2
    assert res.reason in ("stalled", "max_iters")
    r = skills.reach(env, np.array([1.5, -0.2, 0.9]))
    assert not r.success and r.error_code.name == "UNREACHABLE"
    assert _finite_state(standard_world)


def test_position_only_orientation_policy(standard_world):
    """Position-only commands request the default approach orientation and
    fall back to position-only IK when it is infeasible (documented policy)."""
    w = standard_world
    res = w.set_arm_target(np.array([0.36, -0.20, TABLE_TOP_Z + 0.05]))
    assert res.success and w.robot.last_arm_command.orientation_policy == "default"
    # a high point where the tilted top-down orientation is not achievable
    res2 = w.set_arm_target(np.array([0.20, -0.45, 1.25]))
    assert res2.success and w.robot.last_arm_command.orientation_policy in ("default", "position_only")


def test_attachment_and_reset_with_g1(standard_world, env, oracle):
    stone = oracle.object_pos("stone")
    assert skills.approach(env, stone).success
    g = skills.grasp(env, oracle.object_pos("stone"))
    assert g.success, g
    assert env.is_attached() and standard_world.grasp_weld_active()
    # attached object rides with the palm point (no snap, stays within ATTACH_RADIUS)
    before = oracle.object_pos("stone") - env.get_ee_pos()
    lift = skills.move_to(env, env.get_ee_pos() + np.array([0.0, 0.0, 0.10]))
    assert lift.success, lift
    after = oracle.object_pos("stone") - env.get_ee_pos()
    assert np.linalg.norm(after - before) < 0.02
    assert np.linalg.norm(after) <= ATTACH_RADIUS + 0.01
    # no hand/object penetration while holding
    pen = [c for c in standard_world.robot_contacts() if "stone" in c[:2] and c[2] < -0.003]
    assert pen == [], pen
    from tests.conftest import standard_scene

    standard_world.reset(standard_scene())
    assert not env.is_attached() and not standard_world.grasp_weld_active()
    assert standard_world.sim_time == 0.0
    assert np.allclose(standard_world.robot.arm_q(), [RIGHT_ARM_REST[j] for j in (
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")], atol=1e-9)
    assert np.allclose(standard_world.robot.waist_q(), [WAIST_REST[j] for j in (
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")], atol=1e-9)


def test_cameras_and_scene_variant_load():
    from core.world import SCENE_CAM_SYNC_XML, SimWorld

    w = SimWorld(SCENE_CAM_SYNC_XML)
    try:
        assert w.model.body("diag_ball").id > 0
        for cam in CAMERAS:
            assert w.model.camera(cam).id >= 0
        w.reset(SceneConfig(objects=[]))
        assert "diag_ball" not in w.active_objects()
        assert w.try_attach_near_ee() is None  # the ball is never an attachment candidate
    finally:
        w.close()
