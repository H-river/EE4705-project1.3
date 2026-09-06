# Owner: backbone (ALL)
"""Robotiq 2F-85 gripper tests: control/settling, TCP reporting, weld
attachment on the gripper base, physical grasp, wrist-camera occlusion, and
task-pose reachability with the new end-effector."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core import skills
from core.env import RobotEnv
from core.g1 import (
    GRIPPER_BASE_BODY,
    RIGHT_EE_BODY,
    RIGHT_EE_SITE,
    approach_base_pose,
    approach_rotation,
)
from core.oracle import EvalOracle
from core.types import RobotState, SceneConfig, SceneObjectSpec
from core.world import ATTACH_RADIUS, PHYSICAL_GRASP_EXPERIMENTAL, SimWorld

TABLE_TOP_Z = 0.85
GRIPPER_SETTLE_S = 1.0  # declared: full stroke settles within this time


def _settle_gripper(world, side, target, timeout=GRIPPER_SETTLE_S, tol=0.03):
    t0 = world.sim_time
    while world.sim_time - t0 < timeout:
        world.step(10)
        if abs(world.gripper_opening()[side] - target) < tol and world.robot.gripper_driver_speed(side) < 0.05:
            return world.sim_time - t0
    return None


def test_gripper_opens_and_closes_within_range_and_settles(standard_world, env):
    w = standard_world
    for side in ("right", "left"):
        env.set_gripper(side, 0.0)
        t_close = _settle_gripper(w, side, 0.0)
        assert t_close is not None, f"{side} gripper did not close within {GRIPPER_SETTLE_S}s"
        ctrl = float(w.data.ctrl[w.robot._grip_act[side]])
        assert 0.0 <= ctrl <= 255.0
        assert w.robot.gripper_opening()[side] < 0.03
        env.set_gripper(side, 1.0)
        t_open = _settle_gripper(w, side, 1.0)
        assert t_open is not None, f"{side} gripper did not open within {GRIPPER_SETTLE_S}s"
        assert w.robot.gripper_opening()[side] > 0.97
        # partial target tracked by the MEASURED opening
        env.set_gripper(side, 0.5)
        assert _settle_gripper(w, side, 0.5, tol=0.08) is not None
        assert abs(env.get_robot_state().gripper_opening[side] - 0.5) < 0.08
        env.set_gripper(side, 1.0)
        _settle_gripper(w, side, 1.0)
    with pytest.raises(ValueError):
        env.set_gripper("right", 1.5)
    with pytest.raises(ValueError):
        env.set_gripper("middle", 0.5)
    assert w.robot.joint_limit_violations() == []


def test_set_gripper_is_non_blocking(standard_world, env):
    t = standard_world.sim_time
    env.set_gripper("right", 0.0)
    assert standard_world.sim_time == t
    env.set_gripper("right", 1.0)


def test_ee_site_is_pinch_and_robot_state_reports_it(standard_world, env):
    w = standard_world
    assert RIGHT_EE_SITE == "rg_pinch"
    assert w.model.body(w.model.site_bodyid[w.model.site("rg_pinch").id]).name == RIGHT_EE_BODY == GRIPPER_BASE_BODY["right"]
    state = env.get_robot_state()
    assert isinstance(state, RobotState)
    assert np.allclose(state.ee_pos, w.data.site("rg_pinch").xpos, atol=1e-9)
    assert np.allclose(state.ee_pos, env.get_ee_pos(), atol=1e-9)
    assert set(state.gripper_opening) == {"right", "left"}
    assert all(0.0 <= v <= 1.0 for v in state.gripper_opening.values())
    assert abs(np.linalg.norm(state.ee_quat) - 1.0) < 1e-9
    assert state.sim_time == w.sim_time and not state.attached
    # the pinch point lies between the pads: equidistant from the two pad bodies
    pads = [w.data.body(n).xpos for n in ("rg_right_pad", "rg_left_pad")]
    d = [np.linalg.norm(p - np.array(state.ee_pos)) for p in pads]
    assert abs(d[0] - d[1]) < 0.005


def test_weld_attach_on_gripper_base_follows_lift_and_base_move(standard_world, env, oracle):
    w = standard_world
    assert w.grasp_mode == "weld"
    stone = oracle.object_pos("stone")
    assert skills.approach(env, stone).success
    g = skills.grasp(env, oracle.object_pos("stone"))
    assert g.success, g
    assert env.is_attached() and w.grasp_weld_active()
    eq = w.model.equality("grasp_stone")
    assert w.model.body(w.model.eq_obj1id[eq.id]).name == "rg_base"
    rel0 = oracle.object_pos("stone") - env.get_ee_pos()
    assert np.linalg.norm(rel0) <= ATTACH_RADIUS + 1e-6
    # cosmetic close must not push the welded object
    env.set_gripper("right", 0.0)
    w.step(int(round(1.0 / w.timestep)))
    assert np.linalg.norm((oracle.object_pos("stone") - env.get_ee_pos()) - rel0) < 0.005
    # lift and base move: object follows
    assert skills.move_to(env, env.get_ee_pos() + np.array([0.0, 0.0, 0.12])).success
    assert np.linalg.norm((oracle.object_pos("stone") - env.get_ee_pos()) - rel0) < 0.01
    b = w.base_pose()
    env.set_base_target(b[0] + 0.15, b[1] + 0.10, b[2])
    w.step(int(round(2.0 / w.timestep)))
    assert np.linalg.norm((oracle.object_pos("stone") - env.get_ee_pos()) - rel0) < 0.01
    # detach releases it (it falls / rests on the table afterwards)
    z_held = oracle.object_pos("stone")[2]
    assert env.detach() is True
    assert not env.is_attached() and not w.grasp_weld_active()
    w.step(int(round(1.0 / w.timestep)))
    assert oracle.object_pos("stone")[2] < z_held - 0.05
    env.set_gripper("right", 1.0)


@pytest.mark.skipif(PHYSICAL_GRASP_EXPERIMENTAL, reason="physical grasp mode shipped as experimental (see docs/DECISIONS.md §12)")
def test_physical_grasp_cube_from_canonical_pose():
    w = SimWorld(grasp_mode="physical")
    env = RobotEnv(w)
    oracle = EvalOracle(w)
    try:
        w.reset(SceneConfig(objects=[SceneObjectSpec("cube", (0.40, 0.15, 0.88))]))
        w.step(int(round(1.0 / w.timestep)))
        cube = oracle.object_pos("cube")
        assert skills.approach(env, cube).success
        g = skills.grasp(env, oracle.object_pos("cube"))
        assert g.success, (g, w.last_attach_reason)
        assert w.last_attach_reason == "attached" and not w.grasp_weld_active()
        z0 = oracle.object_pos("cube")[2]
        assert skills.move_to(env, env.get_ee_pos() + np.array([0.0, 0.0, 0.10])).success
        w.step(int(round(0.5 / w.timestep)))
        assert env.is_attached(), w.last_attach_reason
        assert oracle.object_pos("cube")[2] - z0 > 0.06
        assert env.detach() is True
        w.step(int(round(1.0 / w.timestep)))
        assert w.gripper_opening()["right"] > 0.9
    finally:
        w.close()


def test_physical_mode_reports_distinct_failure_reasons():
    w = SimWorld(grasp_mode="physical")
    env = RobotEnv(w)
    try:
        w.reset(SceneConfig(objects=[SceneObjectSpec("cube", (0.40, 0.15, 0.88))]))
        w.step(int(round(1.0 / w.timestep)))
        assert env.try_attach_near_ee() is None and w.last_attach_reason == "nothing_in_range"
        state = env.get_robot_state()
        assert state.last_attach_reason == "nothing_in_range" and not state.attached
    finally:
        w.close()


def test_wrist_camera_center_is_not_gripper_open_and_closed(standard_world, env):
    """At the ready posture, for the open AND closed gripper, the central
    20% of both wrist images shows no gripper geometry and its median depth
    is scene depth (> 0.3 m), while the finger pads are in the bottom of
    the frame (later alignment logic uses them as a reference)."""
    w = standard_world
    m = w.model
    for cam, prefix in (("right_wrist", "rg_"), ("left_wrist", "lg_")):
        side = "right" if cam == "right_wrist" else "left"
        grip_bodies = [b for b in range(m.nbody) if m.body(b).name.startswith(prefix)]
        for opening in (1.0, 0.0):
            env.set_gripper(side, opening)
            _settle_gripper(w, side, opening)
            seg = w.render_segmentation(cam)
            body = np.full(seg.shape[:2], -1)
            isg = seg[..., 1] == 5
            for gid in np.unique(seg[..., 0][isg]):
                body[isg & (seg[..., 0] == gid)] = m.geom_bodyid[gid]
            grip = np.isin(body, grip_bodies)
            H, W = grip.shape
            center = grip[int(0.4 * H):int(0.6 * H), int(0.4 * W):int(0.4 * W) + int(0.2 * W)]
            assert center.mean() == 0.0, (cam, opening, center.mean())
            assert grip[int(0.75 * H):, :].mean() > 0.3, (cam, opening, "pads not at the bottom")
            depth = env.get_obs(cam).depth
            cd = depth[int(0.4 * H):int(0.6 * H), int(0.4 * W):int(0.6 * W)]
            assert np.nanmedian(cd) > 0.3, (cam, opening, np.nanmedian(cd))
        env.set_gripper(side, 1.0)
        _settle_gripper(w, side, 1.0)


def test_task_poses_tracked_collision_free(world):
    """approach / grasp / lift / place poses of the smoke_1 layout with the
    2F-85 TCP: pose-constrained tracking < 1 cm / 5 deg, no penetration."""
    scene = SceneConfig(objects=[SceneObjectSpec("stone", (0.40, -0.15, 0.88)), SceneObjectSpec("cube", (0.40, 0.15, 0.88)),
                                 SceneObjectSpec("bottle", (0.55, -0.35, 0.915))])
    world.reset(scene)
    world.step(int(round(1.0 / world.timestep)))
    stone = world.body_pos("stone")
    bx, by, byaw = approach_base_pose(stone[:2], world.base_pose()[:2])
    world.teleport_base(bx, by, byaw)
    world.step(int(round(0.5 / world.timestep)))
    grasp = stone + np.array([0.0, 0.0, skills.GRASP_DESCEND_OFFSET])
    poses = [("approach", grasp + [0, 0, skills.PRE_REACH_HEIGHT], byaw), ("grasp", grasp, byaw), ("lift", grasp + [0, 0, 0.15], byaw)]
    region = np.array([0.40, 0.30, TABLE_TOP_Z])
    for name, p, yaw in poses:
        _track(world, name, p, approach_rotation(yaw))
    px, py, pyaw = approach_base_pose(region[:2], world.base_pose()[:2])
    world.teleport_base(px, py, pyaw)
    world.step(int(round(0.5 / world.timestep)))
    _track(world, "place", region + [0, 0, 0.05], approach_rotation(pyaw))


def _track(world, name, p, rot):
    res = world.set_arm_target(p, rot)
    assert res.success, name
    deadline = world.sim_time + 6.0
    while world.sim_time < deadline:
        world.step(10)
        pe, re = world.robot.tracking_error()
        if pe < 0.01 and re < math.radians(5) and np.max(np.abs(world.robot.arm_qvel())) < 0.05:
            break
    pe, re = world.robot.tracking_error()
    assert pe < 0.01 and math.degrees(re) < 5.0, (name, pe, math.degrees(re))
    bad = [c for c in world.robot_contacts() if not any(o in c[:2] for o in ("stone", "cube", "bottle", "stone2"))]
    assert bad == [], (name, bad)


def test_contact_masks_and_linkage_survive_weld_and_reset(standard_world, env, oracle):
    """Cosmetic closing excludes only gripper contacts with the held object;
    grippers retain mutual collisions and all six linkage equalities."""
    w = standard_world
    m = w.model
    def eligible(a, b):
        ga, gb = m.geom(a).id, m.geom(b).id
        return bool((m.geom_contype[ga] & m.geom_conaffinity[gb]) or
                    (m.geom_contype[gb] & m.geom_conaffinity[ga]))
    def linkage_active():
        for prefix in ("rg_", "lg_"):
            eqs = [i for i in range(m.neq) if m.equality(i).name.startswith(prefix)]
            assert len(eqs) == 3
            assert all(w.data.eq_active[i] for i in eqs)
    linkage_active()
    assert eligible("rg_right_pad1", "lg_right_pad1")
    assert eligible("rg_right_pad1", "rg_left_pad1")
    assert eligible("rg_right_pad1", "table_top")
    assert eligible("rg_right_pad1", "stone_geom")
    assert skills.approach(env, oracle.object_pos("stone")).success
    assert skills.grasp(env, oracle.object_pos("stone")).success
    assert not eligible("rg_right_pad1", "stone_geom")
    assert eligible("stone_geom", "table_top")
    assert eligible("stone_geom", "cube_geom")
    linkage_active()
    assert env.detach()
    assert eligible("rg_right_pad1", "stone_geom")
    assert env.try_attach_near_ee() is not None
    from tests.conftest import standard_scene
    w.reset(standard_scene())
    assert eligible("rg_right_pad1", "stone_geom")
    linkage_active()


def test_gripper_control_rate_limit(standard_world, env):
    w = standard_world
    for opening in (0.0, 1.0):
        env.set_gripper("right", opening)
        prev = float(w.data.ctrl[w.robot._grip_act["right"]])
        for _ in range(250):
            w.step()
            ctrl = float(w.data.ctrl[w.robot._grip_act["right"]])
            assert abs(ctrl - prev) <= 255.0 / 0.8 * w.robot.limits.gripper_driver_speed * w.timestep + 1e-10
            assert 0 <= ctrl <= 255
            prev = ctrl
    for invalid in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError):
            env.set_gripper("right", invalid)
