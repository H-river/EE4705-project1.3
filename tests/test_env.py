# Owner: backbone (ALL)
"""Real-environment acceptance tests (RobotEnv + reference skills; no
TeleportExecutor).  These exercise actual simulated motion, rendering
geometry, and attachment physics.  They must run offline and must not be
skipped: a failure here is a real environment failure."""

from __future__ import annotations

import numpy as np

from core import skills
from core.env import RobotEnv
from core.types import IMAGE_HEIGHT, IMAGE_WIDTH
from core.world import ATTACH_RADIUS

TABLE_TOP_Z = 0.85  # m, top surface of the table in assets/scene_common.xml


def test_rgbd_shapes_and_calibration(standard_world, env):
    obs = env.get_obs()
    assert obs.rgb.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3) and obs.rgb.dtype == np.uint8
    assert obs.depth.shape == (IMAGE_HEIGHT, IMAGE_WIDTH) and obs.depth.dtype == np.float32
    K = obs.intrinsics
    assert K.shape == (3, 3) and np.all(np.isfinite(K)) and K[0, 0] > 0 and K[1, 1] > 0
    T = obs.t_world_camera
    assert T.shape == (4, 4)
    R = T[:3, :3]
    # valid rigid rotation
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    assert np.allclose(T[3], [0, 0, 0, 1])
    # depth has NaN for background (sky), finite values on the table
    assert np.isnan(obs.depth[0, 0])  # top corner: sky
    assert np.isfinite(obs.depth[430, 320])  # lower center: table


def test_known_distance_depth_and_projection(standard_world, env):
    """Unproject NON-CENTRAL table pixels through K, depth and
    T_world_camera: they must land on the table plane z = TABLE_TOP_Z."""
    obs = env.get_obs()
    K, T = obs.intrinsics, obs.t_world_camera
    for (u, v) in ((550, 400), (100, 430), (320, 380)):
        d = obs.depth[v, u]
        assert np.isfinite(d)
        pc = np.array([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d, 1.0])
        pw = T @ pc
        assert abs(pw[2] - TABLE_TOP_Z) < 0.01, f"pixel ({u},{v}) unprojected to z={pw[2]:.3f}"

    # Aligned RGB: those table pixels must show the brown table color
    r, g, b = env.get_obs().rgb[400, 550].astype(int)
    assert r > g > b, "expected table (brownish) color at a table pixel"


def test_rgbd_same_state_alignment(standard_world, env):
    """RGB and depth come from the same simulation state: sim time must not
    advance during a capture."""
    t0 = standard_world.sim_time
    env.get_obs()
    assert standard_world.sim_time == t0


def test_attach_requires_proximity(standard_world, env, oracle):
    stone = oracle.object_pos("stone")
    # Far above the object: outside the attachment threshold -> no attach
    r = skills.reach(env, stone + [0.0, 0.0, ATTACH_RADIUS + 0.12])
    assert r.success
    assert env.try_attach_near_ee() is None
    assert not env.is_attached()
    # At the object: attach succeeds and returns an OPAQUE handle
    r = skills.reach(env, stone)
    assert r.success
    handle = env.try_attach_near_ee()
    assert handle is not None
    assert handle not in ("stone", "stone2", "cube", "bottle"), "handle leaks ground-truth identity"
    assert env.is_attached()
    # only one simultaneous attachment is supported
    assert env.try_attach_near_ee() is None
    env.detach()


def test_attach_does_not_snap_object(standard_world, env, oracle):
    stone = oracle.object_pos("stone")
    assert skills.reach(env, stone).success
    before = oracle.object_pos("stone")
    env.try_attach_near_ee()
    env.step(50)
    after = oracle.object_pos("stone")
    assert np.linalg.norm(after - before) < 0.01, "attachment snapped the object"
    env.detach()


def test_detach_and_reset_clear_attachment(standard_world, env, oracle):
    stone = oracle.object_pos("stone")
    assert skills.reach(env, stone).success
    assert env.try_attach_near_ee() is not None
    assert env.detach() is True
    assert not env.is_attached()
    assert env.detach() is False  # nothing left to detach
    # attach again, then reset: attachment state fully cleared
    assert skills.reach(env, oracle.object_pos("stone")).success
    assert env.try_attach_near_ee() is not None
    from tests.conftest import standard_scene

    standard_world.reset(standard_scene())
    assert not env.is_attached()
    assert standard_world.attached_body_name() is None
    # every GRASP weld inactive (the permanent base weld of the G1 stays on)
    assert not standard_world.grasp_weld_active()
    assert standard_world.sim_time == 0.0


def test_real_skills_pick_and_place(standard_world, env, oracle):
    """Full pick-carry-place cycle through the REAL environment adapters
    (bounded motion, no teleporting), judged by the oracle."""
    stone = oracle.object_pos("stone")
    assert skills.approach(env, stone).success
    g = skills.grasp(env, oracle.object_pos("stone"))
    assert g.success, g
    assert env.is_attached()
    b = oracle.region_bounds()
    target = np.array([b.center_xy[0], b.center_xy[1], b.support_z + 0.16])
    m = skills.move_to(env, target)
    assert m.success, m
    # lower before releasing so the object does not bounce/roll away
    lower = skills.move_to(env, np.array([b.center_xy[0], b.center_xy[1], b.support_z + 0.06]))
    assert lower.success, lower
    p = skills.place(env)
    assert p.success, p
    assert not env.is_attached()
    assert oracle.object_in_region("stone")
    stable, detail = oracle.check_stability("stone")
    assert stable, detail


def test_ordinary_env_paths_never_teleport(standard_world, env, oracle):
    """Position discontinuities do not occur under RobotEnv stepping."""
    pos_before = oracle.object_pos("cube")
    env.set_base_target(0.1, 0.0, 0.2)
    prev = oracle.object_pos("cube")
    for _ in range(20):
        env.step(10)
        cur = oracle.object_pos("cube")
        assert np.linalg.norm(cur - prev) < 0.05, "object jumped during ordinary stepping"
        prev = cur
    assert np.linalg.norm(oracle.object_pos("cube") - pos_before) < 0.05


def test_env_exposes_no_ground_truth():
    from core.env import RobotEnv

    public = {name for name in dir(RobotEnv) if not name.startswith("_")}
    forbidden = {"get_state", "object_pos", "attached_body_name", "model", "data",
                 "oracle", "body_pos", "teleport_body"}
    assert not (public & forbidden), f"RobotEnv leaks ground truth: {public & forbidden}"


def test_cleanup_idempotent():
    from core.world import SimWorld
    from tests.conftest import standard_scene

    w = SimWorld()
    w.reset(standard_scene())
    w.render_rgbd()  # allocate the renderer
    w.close()
    w.close()  # second close is safe
