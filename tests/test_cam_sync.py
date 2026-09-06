# Owner: backbone (ALL)
"""Multi-camera capture tests: atomicity, calibration validity, array
ownership, ID uniqueness, and camera-mount kinematics on the G1."""

from __future__ import annotations

import numpy as np
import pytest

from core.env import RobotEnv
from core.g1 import CAMERAS
from core.obs_store import ObservationStore
from core.types import IMAGE_HEIGHT, IMAGE_WIDTH
from core.world import SimWorld, resolve_camera

TABLE_TOP_Z = 0.85


def _rigid(t: np.ndarray) -> bool:
    r = t[:3, :3]
    return (np.allclose(r @ r.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(r), 1.0, atol=1e-9)
            and np.allclose(t[3], [0, 0, 0, 1]))


def test_multi_capture_shares_sim_time_and_capture_prefix(standard_world, env):
    obs = env.get_obs_multi(list(CAMERAS))
    assert set(obs) == set(CAMERAS)
    times = {o.sim_time for o in obs.values()}
    assert len(times) == 1 and times.pop() == standard_world.sim_time
    prefixes = {o.capture_id.rsplit("_", 1)[0] if not o.capture_id.endswith("_wrist") else o.capture_id.rsplit("_", 2)[0]
                for o in obs.values()}
    assert len(prefixes) == 1
    for cam, o in obs.items():
        assert o.camera_name == cam and o.capture_id.endswith("_" + cam)
        assert o.rgb.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3) and o.rgb.dtype == np.uint8
        assert o.depth.shape == (IMAGE_HEIGHT, IMAGE_WIDTH) and o.depth.dtype == np.float32


def test_capture_leaves_physical_state_and_time_unchanged(standard_world, env):
    standard_world.set_arm_target(np.array([0.36, -0.20, 0.95]))
    standard_world.step(100)  # mid-motion: velocities are non-zero
    qpos = standard_world.data.qpos.copy()
    qvel = standard_world.data.qvel.copy()
    act = standard_world.data.act.copy()
    ctrl = standard_world.data.ctrl.copy()
    t = standard_world.sim_time
    env.get_obs_multi(list(CAMERAS))
    env.get_obs()
    assert np.array_equal(standard_world.data.qpos, qpos)
    assert np.array_equal(standard_world.data.qvel, qvel)
    assert np.array_equal(standard_world.data.act, act)
    assert np.array_equal(standard_world.data.ctrl, ctrl)
    assert standard_world.sim_time == t


def test_camera_transforms_valid_and_distinct(standard_world, env):
    obs = env.get_obs_multi(list(CAMERAS))
    ts = {cam: o.t_world_camera for cam, o in obs.items()}
    for cam, t in ts.items():
        assert _rigid(t), cam
        assert np.all(np.isfinite(obs[cam].intrinsics)) and obs[cam].intrinsics[0, 0] > 0
    cams = list(CAMERAS)
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            assert np.linalg.norm(ts[cams[i]][:3, 3] - ts[cams[j]][:3, 3]) > 0.05, (cams[i], cams[j])


def test_arrays_do_not_alias_between_cameras_or_captures(standard_world):
    store = ObservationStore()
    env = RobotEnv(standard_world, store=store)
    a = env.get_obs_multi(list(CAMERAS))
    snap = {cam: (o.rgb.copy(), o.depth.copy()) for cam, o in a.items()}
    b = env.get_obs_multi(list(CAMERAS))
    for cam in CAMERAS:
        assert a[cam].rgb is not b[cam].rgb and a[cam].depth is not b[cam].depth
        assert not np.shares_memory(a[cam].rgb, b[cam].rgb)
    for i, c1 in enumerate(CAMERAS):
        for c2 in CAMERAS[i + 1:]:
            assert not np.shares_memory(a[c1].rgb, a[c2].rgb) and not np.shares_memory(a[c1].depth, a[c2].depth)
    standard_world.set_arm_target(np.array([0.36, -0.20, 0.95]))
    standard_world.step(400)
    env.get_obs_multi(list(CAMERAS))  # re-render into the renderer's buffers
    for cam in CAMERAS:
        assert np.array_equal(store.get(a[cam].frame_id).rgb, snap[cam][0])
        assert np.array_equal(store.get(a[cam].frame_id).depth, snap[cam][1], equal_nan=True)


def test_repeated_captures_without_stepping_have_unique_ids(standard_world, env):
    ids = []
    frame_ids = []
    for _ in range(3):
        obs = env.get_obs_multi(list(CAMERAS))
        ids += [o.capture_id for o in obs.values()]
        frame_ids += [o.frame_id for o in obs.values()]
        assert len({o.sim_time for o in obs.values()}) == 1
    assert len(set(ids)) == len(ids)
    assert len(set(frame_ids)) == len(frame_ids)
    single = env.get_obs("onboard")
    assert single.camera_name == "head" and single.capture_id not in ids


def test_get_obs_alias_and_bad_camera(standard_world, env):
    assert resolve_camera("onboard") == "head"
    assert env.get_obs("onboard").camera_name == env.get_obs("head").camera_name == "head"
    with pytest.raises(ValueError):
        env.get_obs("nonexistent_cam")
    with pytest.raises(ValueError):
        env.get_obs_multi(["head", "onboard"])  # duplicate after alias resolution


def test_right_arm_motion_moves_right_wrist_camera_only(standard_world, env):
    """Base and waist held: head and left-wrist transforms stay within small
    tolerances (the mocap-welded base is physically simulated, so not
    bitwise-constant) while the right-wrist camera moves with the arm."""
    before = {cam: o.t_world_camera for cam, o in env.get_obs_multi(list(CAMERAS)).items()}
    standard_world.set_arm_target(np.array([0.36, -0.20, 0.95]))
    standard_world.step(int(round(2.0 / standard_world.timestep)))
    after = {cam: o.t_world_camera for cam, o in env.get_obs_multi(list(CAMERAS)).items()}
    moved = np.linalg.norm(after["right_wrist"][:3, 3] - before["right_wrist"][:3, 3])
    assert moved > 0.05, moved
    for cam in ("head", "left_wrist"):
        dp = np.linalg.norm(after[cam][:3, 3] - before[cam][:3, 3])
        drot = np.degrees(np.arccos(np.clip((np.trace(after[cam][:3, :3].T @ before[cam][:3, :3]) - 1) / 2, -1, 1)))
        assert dp < 0.01, (cam, dp)  # 1 cm
        assert drot < 1.0, (cam, drot)  # 1 degree


def test_extrinsics_agree_with_model_camera_poses(standard_world, env):
    obs = env.get_obs_multi(list(CAMERAS))
    m, d = standard_world.model, standard_world.data
    for cam, o in obs.items():
        cid = m.camera(cam).id
        assert np.allclose(o.t_world_camera[:3, 3], d.cam_xpos[cid], atol=1e-9)
        r_gl = np.array(d.cam_xmat[cid]).reshape(3, 3)
        # public frame: +x right (= GL x), +y down (= -GL y), +z forward (= -GL z)
        assert np.allclose(o.t_world_camera[:3, 0], r_gl[:, 0], atol=1e-9)
        assert np.allclose(o.t_world_camera[:3, 1], -r_gl[:, 1], atol=1e-9)
        assert np.allclose(o.t_world_camera[:3, 2], -r_gl[:, 2], atol=1e-9)
        # mounting bodies as documented
    assert m.body(m.cam_bodyid[m.camera("head").id]).name == "torso_link"
    assert m.body(m.cam_bodyid[m.camera("left_wrist").id]).name == "left_wrist_yaw_link"
    assert m.body(m.cam_bodyid[m.camera("right_wrist").id]).name == "right_wrist_yaw_link"


def test_effective_clip_distances():
    w = SimWorld()
    try:
        near, far = w.effective_clip_distances()
    finally:
        w.close()
    assert abs(near - 0.005) < 1e-6
    assert abs(far - 10.0) < 1e-6


def _unproject_check(obs, pixels, plane_z, tol=0.01):
    K, T = obs.intrinsics, obs.t_world_camera
    checked = 0
    for (u, v) in pixels:
        dep = obs.depth[v, u]
        if not np.isfinite(dep):
            continue
        pc = np.array([(u - K[0, 2]) / K[0, 0] * dep, (v - K[1, 2]) / K[1, 1] * dep, dep, 1.0])
        pw = T @ pc
        assert abs(pw[2] - plane_z) < tol, f"{obs.camera_name} pixel ({u},{v}) -> z={pw[2]:.4f}, expected {plane_z}"
        checked += 1
    return checked


def test_depth_calibration_all_cameras(world, env):
    """Non-central pixels unproject onto known planes for EVERY camera.  The
    wrist cameras are pointed at the table with a test pose (they do not
    naturally see it at rest)."""
    from tests.conftest import standard_scene

    world.reset(standard_scene())
    world.step(int(round(1.0 / world.timestep)))
    # head: table plane
    head = env.get_obs("head")
    n = _unproject_check(head, [(550, 400), (100, 430), (320, 380), (200, 300)], TABLE_TOP_Z)
    assert n >= 3
    # right wrist: hold the palm 25 cm above the table looking down (approach axis = -z)
    world.set_arm_target(np.array([0.36, -0.20, TABLE_TOP_Z + 0.25]))
    world.step(int(round(3.0 / world.timestep)))
    rw = env.get_obs("right_wrist")
    seg = world.render_segmentation("right_wrist")
    table_geom = world.model.geom("table_top").id
    ys, xs = np.nonzero((seg[..., 0] == table_geom) & (seg[..., 1] == 5))  # mjOBJ_GEOM == 5
    assert len(xs) > 2000, "right wrist camera does not see the table in the test pose"
    idx = np.linspace(0, len(xs) - 1, 6).astype(int)
    pixels = [(int(xs[i]), int(ys[i])) for i in idx if abs(xs[i] - 319.5) > 40 or abs(ys[i] - 239.5) > 40]
    assert _unproject_check(rw, pixels, TABLE_TOP_Z) >= 3
    # left wrist: hangs beside the body; use the FLOOR plane (z = 0) as calibration geometry
    lw = env.get_obs("left_wrist")
    seg = world.render_segmentation("left_wrist")
    floor_geom = world.model.geom("floor").id
    ys, xs = np.nonzero((seg[..., 0] == floor_geom) & (seg[..., 1] == 5))
    assert len(xs) > 2000, "left wrist camera does not see the floor in the rest pose"
    idx = np.linspace(0, len(xs) - 1, 6).astype(int)
    pixels = [(int(xs[i]), int(ys[i])) for i in idx if abs(xs[i] - 319.5) > 40 or abs(ys[i] - 239.5) > 40]
    assert _unproject_check(lw, pixels, 0.0, tol=0.02) >= 3


def test_wrist_depth_quality_near_range(world, env):
    """Wrist-camera depth is valid (finite, > near clip) close to the hand
    and the invalid mapping applies only to background."""
    from tests.conftest import standard_scene

    world.reset(standard_scene())
    world.step(int(round(1.0 / world.timestep)))
    obs = env.get_obs("right_wrist")
    near, far = world.effective_clip_distances()
    finite = obs.depth[np.isfinite(obs.depth)]
    assert finite.size > 0
    assert finite.min() >= near and finite.max() < 0.98 * far
    assert finite.min() < 0.15, "expected hand geometry closer than 15 cm in the wrist view"
