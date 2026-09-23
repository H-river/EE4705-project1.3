"""Round 6 step 1b: SEARCH only looks where the table can be.

Every generated viewpoint's camera frustum must intersect the table's AABB
(checked with an exact separating-axis test, independent of C's own
projection check), from every start pose used by the smoke and
student_c_v2 trials. The rigid-camera prediction is checked against the
real rendered camera after turning.
"""
import itertools
import math
import pathlib

import numpy as np
import pytest
import yaml

from core.env import RobotEnv
from core.scene_geometry import table_aabb
from core.types import Action, ErrorCode, IMAGE_HEIGHT, IMAGE_WIDTH, Skill
from eval.runner import scene_config_from_spec
from executor.student_c import StudentCExecutor, search_views

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRIALS = sorted((ROOT / "eval/trials/smoke").glob("*.yaml")) + sorted((ROOT / "eval/trials/student_c_v2").glob("*.yaml"))


def frustum_vertices(t_world_camera, k, near=0.05, far=4.0):
    corners = [(0, 0), (IMAGE_WIDTH, 0), (IMAGE_WIDTH, IMAGE_HEIGHT), (0, IMAGE_HEIGHT)]
    rays = [np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0]) for u, v in corners]
    cam = [r * d for d in (near, far) for r in rays]
    return np.array([t_world_camera[:3, :3] @ p + t_world_camera[:3, 3] for p in cam])


def frustum_intersects_aabb(t_world_camera, k, lo, hi):
    """Separating-axis test between two convex polyhedra (exact)."""
    fv = frustum_vertices(t_world_camera, k)
    bv = np.array(list(itertools.product(*zip(lo, hi))))
    faces = [(0, 1, 2), (4, 5, 6), (0, 1, 5), (1, 2, 6), (2, 3, 7), (3, 0, 4)]
    normals = [np.cross(fv[b] - fv[a], fv[c] - fv[a]) for a, b, c in faces]
    fedges = [fv[1] - fv[0], fv[3] - fv[0]] + [fv[i + 4] - fv[i] for i in range(4)]
    bedges = list(np.eye(3))
    axes = normals + bedges + [np.cross(a, b) for a in fedges for b in bedges]
    for axis in axes:
        if np.linalg.norm(axis) < 1e-9:
            continue
        pf, pb = fv @ axis, bv @ axis
        if pf.max() < pb.min() - 1e-9 or pb.max() < pf.min() - 1e-9:
            return False
    return True


def test_separating_axis_check_itself():
    k = np.array([[415.7, 0, 319.5], [0, 415.7, 239.5], [0, 0, 1]])
    t = np.eye(4)  # camera at the origin looking along +z
    assert frustum_intersects_aabb(t, k, (-.1, -.1, 1.0), (.1, .1, 1.2))
    assert not frustum_intersects_aabb(t, k, (-.1, -.1, -2.0), (.1, .1, -1.0))  # behind
    assert not frustum_intersects_aabb(t, k, (5.0, -.1, 1.0), (5.2, .1, 1.2))  # far to the side


@pytest.fixture(scope="module")
def start_poses(world):
    """(trial, base_pose, head camera) for every start pose in the trial sets."""
    env, poses, seen = RobotEnv(world), [], set()
    for path in TRIALS:
        spec = yaml.safe_load(path.read_text())
        key = tuple(sorted((spec["scene"].get("robot_init") or {}).items()))
        if key in seen:
            continue
        seen.add(key)
        world.reset(scene_config_from_spec(spec["scene"]))
        world.step(100)
        obs = env.get_obs()
        poses.append((path.stem, env.get_base_pose().copy(), obs.t_world_camera.copy(), obs.intrinsics.copy()))
    return poses


def test_every_generated_view_frustum_intersects_the_table(start_poses):
    lo, hi = table_aabb()
    assert len(start_poses) >= 3
    for trial, base, cam, k in start_poses:
        views, info = search_views(base, cam, k, n_views=13)
        assert len(info["candidate_yaws"]) == 13 and len(views) + len(info["skipped_yaws"]) == 13
        assert views, trial
        assert info["pitch_sees_table"], trial
        for yaw, view_cam in views:
            assert frustum_intersects_aabb(view_cam, k, lo, hi), (trial, yaw)
            offset = math.atan2(math.sin(yaw - info["table_bearing_rad"]), math.cos(yaw - info["table_bearing_rad"]))
            assert abs(offset) <= math.radians(60) + 1e-9


def test_skipped_views_really_miss_the_table(start_poses):
    lo, hi = table_aabb()
    trial, base, cam, k = start_poses[0]
    # From far behind the table, most of a ±60° sweep sees no table at all.
    far_base = np.array([base[0] - 4.0, base[1] + 3.0, base[2]])
    far_cam = cam.copy()
    far_cam[:2, 3] += far_base[:2] - base[:2]
    views, info = search_views(far_base, far_cam, k, n_views=13)
    assert info["skipped_yaws"]
    for yaw in info["skipped_yaws"]:
        c, s = math.cos(yaw - base[2]), math.sin(yaw - base[2])
        rot = np.eye(4)
        rot[:2, :2] = [[c, -s], [s, c]]
        shifted = np.eye(4)
        shifted[:2, 3] = far_base[:2]
        back = np.eye(4)
        back[:2, 3] = -far_base[:2]
        view_cam = shifted @ rot @ back @ far_cam
        assert not frustum_intersects_aabb(view_cam, k, lo, hi), yaw


class NeverFound:
    def __init__(self):
        self.observations = []

    def ground(self, obs, target):
        self.observations.append(obs)
        return None

    def describe(self, obs, query=None):
        raise AssertionError("SEARCH must use ground()")

    def reset(self):
        pass


def test_search_observes_the_table_in_every_view(world):
    spec = yaml.safe_load((ROOT / "eval/trials/smoke/smoke_4_search.yaml").read_text())
    world.reset(scene_config_from_spec(spec["scene"]))
    world.step(500)
    env = RobotEnv(world)
    start = env.get_obs()
    predicted_views, _ = search_views(env.get_base_pose(), start.t_world_camera, start.intrinsics, n_views=13)
    perception = NeverFound()
    c = StudentCExecutor()
    c.reset()
    result = c.execute(Action(Skill.SEARCH, target="stone"), env, perception)
    assert result.error_code is ErrorCode.SEARCH_NOT_FOUND, result.info
    views = perception.observations
    assert 1 <= len(views) <= 13 and result.info["views"] == len(views)
    lo, hi = table_aabb()
    assert len(views) == len(predicted_views)
    for obs, (yaw, predicted_cam) in zip(views, predicted_views):
        k, t = obs.intrinsics, obs.t_world_camera
        valid = np.isfinite(obs.depth) & (obs.depth > 0)
        v, u = np.nonzero(valid)
        z = obs.depth[valid]
        pts = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z)) @ t[:3, :3].T + t[:3, 3]
        on_table = ((np.abs(pts[:, 2] - hi[2]) < 0.01) & (pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0])
                    & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1]))
        assert on_table.sum() > 500, (obs.frame_id, int(on_table.sum()))
        # The rendered camera heading is where search_views() predicted it.
        # (The base is accepted within BASE_YAW_TOL of the commanded yaw.) Only
        # the heading is checked: the torso leans forward during a sweep (the
        # head pitch drifted 0.87 -> 1.05 rad over 13 views in this scene), so
        # the full rotation is not rigid; the depth check above is what shows
        # the table stays in view.
        heading = lambda cam: math.atan2(cam[1, 2], cam[0, 2])
        err = heading(t) - heading(predicted_cam)
        assert abs(math.atan2(math.sin(err), math.cos(err))) < 0.1, (obs.frame_id, err)
