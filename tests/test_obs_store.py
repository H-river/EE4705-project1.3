# Owner: backbone (ALL)
"""ObservationStore tests: 32-frame window, persistence before eviction,
no aliasing of renderer buffers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.obs_store import ObservationStore
from core.types import IMAGE_HEIGHT, IMAGE_WIDTH, Observation


def make_obs(frame_id: int) -> Observation:
    return Observation(
        frame_id=frame_id,
        rgb=np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), frame_id % 256, dtype=np.uint8),
        depth=np.full((IMAGE_HEIGHT, IMAGE_WIDTH), float(frame_id), dtype=np.float32),
        intrinsics=np.eye(3),
        t_world_camera=np.eye(4),
        sim_time=float(frame_id) * 0.1,
        camera_name="onboard",
    )


def test_capacity_is_32_frames():
    store = ObservationStore()
    for i in range(40):
        store.put(make_obs(i))
    assert len(store) == 32
    assert store.get(7) is None  # evicted
    assert store.get(8) is not None and store.get(39) is not None
    assert store.latest().frame_id == 39


def test_marked_frames_persist_before_eviction(tmp_path):
    store = ObservationStore(persist_dir=tmp_path)
    store.put(make_obs(0))
    store.mark_persist(0)
    for i in range(1, 40):
        store.put(make_obs(i))
    # frame 0 was evicted but must be on disk
    assert store.get(0) is None
    rgb = tmp_path / "frame_000000_rgb.png"
    depth = tmp_path / "frame_000000_depth.npy"
    meta = tmp_path / "frame_000000_meta.json"
    assert rgb.exists() and depth.exists() and meta.exists()
    d = np.load(depth)
    assert d.shape == (IMAGE_HEIGHT, IMAGE_WIDTH) and float(d[0, 0]) == 0.0
    m = json.loads(meta.read_text())
    assert m["frame_id"] == 0 and m["camera_name"] == "onboard"
    assert len(m["intrinsics"]) == 3 and len(m["t_world_camera"]) == 4


def test_flush_persists_resident_marked_frames(tmp_path):
    store = ObservationStore(persist_dir=tmp_path)
    store.put(make_obs(5))
    store.mark_persist(5)
    files = store.flush()
    assert 5 in files
    assert (tmp_path / "frame_000005_rgb.png").exists()


def test_mark_unknown_frame_fails():
    store = ObservationStore()
    with pytest.raises(KeyError):
        store.mark_persist(123)


def test_duplicate_frame_id_rejected():
    store = ObservationStore()
    store.put(make_obs(1))
    with pytest.raises(ValueError):
        store.put(make_obs(1))


def test_clear_flushes_then_empties(tmp_path):
    store = ObservationStore(persist_dir=tmp_path)
    store.put(make_obs(2))
    store.mark_persist(2)
    store.clear()
    assert len(store) == 0
    assert (tmp_path / "frame_000002_rgb.png").exists()


def test_env_observations_do_not_alias_renderer_buffers(standard_world):
    """Two consecutive captures own distinct arrays, and mutating the sim
    afterwards does not change a stored frame."""
    from core.env import RobotEnv

    store = ObservationStore()
    env = RobotEnv(standard_world, store=store)
    a = env.get_obs()
    before = a.rgb.copy()
    b = env.get_obs()
    assert a.rgb is not b.rgb and a.depth is not b.depth
    standard_world.step(300)
    env.get_obs()  # re-render into the renderer's internal buffers
    assert np.array_equal(store.get(a.frame_id).rgb, before)
