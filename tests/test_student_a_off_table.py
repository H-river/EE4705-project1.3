"""Round 6 step 1a: detections whose 3D centre is off the table are dropped.

Fixture: night-run round 2, c_2_03_stone, A audit session 9336ffd31630,
frame 5 (a ground("stone") query during SEARCH). The head camera faced the
floor; the VLM reported two "gray stones" on the robot's shadow, which A
tracked as AMBIGUOUS a5/a6 and which later kept the real stone ambiguous.
"""
import json
import pathlib

import numpy as np
import pytest

from core.llm_client import FakeTransport, LLMClient, LLMConfig
from core.types import Observation
from core.scene_geometry import table_aabb, table_top_z
from perception.config import VisionConfig
from perception.depth_geometry import pixel_box
from perception.run import offline_observation, offline_wire
from perception.student_a import StudentAPerception, drop_off_table, estimate_center_z

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'phantom_shadow_frame'


@pytest.fixture(scope='module')
def phantom():
    from PIL import Image
    meta = json.loads((FIXTURE / 'meta.json').read_text())
    obs = Observation(meta['frame_id'], np.array(Image.open(FIXTURE / 'rgb.png')),
                      np.load(FIXTURE / 'depth.npz')['depth'], np.array(meta['intrinsics']),
                      np.array(meta['t_world_camera']), meta['sim_time'], meta['camera_name'])
    return obs, meta['wire']


def make_a(tmp_path, wire):
    transport = FakeTransport([FakeTransport.completion(json.dumps(wire))])
    config = VisionConfig(LLMConfig('test-vlm', 'https://fixture.invalid/v1', api_key='test-secret',
                                    max_retries=0, retry_backoff_s=0), str(tmp_path / 'audit'))
    return StudentAPerception(config, LLMClient(config.llm, transport=transport))


def test_table_top_comes_from_the_scene_file():
    lo, hi = table_aabb()
    assert table_top_z() == pytest.approx(0.85) and hi[2] > lo[2]


def test_shadow_stones_are_estimated_on_the_floor(phantom):
    obs, wire = phantom
    for d in wire['detections']:
        z, basis = estimate_center_z(obs, pixel_box(d['bbox']), d['name'], d['color'])
        assert z is not None and z < table_top_z() - 0.5, (d, z, basis)


def test_phantoms_are_dropped_with_reason_and_selection_is_remapped(phantom):
    obs, wire = phantom
    kept, dropped, _ = drop_off_table(wire, obs)
    assert kept['detections'] == [] and kept['selected'] == []
    assert [d['reason'] for d in dropped] == ['off_table', 'off_table']
    assert all(d['table_top_z'] == table_top_z() for d in dropped)


def test_ground_returns_nothing_and_the_audit_logs_the_drop(phantom, tmp_path):
    obs, wire = phantom
    a = make_a(tmp_path, wire)
    assert a.ground(obs, 'stone') is None
    audit = json.loads(pathlib.Path(a.last_diagnostics['audit_file']).read_text())
    assert [d['reason'] for d in audit['dropped_detections']] == ['off_table', 'off_table']
    assert audit['wire'] == wire  # the model's answer is kept as received
    assert a._tracks == {}  # no identity was created for a phantom


def test_table_objects_and_region_are_kept(tmp_path):
    obs, wire = offline_observation(), offline_wire()
    kept, dropped, _ = drop_off_table(wire, obs)
    assert dropped == [] and kept == wire
