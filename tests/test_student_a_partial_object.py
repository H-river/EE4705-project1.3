"""Round 6 step 2: an object whose box is clipped by the image edge is localized.

Fixture: round 5, c_2_09_bottle, A audit session 0402adf6d323, frame 6: the
describe() right after APPROACH. The bottle box [526, 232, 640, 375] touches
the right edge, so the bottle was UNLOCALIZED and the next GRASP failed with
TARGET_LOST. The bottle had not moved since the trial started, so the trial
YAML's spawn position (0.5, -0.25) on the 0.85 m table top, centre 0.06 m above
it, is the expected centre. It is used only as a test expectation, never by A.
"""
from dataclasses import replace
import json
import pathlib

import numpy as np
import pytest

from core.llm_client import FakeTransport, LLMClient, LLMConfig
from core.types import GroundStatus, Observation
from perception.config import VisionConfig
from perception.depth_geometry import localize, pixel_box
from perception.student_a import StudentAPerception, partial_object_center

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'clipped_bottle_frame'
EXPECTED_BOTTLE = np.array([0.5, -0.25, 0.91])


@pytest.fixture(scope='module')
def frame():
    from PIL import Image
    meta = json.loads((FIXTURE / 'meta.json').read_text())
    obs = Observation(meta['frame_id'], np.array(Image.open(FIXTURE / 'rgb.png')),
                      np.load(FIXTURE / 'depth.npz')['depth'], np.array(meta['intrinsics']),
                      np.array(meta['t_world_camera']), meta['sim_time'], meta['camera_name'])
    return obs, meta['wire']


def bottle_box(wire):
    return pixel_box(next(d for d in wire['detections'] if d['name'] == 'bottle')['bbox'])


def make_a(tmp_path, wires):
    transport = FakeTransport([FakeTransport.completion(json.dumps(w)) for w in wires])
    config = VisionConfig(LLMConfig('test-vlm', 'https://fixture.invalid/v1', api_key='test-secret',
                                    max_retries=0, retry_backoff_s=0), str(tmp_path / 'audit'))
    return StudentAPerception(config, LLMClient(config.llm, transport=transport))


def test_full_fit_still_refuses_the_clipped_bottle(frame):
    obs, wire = frame
    assert localize(obs, bottle_box(wire), 'bottle', 'green') == (None, 'box touches image boundary')


def test_clipped_bottle_is_localized_near_its_true_centre(frame, tmp_path):
    obs, wire = frame
    scene = make_a(tmp_path, [wire]).describe(obs)
    bottle = next(g for g in scene.objects if g.name == 'bottle')
    assert bottle.status is GroundStatus.LOCALIZED
    assert bottle.attributes['pos_basis'] == 'depth_patch_partial'
    assert np.linalg.norm(np.asarray(bottle.pos_world) - EXPECTED_BOTTLE) < 0.02


def test_never_for_the_held_instance(frame, tmp_path):
    obs, wire = frame
    a = make_a(tmp_path, [wire, wire])
    first = next(g for g in a.describe(obs).objects if g.name == 'bottle')
    a._held_id = first.instance_id
    rgb = obs.rgb.copy()
    rgb[0, 0, 0] ^= 1  # new image bytes -> a new model call with the same answer
    second = a.describe(replace(obs, rgb=rgb, frame_id=obs.frame_id + 1))
    bottle = next(g for g in second.objects if g.name == 'bottle')
    assert bottle.instance_id == first.instance_id
    assert bottle.status is GroundStatus.UNLOCALIZED and 'pos_basis' not in bottle.attributes


def test_a_lifted_object_is_not_partially_localized(frame):
    obs, wire = frame
    t = obs.t_world_camera.copy()
    t[2, 3] += 0.20  # the same image seen 20 cm higher: the bottle would be off the table top
    pos, detail = partial_object_center(replace(obs, t_world_camera=t), bottle_box(wire), 'bottle', 'green')
    assert pos is None and 'not resting on the table' in detail


def test_mostly_invalid_depth_is_refused(frame):
    obs, wire = frame
    x1, y1, x2, y2 = bottle_box(wire)
    depth = obs.depth.copy()
    depth[y1:y1 + int(0.6 * (y2 - y1)), x1:x2] = np.nan
    pos, detail = partial_object_center(replace(obs, depth=depth), (x1, y1, x2, y2), 'bottle', 'green')
    assert pos is None and 'valid' in detail
