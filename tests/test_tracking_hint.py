"""Round 6 step 3: TrackingHint (contract v3) keeps identities across grasp/release.

(i) smoke_4 run 3 (round 3, runs/night/r3_c): after a failed PLACE the released
gray stone was re-identified a11 -> a16, and B's contract (goal pinned to a11)
rejected every replan. Fixture: A audit session 7509bebc9df5, frame 24, with
the tracker state A had just before it (reconstructed from the same session's
audits: a11 at the stone's pre-grasp position, position-less phantom gray-stone
tracks a3/a4/a7/a10 from floor views, a15 from the clipped frame 23).
(ii) a held stone keeps its id across 5 frames while the base moves.
"""
import json
import pathlib

import numpy as np
import pytest

from core.interfaces import accepts_hint, describe_with_hint
from core.llm_client import FakeTransport, LLMClient, LLMConfig
from core.orchestrator import Orchestrator
from core.types import (CONTRACT_VERSION, ExecutionContext, GroundStatus, Observation, ReleasedHint,
                        SceneDescription, TrackingHint)
from perception.config import VisionConfig
from perception.run import offline_observation, offline_wire
from perception.student_a import StudentAPerception

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'reid_after_place_frame'
GRAY = ('stone', 'gray')


def make_a(tmp_path, wires):
    transport = FakeTransport([FakeTransport.completion(json.dumps(w)) for w in wires])
    config = VisionConfig(LLMConfig('test-vlm', 'https://fixture.invalid/v1', api_key='test-secret',
                                    max_retries=0, retry_backoff_s=0), str(tmp_path / 'audit'))
    return StudentAPerception(config, LLMClient(config.llm, transport=transport))


def track(key, pos):
    return {'key': key, 'pos': None if pos is None else np.array(pos),
            'last_localized_frame': None if pos is None else 18, 'last_localized_sim_time': None}


@pytest.fixture(scope='module')
def reid_frame():
    from PIL import Image
    meta = json.loads((FIXTURE / 'meta.json').read_text())
    obs = Observation(meta['frame_id'], np.array(Image.open(FIXTURE / 'rgb.png')),
                      np.load(FIXTURE / 'depth.npz')['depth'], np.array(meta['intrinsics']),
                      np.array(meta['t_world_camera']), meta['sim_time'], meta['camera_name'])
    return obs, meta['wire']


def a_before_frame_24(tmp_path, wire):
    a = make_a(tmp_path, [wire])
    a._tracks = {k: track(GRAY, None) for k in ('a3', 'a4', 'a7', 'a10', 'a15')}
    a._tracks.update(a11=track(GRAY, (0.4, -0.15, 0.875)),
                     a12=track(('bottle', 'green'), None),
                     a13=track(('red_region', 'red'), (0.3999, 0.3016, 0.8532)),
                     a14=track(('cube', 'blue'), (0.4216, 0.1690, 0.8751)))
    a._next_id = 16
    return a


def test_contract_version_bumped():
    assert CONTRACT_VERSION >= 3  # v3 added the hint (v4: REJECTED plans)


def test_without_hint_the_released_stone_is_reidentified(reid_frame, tmp_path):
    obs, wire = reid_frame
    scene = a_before_frame_24(tmp_path, wire).describe(obs)
    stone = next(g for g in scene.objects if g.name == 'stone')
    assert stone.instance_id == 'a16'  # the round-3 failure, reproduced


def test_released_hint_keeps_the_id_after_place(reid_frame, tmp_path):
    obs, wire = reid_frame
    a = a_before_frame_24(tmp_path, wire)
    # Gripper right after the release, above the red region.
    hint = TrackingHint(released=ReleasedHint('a11', (0.40, 0.30, 0.99)))
    scene = a.describe(obs, hint=hint)
    stone = next(g for g in scene.objects if g.name == 'stone')
    assert stone.instance_id == 'a11' and stone.status is GroundStatus.LOCALIZED
    assert np.linalg.norm(np.asarray(stone.pos_world)[:2] - (0.389, 0.268)) < 0.01
    assert {g.instance_id for g in scene.objects} == {'a11', 'a14'}


def test_held_stone_keeps_its_id_across_five_frames_while_the_base_moves(tmp_path, monkeypatch):
    import perception.student_a as module
    obs = offline_observation()
    stone = dict(next(d for d in offline_wire()['detections'] if d['name'] == 'stone'))
    wires = [{'detections': [dict(stone)], 'selected': [], 'answer': ''} for _ in range(6)]
    # The stone is carried: each frame it is 0.2 m further along y (beyond the
    # 15 cm association gate), as the base drives with it in the gripper.
    path = [(0.40, -0.15 + 0.2 * k, 0.95) for k in range(6)]
    fits = iter([(p, 'test depth') for p in path])
    monkeypatch.setattr(module, 'localize', lambda *args: next(fits))

    def frame(k):
        rgb = obs.rgb.copy()
        rgb[0, 0, 0] = k  # new bytes -> a new model call
        return Observation(k, rgb, obs.depth, obs.intrinsics, obs.t_world_camera, float(k), 'head')

    a = make_a(tmp_path, wires)
    first = a.describe(frame(0))
    held = first.objects[0].instance_id
    a._tracks['a_phantom'] = track(GRAY, None)  # an earlier phantom duplicate of the same class/colour
    ids = []
    for k in range(1, 6):
        gripper = (path[k][0], path[k][1], path[k][2] + 0.05)
        scene = a.describe(frame(k), hint=TrackingHint(held_object_id=held, held_pos_world=gripper))
        (g,) = scene.objects
        ids.append(g.instance_id)
        assert g.attributes['pos_basis'] == 'held_hint' and g.pos_world == pytest.approx(gripper)
    assert ids == [held] * 5


def test_held_stone_without_hint_loses_its_id(tmp_path, monkeypatch):
    import perception.student_a as module
    obs = offline_observation()
    stone = dict(next(d for d in offline_wire()['detections'] if d['name'] == 'stone'))
    wires = [{'detections': [dict(stone)], 'selected': [], 'answer': ''} for _ in range(2)]
    fits = iter([((0.40, -0.15, 0.95), 'test depth'), ((0.40, 0.05, 0.95), 'test depth')])
    monkeypatch.setattr(module, 'localize', lambda *args: next(fits))
    a = make_a(tmp_path, wires)
    held = a.describe(obs).objects[0].instance_id
    a._tracks['a_phantom'] = track(GRAY, None)
    rgb = obs.rgb.copy()
    rgb[0, 0, 0] = 1
    moved = Observation(1, rgb, obs.depth, obs.intrinsics, obs.t_world_camera, 1.0, 'head')
    assert a.describe(moved).objects[0].instance_id != held


class OldPerception:
    def describe(self, obs):
        return SceneDescription(frame_id=obs)


class NewPerception:
    def __init__(self):
        self.hints = []

    def describe(self, obs, query=None, *, hint=None):
        self.hints.append(hint)
        return SceneDescription(frame_id=obs)


def test_hint_is_passed_only_to_perceptions_that_take_it():
    hint = TrackingHint(held_object_id='a1', held_pos_world=(0, 0, 1))
    assert not accepts_hint(OldPerception()) and accepts_hint(NewPerception())
    assert describe_with_hint(OldPerception(), 7, hint=hint).frame_id == 7
    new = NewPerception()
    describe_with_hint(new, 7, hint=hint)
    assert new.hints == [hint]


class FakeEnv:
    def get_ee_pos(self):
        return np.array([0.4, 0.3, 1.0])


def test_orchestrator_builds_the_hint_from_its_execution_context():
    orch = Orchestrator.__new__(Orchestrator)
    orch.env = FakeEnv()
    ctx = ExecutionContext(scene=SceneDescription(), held_instance_id='a2')
    hint = orch._tracking_hint(ctx, None)
    assert hint.held_object_id == 'a2' and hint.held_pos_world == (0.4, 0.3, 1.0)
    released = ReleasedHint('a2', (0.4, 0.3, 1.0))
    ctx = ExecutionContext(scene=SceneDescription(), last_release_instance_id='a2')
    hint = orch._tracking_hint(ctx, released)
    assert hint.held_object_id is None and hint.released == released
    ctx = ExecutionContext(scene=SceneDescription(), held_instance_id='a2', last_release_instance_id='a2')
    assert orch._tracking_hint(ctx, released).released is None  # grasped again: no longer released
