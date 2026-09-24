"""Final2 3.2: an object whose box touches the TOP image edge is localized
from its visible lower part (pos_basis='depth_patch_partial_top').

Fixture: final RUN 5 f34_c_eval_09_bottle, A audit 17f75a43a18a frame 8 (a
SEARCH view). The bottle box [491, 0, 563, 66] touches the top edge; the
side-edge rule pushed the visible centre along the downward view ray and
reported 'clipped object centre is below the table top' in every SEARCH view,
so SEARCH was exhausted. The bottle was never touched in that trial, so its
spawn position (0.587, -0.296) is the expectation (test only, never used by A).
Figure: docs/night_run/figs/bottle_clipped.png."""
from dataclasses import replace
import json
import pathlib

import numpy as np
import pytest

from core.types import GroundStatus, Observation
from perception.depth_geometry import pixel_box
from perception.student_a import partial_object_center, partial_top_center
from tests.test_student_a_partial_object import make_a

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'top_clipped_bottle_frame'
TRUTH_XY = np.array([0.587, -0.296])


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


def test_side_rule_rejected_this_bottle(frame):
    obs, wire = frame
    assert bottle_box(wire)[1] == 0
    assert partial_object_center(obs, bottle_box(wire), 'bottle', 'green') == (
        None, 'clipped object centre is below the table top')


def test_top_clipped_bottle_is_localized_near_its_spawn(frame):
    obs, wire = frame
    pos, detail = partial_top_center(obs, bottle_box(wire), 'bottle', 'green')
    assert np.linalg.norm(np.array(pos[:2]) - TRUTH_XY) < 0.015
    assert pos[2] == pytest.approx(0.85 + 0.06, abs=0.01)  # resting height
    assert 'top-clipped' in detail


def test_describe_reports_the_top_basis(frame, tmp_path):
    obs, wire = frame
    bottle = next(g for g in make_a(tmp_path, [wire]).describe(obs).objects if g.name == 'bottle')
    assert bottle.status is GroundStatus.LOCALIZED
    assert bottle.attributes['pos_basis'] == 'depth_patch_partial_top'


def test_a_lifted_object_is_not_given_a_resting_position(frame):
    obs, wire = frame
    lifted = replace(obs, depth=obs.depth - 0.25)  # every point 25 cm nearer the camera: not on the table
    pos, detail = partial_top_center(lifted, bottle_box(wire), 'bottle', 'green')
    assert pos is None and 'table top' in detail


def test_never_for_the_held_instance(frame, tmp_path):
    obs, wire = frame
    a = make_a(tmp_path, [wire, wire])
    first = next(g for g in a.describe(obs).objects if g.name == 'bottle')
    a._held_id = first.instance_id
    second = next(g for g in a.describe(replace(obs, frame_id=9, sim_time=obs.sim_time + 1)).objects
                  if g.name == 'bottle')
    assert second.status is not GroundStatus.LOCALIZED or second.attributes.get('pos_basis') != 'depth_patch_partial_top'
