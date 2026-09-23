"""EXPERIMENTAL: a red region clipped by the image edge is still localized.

Fixture: a_demo frame 8 from the night run (tests/fixtures/region_edge_frame).
The VLM box [0,125,330,425] touches the left edge, so localize() returned
"box touches image boundary", the region stayed UNLOCALIZED, and a physically
correct PLACE could not be verified (c_2_01/02/08, smoke_2, a_demo).
The evaluator's ground-truth region centre for this frame is (0.4, 0.3, 0.853);
it is used here only as a test expectation, never by A.
"""
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from core.types import GroundStatus
from perception.depth_geometry import localize, pixel_box
from perception.student_a import partial_region_center

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'region_edge_frame'
GT_CENTER = (0.4, 0.3, 0.853)


@pytest.fixture(scope='module')
def frame():
    from PIL import Image
    meta = json.loads((FIXTURE / 'meta.json').read_text())
    obs = SimpleNamespace(rgb=np.array(Image.open(FIXTURE / 'rgb.png')),
                          depth=np.load(FIXTURE / 'depth.npz')['depth'],
                          intrinsics=np.array(meta['intrinsics']),
                          t_world_camera=np.array(meta['t_world_camera']),
                          frame_id=meta['frame_id'], sim_time=meta['sim_time'])
    region = next(d for d in meta['detections'] if d['name'] == 'red_region')
    return obs, region


def test_full_localize_still_refuses_the_clipped_box(frame):
    obs, region = frame
    pos, detail = localize(obs, pixel_box(region['bbox']), 'red_region', 'red')
    assert pos is None and detail == 'box touches image boundary'


def test_partial_fit_recovers_the_region_centre(frame):
    obs, region = frame
    pos, detail = partial_region_center(obs, pixel_box(region['bbox']))
    assert pos is not None, detail
    assert np.allclose(pos, GT_CENTER, atol=0.02), pos


def test_tracker_marks_the_partial_basis(frame):
    from perception.student_a import StudentAPerception
    obs, region = frame
    tracked = StudentAPerception._track(SimpleNamespace(_tracks={}, _next_id=0,
                                                        _new_id=lambda: 'a0'),
                                        [region], obs, 'fixture')
    g = tracked[0]
    assert g.status is GroundStatus.LOCALIZED
    assert g.attributes['pos_basis'] == 'depth_patch_partial'
    assert np.allclose(g.pos_world, GT_CENTER, atol=0.02)


def test_mostly_invalid_depth_patch_is_refused(frame):
    obs, region = frame
    blind = SimpleNamespace(**{**obs.__dict__, 'depth': np.zeros_like(obs.depth)})
    pos, detail = partial_region_center(blind, pixel_box(region['bbox']))
    assert pos is None and 'valid' in detail
