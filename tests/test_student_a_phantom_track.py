"""A phantom duplicate in one frame must not make the class AMBIGUOUS forever.

From smoke_4 in the night run (runs/night/audit_a/80e310445d48): the VLM
reported TWO gray stones in the first search view (only one exists). Both were
UNLOCALIZED, so A created two position-less tracks; from then on every frame
with a single stone was AMBIGUOUS, SEARCH never accepted it, and the episode
ended SEARCH_EXHAUSTED after 2x13 views.
"""
import numpy as np
import pytest

from core.types import GroundStatus
from perception.run import offline_observation, offline_wire
from perception.student_a import StudentAPerception


class Tracker:
    """Only the tracking state of StudentAPerception, without the API client."""

    def __init__(self):
        self._tracks, self._next_id = {}, 0

    _new_id = StudentAPerception._new_id
    track = StudentAPerception._track


@pytest.fixture(scope='module')
def obs():
    return offline_observation()


def stone(bbox):
    return {'name': 'stone', 'color': 'gray', 'bbox': bbox, 'confidence': 0.9}


def test_single_detection_after_a_phantom_duplicate_is_not_ambiguous(obs):
    t = Tracker()
    # Frame 1: hallucinated duplicate, both boxes clipped by the image edge.
    first = t.track([stone([0, 400, 120, 520]), stone([880, 400, 1000, 520])], obs, 'test')
    assert all(g.status is GroundStatus.UNLOCALIZED for g in first)
    # Frame 2: the real single stone, fully inside the frame.
    real = next(d for d in offline_wire()['detections'] if d['name'] == 'stone')
    second = t.track([dict(real)], obs, 'test')
    assert len(second) == 1
    assert second[0].status is GroundStatus.LOCALIZED
    assert second[0].pos_world is not None


def test_two_indistinguishable_detections_are_still_ambiguous(obs):
    t = Tracker()
    real = next(d for d in offline_wire()['detections'] if d['name'] == 'stone')
    t.track([dict(real)], obs, 'test')
    x1, y1, x2, y2 = real['bbox']
    shifted = {**real, 'bbox': [x1 - 60, y1, x2 - 60, y2]}
    out = t.track([dict(real), shifted], obs, 'test')
    assert any(g.status is GroundStatus.AMBIGUOUS for g in out)
