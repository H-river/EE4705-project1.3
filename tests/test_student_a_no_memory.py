"""Stage M2: memory belongs to B (planner/memory.py). A reports only what the
current frame shows: no memory_* attributes, even for an instance that was
LOCALIZED earlier and is UNLOCALIZED now."""
from dataclasses import replace

from core.types import GroundStatus
from perception.student_a import StudentAPerception
from tests.test_student_a_partial_object import frame, make_a  # noqa: F401  (fixture)


def test_no_memory_attributes_after_losing_localization(frame, tmp_path):
    obs, wire = frame
    a = make_a(tmp_path, [wire, wire])
    first = a.describe(obs)
    bottle = next(g for g in first.objects if g.name == 'bottle')
    assert bottle.status is GroundStatus.LOCALIZED
    a._held_id = bottle.instance_id  # held: this frame gives it no position
    rgb = obs.rgb.copy()
    rgb[0, 0, 0] ^= 1
    second = a.describe(replace(obs, rgb=rgb, frame_id=obs.frame_id + 1))
    again = next(g for g in second.objects if g.name == 'bottle')
    assert again.status is GroundStatus.UNLOCALIZED
    for scene in (first, second):
        for g in scene.objects + scene.regions:
            assert not any(k.startswith('memory') for k in g.attributes), g.attributes
            assert 'color' in g.attributes


def test_a_has_no_recall():
    assert not hasattr(StudentAPerception, 'recall')
