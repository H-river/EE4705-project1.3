"""Round 9 step 4: a released object that the VLM mislabels is still found by
depth on the region (RUN 1 f14/f31: the green bottle standing on the red region
came back as 'gray stone' with the region's box, so PLACE's check and the
final verification said 'exact object or region instance is not visible')."""
import numpy as np
import pytest

from core.types import GroundedObject, GroundStatus, Observation, ReleasedHint, TrackingHint
from perception.student_a import StudentAPerception
from tests.test_student_a_partial_object import make_a

REGION = (0.40, 0.30, 0.853)
F = 500.0


def top_down_obs(object_height=None, object_xy=(0.40, 0.30), frame_id=7):
    """Camera 0.65 m above the region looking straight down; the region box is
    pixels (220..420, 140..340) = +-0.26 m around its centre."""
    k = np.array([[F, 0, 320], [0, F, 240], [0, 0, 1.0]])
    t = np.eye(4)
    t[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    t[:3, 3] = [REGION[0], REGION[1], REGION[2] + 0.65]
    depth = np.full((480, 640), 0.65, dtype=np.float32)
    if object_height is not None:
        u = 320 + (object_xy[0] - REGION[0]) * F / 0.65
        v = 240 - (object_xy[1] - REGION[1]) * F / 0.65
        depth[int(v) - 15:int(v) + 15, int(u) - 15:int(u) + 15] = 0.65 - object_height
    return Observation(frame_id, np.zeros((480, 640, 3), np.uint8), depth, k, t, 1.0, "head")


def region_ground(frame_id=7):
    return GroundedObject("a2", "red_region", GroundStatus.LOCALIZED, bbox_xyxy=(220, 140, 420, 340),
                          pos_world=REGION, kind="region", frame_id=frame_id, region_half_extents_xy=(.08, .08),
                          attributes={"color": "red"})


def a_with_hint(tmp_path, hint):
    a = make_a(tmp_path, [])
    a._tracks = {"a0": {"key": ("bottle", "green"), "pos": (0.5, -0.25, 0.91)}}
    a._hint, a._fresh_hint = hint, False
    return a


RELEASED = TrackingHint(released=ReleasedHint("a0", (0.40, 0.30, 1.05)))
HELD = TrackingHint(held_object_id="a0", held_pos_world=(0.41, 0.29, 1.03))


@pytest.mark.parametrize("hint", [RELEASED, HELD])
def test_mislabelled_object_on_the_region_is_found_by_depth(tmp_path, hint):
    a = a_with_hint(tmp_path, hint)
    out = a._depth_in_region(top_down_obs(0.12), [region_ground()])
    bottle = next(g for g in out if g.instance_id == "a0")
    assert bottle.status is GroundStatus.LOCALIZED and bottle.name == "bottle"
    assert bottle.attributes["pos_basis"] == "depth_in_region"
    assert bottle.pos_world == pytest.approx((0.40, 0.30, REGION[2] + 0.12), abs=0.01)


def test_empty_region_gives_nothing(tmp_path):
    a = a_with_hint(tmp_path, RELEASED)
    assert len(a._depth_in_region(top_down_obs(None), [region_ground()])) == 1


def test_not_when_the_instance_is_detected_or_the_hint_is_fresh_or_elsewhere(tmp_path):
    seen = GroundedObject("a0", "bottle", GroundStatus.UNLOCALIZED, frame_id=7, attributes={"color": "green"})
    a = a_with_hint(tmp_path, RELEASED)
    assert len(a._depth_in_region(top_down_obs(0.12), [region_ground(), seen])) == 2
    a._fresh_hint = True  # the orchestrator's own perceive call
    assert len(a._depth_in_region(top_down_obs(0.12), [region_ground()])) == 1
    far = TrackingHint(released=ReleasedHint("a0", (0.60, -0.20, 1.0)))
    a = a_with_hint(tmp_path, far)
    assert len(a._depth_in_region(top_down_obs(0.12), [region_ground()])) == 1


def test_too_tall_cluster_is_not_an_object(tmp_path):
    a = a_with_hint(tmp_path, RELEASED)
    assert len(a._depth_in_region(top_down_obs(0.40), [region_ground()])) == 1  # arm/hand, not an object


def test_describe_marks_hinted_calls_fresh(tmp_path):
    a = make_a(tmp_path, [])
    calls = []
    a._observe = lambda obs, query, grounding=False: (calls.append(a._fresh_hint), None, None)[1:] or (None,)
    a.describe(None, hint=RELEASED)
    a.describe(None)
    assert calls == [True, False] and a._fresh_hint is False
