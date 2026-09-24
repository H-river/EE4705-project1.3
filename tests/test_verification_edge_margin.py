"""Final2: the visual placement claim keeps 1 cm from the region edge.

s1_diet f20: final verification measured the dark_red stone at offset
(-0.0777, -0.0766) of a 0.08 m half extent and claimed success; the oracle
found it outside the region (false claim)."""
from core.types import GroundedObject, GroundStatus, SceneDescription
from core.verification import VISION_EDGE_MARGIN_M, check_placement

REGION = (0.40, 0.30, 0.853)


def scene(dx, dy):
    obj = GroundedObject("a3", "stone", GroundStatus.LOCALIZED, pos_world=(0.40 + dx, 0.30 + dy, 0.871), frame_id=5)
    region = GroundedObject("a2", "red_region", GroundStatus.LOCALIZED, kind="region", pos_world=REGION,
                            region_half_extents_xy=(0.08, 0.08), frame_id=5)
    return SceneDescription([obj], [region], frame_id=5)


def test_f20_corner_estimate_is_not_claimed():
    assert not check_placement(scene(-0.0777, -0.0766), "a3", "a2", attached=False).passed


def test_largest_true_positive_on_disk_still_passes():
    assert check_placement(scene(0.0665, 0.01), "a3", "a2", attached=False).passed
    assert check_placement(scene(0.0, 0.0), "a3", "a2", attached=False).passed


def test_margin_is_one_centimetre():
    assert VISION_EDGE_MARGIN_M == 0.01
    assert check_placement(scene(0.069, 0.0), "a3", "a2", attached=False).passed
    assert not check_placement(scene(0.071, 0.0), "a3", "a2", attached=False).passed
