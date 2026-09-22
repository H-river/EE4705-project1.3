"""validate_wire drops near-full-frame boxes instead of rejecting the frame."""
import pytest

from perception.vision_contract import validate_wire


def det(name, bbox, color=''):
    return {'name': name, 'color': color, 'bbox': bbox, 'confidence': 0.9}


def wire(detections, selected):
    return {'detections': detections, 'selected': selected, 'answer': ''}


def test_oversized_detection_dropped_and_selected_remapped():
    w = wire([det('stone', [100, 100, 200, 200], 'gray'),
              det('red_region', [0, 0, 900, 900], 'red'),   # exceeds 600x600
              det('cube', [300, 300, 400, 400], 'blue')],
             [0, 2])
    out = validate_wire(w)
    assert out is w
    assert [d['name'] for d in w['detections']] == ['stone', 'cube']
    assert w['selected'] == [0, 1]


def test_selected_oversized_detection_is_removed_from_selected():
    w = wire([det('red_region', [0, 0, 999, 999], 'red'),
              det('stone', [100, 100, 200, 200], 'gray')],
             [0, 1])
    validate_wire(w)
    assert [d['name'] for d in w['detections']] == ['stone']
    assert w['selected'] == [0]


def test_box_wide_but_not_tall_is_kept():
    w = wire([det('red_region', [0, 400, 900, 700], 'red')], [0])
    validate_wire(w)
    assert len(w['detections']) == 1 and w['selected'] == [0]


def test_other_violations_still_raise():
    w = wire([det('red_region', [0, 0, 900, 900], 'red')], [5])
    with pytest.raises(ValueError):
        validate_wire(w)


# Fix 2.8: degenerate boxes. Fixture = live qwen3-vl reply for smoke_1 frame 18
# (runs/night/audit_a/994d822994df/c8b04d10d03c/audit.json), which used
# [0,0,0,0] with confidence 0 as a "not visible" placeholder.
def test_zero_placeholder_box_dropped_live_fixture():
    w = wire([det('stone', [125, 300, 245, 450], 'gray'),
              {'name': 'red_region', 'color': 'red', 'bbox': [0, 0, 0, 0], 'confidence': 0.0},
              det('bottle', [360, 0, 455, 95], 'green')],
             [])
    validate_wire(w)
    assert [d['name'] for d in w['detections']] == ['stone', 'bottle']
    assert w['selected'] == []


@pytest.mark.parametrize('bbox', [[300, 300, 200, 400],   # x1 > x2
                                  [300, 300, 400, 300],   # y1 == y2
                                  [300, 300, 301, 400]])  # width < 2
def test_unselected_degenerate_box_dropped_and_selected_remapped(bbox):
    w = wire([det('cube', bbox, 'blue'), det('stone', [100, 100, 200, 200], 'gray')], [1])
    validate_wire(w)
    assert [d['name'] for d in w['detections']] == ['stone']
    assert w['selected'] == [0]


def test_selected_degenerate_box_still_raises_for_repair():
    w = wire([det('stone', [300, 300, 200, 400], 'gray')], [0])
    with pytest.raises(ValueError):
        validate_wire(w)
