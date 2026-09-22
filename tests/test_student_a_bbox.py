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
