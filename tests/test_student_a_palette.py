"""Round fix: impossible class/colour detections are dropped or normalised.

Fixtures are live qwen3-vl replies from runs/night/audit_a (STAGE 7):
- 861f75731660 frame 6, ground("cube"): the red square was returned as a
  "red cube" (only a blue cube exists), SEARCH accepted it, and describe() on
  the same image found no cube -> SEARCH loop until max_total_plans (c_2_06/07).
- bc8ef8a6db24 frame 17, ground("stone"): the dark red stone was labelled
  "red", so its tracker key differed from describe()'s "dark_red".
"""
from perception.vision_contract import CLASS_COLORS, validate_wire


def test_palette_comes_from_public_vocabulary():
    assert CLASS_COLORS['stone'] == {'gray', 'dark_red'}
    assert CLASS_COLORS['cube'] == {'blue'}
    assert CLASS_COLORS['red_region'] == {'red'}


def test_selected_red_cube_is_dropped_live_fixture():
    w = {'detections': [{'name': 'cube', 'color': 'red', 'bbox': [520, 443, 916, 897], 'confidence': 0.98}],
         'selected': [0], 'answer': ''}
    validate_wire(w)
    assert w['detections'] == [] and w['selected'] == []


def test_red_stone_normalised_to_dark_red_live_fixture():
    w = {'detections': [{'name': 'stone', 'color': 'red', 'bbox': [400, 500, 480, 580], 'confidence': 0.9},
                        {'name': 'red_region', 'color': 'red', 'bbox': [100, 400, 380, 700], 'confidence': 0.9}],
         'selected': [0], 'answer': ''}
    validate_wire(w)
    assert [d['color'] for d in w['detections']] == ['dark_red', 'red'] and w['selected'] == [0]


def test_off_palette_item_dropped_and_selected_remapped():
    w = {'detections': [{'name': 'bottle', 'color': 'blue', 'bbox': [10, 10, 60, 90], 'confidence': 0.9},
                        {'name': 'cube', 'color': 'blue', 'bbox': [300, 300, 380, 380], 'confidence': 0.9},
                        {'name': 'stone', 'color': '', 'bbox': [500, 500, 560, 560], 'confidence': 0.9}],
         'selected': [1, 2], 'answer': ''}
    validate_wire(w)
    assert [d['name'] for d in w['detections']] == ['cube', 'stone'] and w['selected'] == [0, 1]


# Fix K7: the provider's schema check used to raise on these, which killed the
# whole frame and ended smoke_1/smoke_3/smoke_4 in ERROR. Both fixtures are
# live qwen3-vl replies from the night run.
def test_color_empty_is_normalised_to_no_colour():
    w = {'detections': [{'name': 'bottle', 'color': 'empty', 'bbox': [360, 0, 455, 95], 'confidence': 0.9}],
         'selected': [], 'answer': ''}
    validate_wire(w)
    assert w['detections'][0]['color'] == ''


def test_short_bbox_is_dropped_when_not_selected():
    w = {'detections': [{'name': 'stone', 'color': 'gray', 'bbox': [100, 100, 200, 200], 'confidence': 0.9},
                        {'name': 'cube', 'color': 'blue', 'bbox': [300, 300, 380], 'confidence': 0.9}],
         'selected': [0], 'answer': ''}
    validate_wire(w)
    assert [d['name'] for d in w['detections']] == ['stone'] and w['selected'] == [0]


def test_short_bbox_is_dropped_when_it_is_the_answer():
    w = {'detections': [{'name': 'cube', 'color': 'blue', 'bbox': [300, 300, 380], 'confidence': 0.9}],
         'selected': [0], 'answer': ''}
    validate_wire(w)
    assert w['detections'] == [] and w['selected'] == []
