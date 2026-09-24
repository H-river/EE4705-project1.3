"""Round 7 made SEARCH carry the goal colour ('blue cube'); the planning
suite labels list the class ('cube'). A contradicting colour or class still fails."""
from eval.b_benchmark import _search_target_ok


def test_coloured_search_target_matches_its_class_label():
    goal = {"object_name": "cube", "object_color": "blue"}
    assert _search_target_ok("cube", ["cube"], goal)
    assert _search_target_ok("blue cube", ["cube"], goal)
    assert not _search_target_ok("red cube", ["cube"], goal)
    assert not _search_target_ok("blue cube", ["bottle"], goal)
    assert not _search_target_ok("blue cube", ["cube"], {"object_name": "cube", "object_color": ""})
