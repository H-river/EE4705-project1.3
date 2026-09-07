"""Evidence checks for the offline baseline; these are not course AI scores."""
from dataclasses import replace

import numpy as np
import pytest

from core.env import RobotEnv
from core.oracle import EvalOracle
from core.types import ExecutionContext, GroundedObject, GroundStatus, PlanStatus, SceneDescription, SceneObjectSpec, Skill
from core.validation import validate_plan
from demo.components import load_components
from demo.run import demo_scene, run_episode
from perception.demo_rgbd import RGBDPerception
from planner.demo_rules import DemoPlanner


def test_rgbd_localization_tracks_a_changed_image_and_does_not_invent_missing_objects(world):
    config = demo_scene()
    config.objects[0] = SceneObjectSpec("stone", (.43, -.12, .88))
    world.reset(config)
    world.step(500)
    obs = RobotEnv(world).get_obs()
    perception = RGBDPerception()
    scene = perception.describe(obs)
    stone = next(g for g in scene.objects if g.name == "stone")
    assert stone.source == "rgbd_demo" and stone.status is GroundStatus.LOCALIZED
    assert stone.frame_id == obs.frame_id and scene.sim_time == obs.sim_time
    assert np.linalg.norm(np.asarray(stone.pos_world) - EvalOracle(world).object_pos("stone")) < .012
    assert stone.instance_id != "stone"
    blank = replace(obs, rgb=np.zeros_like(obs.rgb))
    assert perception.describe(blank).objects == []
    assert perception.ground(blank, "stone") is None
    # Valid RGB with no metric depth must not produce a fabricated 3D point.
    invalid_depth = replace(obs, depth=np.full_like(obs.depth, np.nan))
    assert perception.describe(invalid_depth).objects == []


def planning_scene():
    return SceneDescription(objects=[GroundedObject("p7", "stone", GroundStatus.LOCALIZED,
                            pos_world=(.4, -.15, .875), attributes={"color": "gray"})],
                            regions=[GroundedObject("p9", "red_region", GroundStatus.LOCALIZED,
                            kind="region", pos_world=(.4, .3, .853), region_half_extents_xy=(.08, .08))])


@pytest.mark.parametrize("instruction", ["Move the stone to the red area.", "Put the rock in the red zone."])
def test_plan_preserves_grounded_ids_and_can_be_validated(instruction):
    scene = planning_scene()
    plan = DemoPlanner().plan(instruction, scene)
    assert plan.status is PlanStatus.READY
    assert validate_plan(plan, ExecutionContext(scene)) == []
    assert next(a for a in plan.actions if a.skill is Skill.GRASP).target == "p7"
    assert next(a for a in plan.actions if a.skill is Skill.PLACE).target == "p9"
    verify = next(a for a in plan.actions if a.skill is Skill.VERIFY)
    assert verify.params == {"condition": "object_in_region", "object": "p7", "region": "p9"}


def test_planner_handles_missing_ambiguous_refused_and_already_held_states():
    planner, scene = DemoPlanner(), planning_scene()
    missing = planner.plan("Move the stone to the red area", SceneDescription())
    assert missing.status is PlanStatus.NEEDS_SEARCH
    scene.objects.append(GroundedObject("p8", "stone", GroundStatus.LOCALIZED,
                                        pos_world=(.5, -.2, .875), attributes={"color": "dark_red"}))
    ambiguous = planner.plan("Move the stone to the red area", scene)
    assert ambiguous.status is PlanStatus.NEEDS_CLARIFICATION and not ambiguous.actions
    assert planner.plan("Pour the bottle into the cup", scene).status is PlanStatus.INFEASIBLE
    assert planner.plan("Do not move the stone to the red area", scene).status is PlanStatus.INFEASIBLE
    context = ExecutionContext(scene, held_instance_id="p7")
    replanned = planner.replan("Move the stone to the red area", scene, [], context)
    assert not any(a.skill is Skill.GRASP for a in replanned.actions)
    assert validate_plan(replanned, context) == []


@pytest.mark.parametrize("scenario", ["success", "retry"])
def test_real_simulated_demo_and_recorded_recovery(tmp_path, scenario):
    result = run_episode(tmp_path / scenario, scenario=scenario, video=False)
    assert result["claimed_success"] is True, result["episode"]
    assert result["actual_success"] is True, result["actual"]
    assert result["actual"]["checks"]["stable"] is True
    grasps = [e["data"] for e in result["events"] if e["type"] == "backbone.action" and e["data"]["skill"] == "GRASP"]
    if scenario == "retry":
        assert [(g["attempt"], g["success"]) for g in grasps] == [(1, False), (2, True)]
        assert grasps[0]["error"] == "GRASP_MISSED"
    else:
        assert len(grasps) == 1 and grasps[0]["success"]
    assert result["grasp_records"][0]["held_gt_id"] == "stone"
    assert result["observations"]
    assert result["frames"][0]["display_hold"]
    assert all(a["sim_time"] <= b["sim_time"] for a, b in zip(result["frames"], result["frames"][1:]))
    assert (tmp_path / scenario / "index.html").exists()
    with pytest.raises(ValueError, match="not empty"):
        run_episode(tmp_path / scenario, video=False)


def test_student_swap_has_no_silent_stub_fallback(monkeypatch):
    from planner.student_b import StudentBPlanner
    monkeypatch.setattr(StudentBPlanner, "IMPLEMENTED", False)
    with pytest.raises(ValueError, match="still a stub"):
        load_components(["B"])
    with pytest.raises(ValueError, match="requires DemoExecutor"):
        load_components(["C"], retry=True)
