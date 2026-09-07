# Owner: backbone (ALL)
"""Regression tests for the A -> B -> C handoff and terminal safety."""

from types import SimpleNamespace

import numpy as np
import pytest

from core import skills
from core.action_targets import TargetResolutionError, resolve_action_position
from core.mocks import GTPerception, RulePlanner, ScriptedClarifier, TeleportExecutor
from core.orchestrator import Orchestrator, OrchestratorConfig
from core.types import (
    Action, ErrorCode, ExecutionContext, GroundedObject, GroundStatus,
    Plan, SceneDescription, Skill, TrialOutcome,
)
from core.validation import validate_plan
from core.verification import check_placement, verify_placement


def scene_at(pos=(0.4, 0.1, 0.88), *, frame_id=1, sim_time=0.0):
    return SceneDescription(
        objects=[GroundedObject("p0", "stone", GroundStatus.LOCALIZED,
                                pos_world=pos, frame_id=frame_id)],
        regions=[GroundedObject("p1", "red_region", GroundStatus.LOCALIZED,
                                pos_world=(0.4, 0.1, 0.85), kind="region",
                                frame_id=frame_id)],
        frame_id=frame_id, sim_time=sim_time,
    )


class IDOnlyPlanner(RulePlanner):
    def plan(self, instruction, scene):
        plan = super().plan(instruction, scene)
        for action in plan.actions:
            action.params.pop("pos", None)
        return plan


def test_id_only_plan_validates_and_executes(standard_world, env, oracle):
    perception, planner = GTPerception(oracle), IDOnlyPlanner()
    scene = perception.describe(env.get_obs())
    plan = planner.plan("move the stone onto the red region", scene)
    assert validate_plan(plan, ExecutionContext(scene=scene)) == []
    assert all("pos" not in a.params for a in plan.actions)
    episode = Orchestrator(perception, planner, TeleportExecutor(standard_world),
                           env, ScriptedClarifier([])).run("move the stone onto the red region")
    assert episode.outcome is TrialOutcome.CLAIMED_SUCCESS, episode
    assert oracle.object_in_region("stone") and not env.is_attached()


def test_target_resolution_uses_exact_id_and_transport_clearance():
    scene = scene_at()
    assert resolve_action_position(Action(Skill.GRASP, "p0"), scene) == pytest.approx((.4, .1, .88))
    assert resolve_action_position(Action(Skill.MOVE_TO, "p1"), scene) == pytest.approx((.4, .1, 1.03))
    assert resolve_action_position(Action(Skill.PLACE, "p1"), scene) == pytest.approx((.4, .1, .85))
    # Explicit legacy waypoints are still supported (including deliberate bad plans).
    action = Action(Skill.MOVE_TO, "p1", {"pos": [.3, .2, 1.1]})
    assert resolve_action_position(action, scene) == pytest.approx((.3, .2, 1.1))
    with pytest.raises(TargetResolutionError):
        resolve_action_position(Action(Skill.GRASP, "stone"), scene)
    scene.objects[0].status = GroundStatus.AMBIGUOUS
    with pytest.raises(TargetResolutionError):
        resolve_action_position(Action(Skill.GRASP, "p0"), scene)


def test_validator_rejects_unlocalized_place_and_bad_waypoints():
    scene = scene_at()
    scene.regions[0].status = GroundStatus.UNLOCALIZED
    scene.regions[0].pos_world = None
    context = ExecutionContext(scene, held_instance_id="p0")
    errors = validate_plan(Plan(actions=[Action(Skill.PLACE, "p1")]), context)
    assert any(e.code == "UNLOCATED_REFERENCE" for e in errors)
    action = Action(Skill.PLACE, "p1", {"pos": [.4, .1, .85]})
    assert validate_plan(Plan(actions=[action]), context) == []
    for skill in (Skill.REACH, Skill.GRASP, Skill.PLACE):
        action = Action(skill, "p1" if skill is Skill.PLACE else "p0", {"pos": [float("nan"), 0., 0.]})
        assert any(e.code == "BAD_PARAM" for e in validate_plan(Plan(actions=[action]), context))


def test_stop_failure_is_visible(standard_world, env, oracle, monkeypatch):
    def broken_stop():
        raise RuntimeError("simulated controller failure")
    monkeypatch.setattr(env, "stop_motion", broken_stop)
    episode = Orchestrator(GTPerception(oracle), RulePlanner(), TeleportExecutor(standard_world),
                           env, ScriptedClarifier([])).run("please fly to the moon")
    assert episode.outcome is TrialOutcome.ERROR and not episode.claimed_success
    assert "stop_motion failed" in episode.error
    assert episode.events[-1]["type"] == "safe_stop" and not episode.events[-1]["success"]


def test_stop_cancels_motion_without_teleport_and_can_resume(standard_world, env):
    w = standard_world
    target = np.array([.36, -.20, .95])
    env.set_arm_target(target)
    env.set_base_target(.12, -.10, .2)
    w.robot.set_waist_target(.35, 0., 0.)
    env.set_gripper("left", 0.)
    env.step(50)
    before = env.get_robot_state()
    qpos, qvel, time = w.data.qpos.copy(), w.data.qvel.copy(), env.sim_time()
    env.stop_motion()
    np.testing.assert_array_equal(w.data.qpos, qpos)
    np.testing.assert_array_equal(w.data.qvel, qvel)
    assert env.sim_time() == time
    assert np.linalg.norm(np.array(before.ee_pos) - target) > .1
    max_ee_travel = 0.
    for _ in range(20):
        env.step(50)
        max_ee_travel = max(max_ee_travel, np.linalg.norm(env.get_ee_pos() - before.ee_pos))
    # Allow physical braking/compliance, but reject continuation of the old reach.
    assert max_ee_travel < .03, max_ee_travel
    assert np.linalg.norm(env.get_base_pose()[:2] - before.base_pose[:2]) < .01
    assert np.max(np.abs(w.robot.arm_qvel())) < .05
    assert abs(env.get_robot_state().gripper_opening["left"] - before.gripper_opening["left"]) < .10
    stopped = env.get_ee_pos().copy()
    assert skills.reach(env, target).success
    assert np.linalg.norm(env.get_ee_pos() - stopped) > .1


def test_stop_preserves_held_object(standard_world, env, oracle):
    assert skills.approach(env, oracle.object_pos("stone")).success
    assert skills.grasp(env, oracle.object_pos("stone")).success
    env.stop_motion()
    env.step(250)
    assert env.is_attached() and oracle.held_gt_id() == "stone"


def test_refusal_also_stops_an_active_reach(standard_world, env, oracle):
    env.set_arm_target(np.array([.36, -.20, .95]))
    env.step(50)
    before = env.get_ee_pos().copy()
    result = Orchestrator(GTPerception(oracle), RulePlanner(), TeleportExecutor(standard_world),
                          env, ScriptedClarifier([])).run("please fly to the moon")
    assert result.outcome is TrialOutcome.REFUSED
    assert any(e["type"] == "safe_stop" and e["success"] for e in result.events)
    env.step(1000)
    assert np.linalg.norm(env.get_ee_pos() - before) < .03


@pytest.mark.parametrize("pos, expected", [
    ((.4, .1, .88), True),
    ((.4, .1, .845), True),  # preserve the oracle's inclusive lower bound
    ((.4, .1, .97), True),   # and inclusive upper bound
    ((.4, .1, 1.60), False),  # same xy, floating 75 cm over the support
    ((.505, .1, .88), False),  # formerly accepted by the extra 3 cm xy slack
    ((.4, .1, .80), False),
])
def test_placement_support_geometry(pos, expected):
    assert check_placement(scene_at(pos), "p0", "p1", attached=False).passed is expected


def test_verification_requires_release_3d_and_exact_instance():
    scene = scene_at()
    assert not check_placement(scene, "p0", "p1", attached=True).passed
    scene.objects[0].status = GroundStatus.UNLOCALIZED
    scene.objects[0].pos_world = None
    scene.objects[0].bbox_xyxy = (100, 100, 110, 110)
    assert not check_placement(scene, "p0", "p1", attached=False).passed
    scene = scene_at()
    scene.objects[0].instance_id = "different_stone"
    assert not check_placement(scene, "p0", "p1", attached=False).passed
    scene.objects[0].instance_id = "p0"
    scene.objects[0].frame_id = 0
    assert not check_placement(scene, "p0", "p1", attached=False).passed


def test_perceived_region_extent_overrides_nominal_size():
    scene = scene_at((.43, .1, .88))
    scene.regions[0].region_half_extents_xy = (.02, .02)
    assert not check_placement(scene, "p0", "p1", attached=False).passed
    scene.objects[0].pos_world = (.55, .1, .88)
    scene.regions[0].region_half_extents_xy = (.2, .08)
    assert check_placement(scene, "p0", "p1", attached=False).passed


@pytest.mark.parametrize("half", [(0., .1), (-.1, .1), (.1,), (float("nan"), .1)])
def test_region_extent_rejects_invalid_geometry(half):
    with pytest.raises(ValueError):
        GroundedObject("p1", "region", GroundStatus.LOCALIZED, kind="region",
                       pos_world=(.4, .1, .85), region_half_extents_xy=half)


class VerificationEnv:
    """Minimal deterministic capture clock; no renderer needed for timing cases."""
    def __init__(self):
        self.time, self.frame = 0., 0

    def step(self, n):
        self.time += n * self.timestep()

    def timestep(self):
        return .002

    def get_obs(self):
        self.frame += 1
        return SimpleNamespace(frame_id=self.frame, sim_time=self.time)

    def is_attached(self):
        return False


class TrackingPerception:
    def __init__(self, drift=0., stale=False):
        self.scene, self.drift, self.stale = scene_at(), drift, stale

    def describe(self, obs):
        # Intentionally reuses/mutates objects to expose snapshot aliasing bugs.
        scene = self.scene
        scene.frame_id, scene.sim_time = obs.frame_id, obs.sim_time
        for obj in scene.objects + scene.regions:
            obj.frame_id = obs.frame_id
        if self.stale:
            scene.frame_id -= 1
        scene.objects[0].pos_world = (.4 + (obs.frame_id - 1) * self.drift, .1, .88)
        return scene


@pytest.mark.parametrize("drift,stale,expected", [(0., False, True), (.025, False, False), (0., True, False)])
def test_fresh_two_frame_verification(drift, stale, expected):
    env = VerificationEnv()
    result = verify_placement(env, TrackingPerception(drift, stale), "p0", "p1")
    assert result.passed is expected, result
    if not stale:
        assert env.frame == 2 and env.time == pytest.approx(.2)


class OffTargetWithVerifyPlanner(RulePlanner):
    def plan(self, instruction, scene):
        plan = super().plan(instruction, scene)
        for action in plan.actions:
            if action.skill in (Skill.MOVE_TO, Skill.PLACE):
                action.params["pos"][0] -= .30
                action.params["pos"][1] -= .30
        grasp = next(a for a in plan.actions if a.skill is Skill.GRASP)
        place = next(a for a in plan.actions if a.skill is Skill.PLACE)
        plan.actions.insert(-1, Action(Skill.VERIFY, params={
            "condition": "object_in_region", "object": grasp.target, "region": place.target,
        }))
        return plan


class LyingVerifyExecutor(TeleportExecutor):
    def execute(self, action, env, perception):
        if action.skill is Skill.VERIFY:
            return self._result(action, env, True, ErrorCode.NONE)
        return super().execute(action, env, perception)


def test_in_plan_verify_cannot_bypass_final_check(standard_world, env, oracle):
    episode = Orchestrator(GTPerception(oracle), OffTargetWithVerifyPlanner(),
                           LyingVerifyExecutor(standard_world), env, ScriptedClarifier([]),
                           config=OrchestratorConfig(max_replans=0)).run("move the stone onto the red region")
    assert any(e["type"] == "action" and e["skill"] == "VERIFY" and e["success"]
               for e in episode.events), episode
    assert not episode.claimed_success and episode.outcome is not TrialOutcome.CLAIMED_SUCCESS
    assert episode.verification is not None and not episode.verification.passed
    assert not oracle.object_in_region("stone")


@pytest.mark.parametrize('failed_skill', [Skill.GRASP, Skill.PLACE])
def test_partial_skill_failure_replans_with_actual_attachment_state(standard_world,env,oracle,failed_skill):
    from core.types import PlanStatus
    class RecordingPlanner(RulePlanner):
        def replan(self,instruction,scene,history,context,clarification=None):
            self.seen=(context.held_instance_id,context.last_release_instance_id)
            return Plan(status=PlanStatus.INFEASIBLE,reason='End this focused state handoff test')
    class FailureAfterEffect(TeleportExecutor):
        def execute(self,action,env,perception):
            result=super().execute(action,env,perception)
            if action.skill is failed_skill and result.success:
                self.target=action.target
                result.success=False
                result.error_code=ErrorCode.TIMEOUT
                if action.skill is Skill.GRASP:
                    result.info['held_instance_id']=action.target
            return result
    planner,executor=RecordingPlanner(),FailureAfterEffect(standard_world)
    episode=Orchestrator(GTPerception(oracle),planner,executor,env,ScriptedClarifier([])).run(
        'move the stone onto the red region')
    assert episode.outcome is TrialOutcome.REFUSED,episode
    if failed_skill is Skill.GRASP:
        assert planner.seen==(executor.target,None) and env.is_attached()
    else:
        assert planner.seen[0] is None and planner.seen[1] is not None and not env.is_attached()
    attempts=[e for e in episode.events if e['type']=='action' and e['skill']==failed_skill.value]
    assert len(attempts)==1  # No second GRASP/PLACE using the obsolete state.
    assert any(e['type']=='partial_action_state' for e in episode.events)
