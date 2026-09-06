# Owner: backbone (ALL)
"""Contract acceptance tests: the five smoke trials plus the behavioral
guarantees the backbone must enforce (ID separation, independent
wrong-object detection, verification-gated success claims, bounded
clarification/search, clean episode resets)."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Optional

import pytest

import eval.runner as runner_mod
from core.env import RobotEnv
from core.interfaces import Perception, Planner
from core.mocks import GTPerception, RulePlanner, ScriptedClarifier, TeleportExecutor
from core.orchestrator import Orchestrator, OrchestratorConfig
from core.types import (
    Action,
    GroundedObject,
    Observation,
    Plan,
    PlanStatus,
    SceneDescription,
    Skill,
    TrialOutcome,
)
from eval.criteria import evaluate_actual
from eval.runner import GraspSpyExecutor
from tests.conftest import standard_scene

TRIALS_DIR = pathlib.Path(__file__).resolve().parent.parent / "eval" / "trials" / "smoke"
GT_IDS = {"stone", "stone2", "cube", "bottle", "red_region"}


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke_runs")
    code = runner_mod.main(["--mode", "e2e", "--trials", str(TRIALS_DIR), "--mock-all",
                            "--out", str(out)])
    run_dir = next(out.iterdir())
    records = {}
    for p in run_dir.glob("*/trial_record.json"):
        rec = json.loads(p.read_text())
        records[rec["trial_id"]] = rec
    return code, run_dir, records


# ------------------------------------------------------------ five smoke trials

def test_smoke_run_exit_code_zero(smoke_run):
    code, _, records = smoke_run
    assert code == 0, {k: v.get("extra", {}).get("expectation_failures") for k, v in records.items()}
    assert len(records) == 5


@pytest.mark.parametrize("trial_id", [
    "smoke_1_standard", "smoke_2_scene_variation", "smoke_3_instruction_variation",
    "smoke_4_search", "smoke_5_clarification",
])
def test_smoke_trial_passes(smoke_run, trial_id):
    _, _, records = smoke_run
    rec = records[trial_id]
    assert rec["extra"]["expectation_failures"] == []
    assert rec["outcome"] == "CLAIMED_SUCCESS"
    assert rec["claimed_success"] is True
    assert rec["actual_success"] is True
    # intermediate events, not just outcomes:
    ok_skills = [e["skill"] for e in rec["events"] if e.get("type") == "action" and e.get("success")]
    assert "GRASP" in ok_skills and "PLACE" in ok_skills
    assert rec["infrastructure_check"] is True  # mock-only => labeled


def test_smoke_search_trial_required_search(smoke_run):
    _, _, records = smoke_run
    rec = records["smoke_4_search"]
    ok_skills = [e["skill"] for e in rec["events"] if e.get("type") == "action" and e.get("success")]
    assert "SEARCH" in ok_skills
    # SEARCH truly changed the view: the first perceive saw no objects
    first_perceive = next(e for e in rec["events"] if e["type"] == "perceive")
    assert first_perceive["objects"] == []


def test_smoke_clarification_trial_consumed_script(smoke_run):
    _, _, records = smoke_run
    rec = records["smoke_5_clarification"]
    assert len(rec["clarifications"]) == 1
    assert rec["clarifications"][0]["response"] == "the gray one"
    assert rec["extra"]["unconsumed_clarifications"] == 0
    # the grasped object is the gray stone, not the distractor
    assert rec["extra"]["grasp_records"][0]["held_gt_id"] == "stone"


def test_run_metadata_records_module_config(smoke_run):
    _, run_dir, records = smoke_run
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["infrastructure_check"] is True
    for rec in records.values():
        assert "(MOCK)" in rec["module_config"]["perception"]
        assert "(MOCK)" in rec["module_config"]["planner"]
        assert "(MOCK)" in rec["module_config"]["executor"]


# ------------------------------------------------------------ id separation

def test_perceived_ids_differ_from_ground_truth_ids(smoke_run):
    _, _, records = smoke_run
    for rec in records.values():
        for e in rec["events"]:
            if e.get("type") == "perceive":
                for pid in e["objects"]:
                    assert re.fullmatch(r"p\d+", pid), pid
                    assert pid not in GT_IDS


# ------------------------------------------------------------ wrong object

def test_wrong_object_detected_by_evaluator(world, oracle):
    """Inject wrong-object grasping physically; the evaluator must detect it
    purely from observed oracle state (grasp-time held identity), not from
    the injection configuration."""
    world.reset(standard_scene(seed=7))
    world.step(200)
    env = RobotEnv(world)
    executor = TeleportExecutor(world, seed=7, wrong_object_prob=1.0)
    spy = GraspSpyExecutor(executor, oracle)
    orch = Orchestrator(GTPerception(oracle), RulePlanner(), spy, env, ScriptedClarifier([]),
                        config=OrchestratorConfig())
    episode = orch.run("move the stone onto the red region")
    expected = {"target": "stone", "region": "red_region", "feasible": True}
    actual = evaluate_actual(oracle, expected, episode.outcome, spy.grasp_records,
                             episode.clarifications)
    assert spy.grasp_records, "no grasp happened"
    assert spy.grasp_records[0]["held_gt_id"] != "stone"
    assert actual.wrong_object is True
    assert actual.actual_success is False


# ------------------------------------------------------------ success gating

class _OffTargetPlanner(Planner):
    """Emits a plan whose PLACE position is OUTSIDE the region and contains
    no VERIFY: every action succeeds, but final verification must fail and
    the orchestrator must refuse to claim success."""

    def plan(self, instruction: str, scene: SceneDescription) -> Plan:
        target = next((g for g in scene.objects if g.name == "stone"), None)
        region = next((r for r in scene.regions if r.name == "red_region"), None)
        assert target is not None and region is not None
        off = [region.pos_world[0] - 0.30, region.pos_world[1] - 0.30, region.pos_world[2]]
        return Plan(status=PlanStatus.READY, actions=[
            Action(Skill.APPROACH, target=target.instance_id, params={"pos": list(target.pos_world)}),
            Action(Skill.GRASP, target=target.instance_id, params={"pos": list(target.pos_world)}),
            Action(Skill.MOVE_TO, target=region.instance_id,
                   params={"pos": [off[0], off[1], off[2] + 0.18]}),
            Action(Skill.PLACE, target=region.instance_id,
                   params={"object": target.instance_id, "pos": off}),
            Action(Skill.STOP),
        ])

    def replan(self, instruction, scene, history, context, clarification=None) -> Plan:
        return self.plan(instruction, scene)


def test_no_claim_without_valid_final_verification(world, oracle):
    """Completing an action list is not success: with all actions succeeding
    but the object placed off-region, CLAIMED_SUCCESS must not be returned."""
    world.reset(standard_scene(seed=8))
    world.step(200)
    env = RobotEnv(world)
    orch = Orchestrator(GTPerception(oracle), _OffTargetPlanner(),
                        TeleportExecutor(world, seed=8), env, ScriptedClarifier([]),
                        config=OrchestratorConfig(max_replans=1, max_total_plans=3))
    episode = orch.run("move the stone onto the red region")
    action_events = [e for e in episode.events if e["type"] == "action"]
    assert any(e["skill"] == "PLACE" and e["success"] for e in action_events)
    assert episode.claimed_success is False
    assert episode.outcome is not TrialOutcome.CLAIMED_SUCCESS
    assert not oracle.object_in_region("stone")


# ------------------------------------------------------------ clarification bounds

def test_exhausted_clarification_terminates(world, oracle):
    """Two stones, NO scripted responses: the system must terminate with
    CLARIFICATION_EXHAUSTED instead of looping."""
    cfg = standard_scene(seed=9)
    from core.types import SceneObjectSpec

    cfg.objects.append(SceneObjectSpec("stone2", (0.80, -0.02, 0.43)))
    world.reset(cfg)
    world.step(200)
    env = RobotEnv(world)
    orch = Orchestrator(GTPerception(oracle), RulePlanner(), TeleportExecutor(world, seed=9),
                        env, ScriptedClarifier([]), config=OrchestratorConfig())
    episode = orch.run("move the stone onto the red region")
    assert episode.outcome is TrialOutcome.CLARIFICATION_EXHAUSTED
    assert episode.claimed_success is False


def test_repeated_unhelpful_clarification_terminates(world, oracle):
    """Scripted responses that never disambiguate must still terminate
    within the clarification budget."""
    cfg = standard_scene(seed=10)
    from core.types import SceneObjectSpec

    cfg.objects.append(SceneObjectSpec("stone2", (0.80, -0.02, 0.43)))
    world.reset(cfg)
    world.step(200)
    env = RobotEnv(world)
    clarifier = ScriptedClarifier(["the stone", "the stone", "the stone", "the stone"])
    orch = Orchestrator(GTPerception(oracle), RulePlanner(), TeleportExecutor(world, seed=10),
                        env, clarifier, config=OrchestratorConfig(max_clarifications=2))
    episode = orch.run("move the stone onto the red region")
    assert episode.outcome is TrialOutcome.CLARIFICATION_EXHAUSTED
    assert len(episode.clarifications) <= 2  # bounded, each consumed once
    assert clarifier.remaining() >= 2  # unused scripted responses NOT consumed


# ------------------------------------------------------------ fatal search

class _FatalSearchPerception(Perception):
    """describe() sees nothing; ground() blows up (camera failure model)."""

    def describe(self, obs: Observation, query: Optional[str] = None) -> SceneDescription:
        return SceneDescription(frame_id=obs.frame_id, sim_time=obs.sim_time)

    def ground(self, obs: Observation, target: str) -> Optional[GroundedObject]:
        raise RuntimeError("simulated camera failure")


def test_fatal_search_error_terminates_safely(world, oracle):
    world.reset(standard_scene(seed=11))
    world.step(200)
    env = RobotEnv(world)
    orch = Orchestrator(_FatalSearchPerception(), RulePlanner(), TeleportExecutor(world, seed=11),
                        env, ScriptedClarifier([]), config=OrchestratorConfig())
    episode = orch.run("move the stone onto the red region")
    assert episode.outcome is TrialOutcome.ERROR
    assert episode.claimed_success is False
    assert any(e["type"] == "safe_stop" for e in episode.events)
    # distinguished from the recoverable outcome:
    assert episode.outcome is not TrialOutcome.SEARCH_EXHAUSTED


def test_search_not_found_is_recoverable_not_fatal(world, oracle):
    """A clean not-found (bottle genuinely absent) ends in SEARCH_EXHAUSTED,
    not ERROR."""
    from core.types import SceneConfig, SceneObjectSpec

    world.reset(SceneConfig(seed=12, objects=[SceneObjectSpec("cube", (0.60, 0.15, 0.43))]))
    world.step(200)
    env = RobotEnv(world)
    orch = Orchestrator(GTPerception(oracle), RulePlanner(), TeleportExecutor(world, seed=12),
                        env, ScriptedClarifier([]), config=OrchestratorConfig())
    episode = orch.run("move the stone onto the red region")
    assert episode.outcome is TrialOutcome.SEARCH_EXHAUSTED
    assert episode.claimed_success is False


# ------------------------------------------------------------ reset hygiene

def test_episode_reset_clears_tracking_attachment_history(world, oracle):
    world.reset(standard_scene(seed=13))
    world.step(200)
    env = RobotEnv(world)
    perception = GTPerception(oracle)
    # First episode: perceive and attach something.
    obs = env.get_obs()
    scene1 = perception.describe(obs)
    assert scene1.objects, "expected visible objects"
    from core import skills

    assert skills.reach(env, oracle.object_pos("stone")).success
    assert env.try_attach_near_ee() is not None
    assert oracle.held_gt_id() == "stone"

    # Reset everything (as the trial runner does between trials).
    world.reset(standard_scene(seed=13))
    perception.reset()
    world.step(200)

    assert oracle.held_gt_id() is None
    assert not env.is_attached()
    assert world.sim_time > 0  # settled fresh episode, own clock
    # Perception ID lifecycle restarted: first-seen object is p0 again.
    scene2 = perception.describe(env.get_obs())
    ids = sorted(g.instance_id for g in scene2.objects)
    assert ids == sorted(g.instance_id for g in scene1.objects)
    assert ids[0] == "p0"
