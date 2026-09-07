"""Run and record the smallest offline A -> B -> C workflow."""
from __future__ import annotations

import argparse
import pathlib

from core.env import RobotEnv
from core.interfaces import ClarificationProvider
from core.obs_store import ObservationStore
from core.oracle import EvalOracle
from core.orchestrator import Orchestrator, OrchestratorConfig
from core.types import SceneConfig, SceneObjectSpec
from demo.components import ObservedExecutor, ObservedPerception, ObservedPlanner, load_components
from demo.recording import RecordedWorld, Recorder
from eval.criteria import evaluate_actual
from eval.runner import GraspSpyExecutor


def demo_scene():
    return SceneConfig(seed=0, robot_init={"x": -.12, "y": 0., "yaw": .15}, objects=[
        SceneObjectSpec("stone", (.4, -.15, .88)),
        SceneObjectSpec("cube", (.4, .15, .88)),
        SceneObjectSpec("bottle", (.55, -.35, .915)),
    ])


class NoClarification(ClarificationProvider):
    def ask(self, question):
        return None


class DemoOrchestrator(Orchestrator):
    def __init__(self, *args, recorder, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder

    def _event(self, result, kind, **data):
        super()._event(result, kind, **data)
        self.recorder.event("backbone." + kind, data)


def run_episode(out, scenario="success", instruction="Move the stone to the red area.", video=True, fps=10, students=(),
                expected_target="stone", max_action_attempts=2, scene_config=None):
    components, labels = load_components(students, retry=scenario == "retry")
    out = pathlib.Path(out).resolve()
    # An episode is an evidence bundle: never mix a new run with old artifacts.
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}. Choose a new --out directory.")
    world = RecordedWorld()
    recorder = None
    try:
        config = scene_config or demo_scene()
        world.reset(config)
        world.step(500)  # settle for 1 s before the recorded episode starts
        store = ObservationStore(persist_dir=out / "observations")
        env = RobotEnv(world, store=store)
        recorder = Recorder(world, out, instruction, scenario, fps=fps, video=video)
        recorder.module_modes = " | ".join(f"{r}: {'student' if r in students else 'demo'}" for r in "ABC")
        if "B" in students and hasattr(components["B"], "client"):
            recorder.module_modes = recorder.module_modes.replace("B: student", "B: Qwen/" + components["B"].client.response_source)
        world.recorder = recorder
        perception = ObservedPerception(components["A"], recorder, store)
        planner = ObservedPlanner(components["B"], recorder)
        executor = ObservedExecutor(components["C"], recorder)
        oracle = EvalOracle(world)  # evaluator only; never given to A/B/C
        spy = GraspSpyExecutor(executor, oracle)
        recorder.capture(hold=.5, snapshot="start.png")
        result = DemoOrchestrator(perception, planner, spy, env, NoClarification(),
                                  config=OrchestratorConfig(max_replans=1, max_search_attempts=1,
                                                            max_action_attempts=max_action_attempts),
                                  store=store, recorder=recorder).run(instruction)
        recorder.stage = "EVALUATOR / CHECK"
        recorder.message = "Independent truth check: object identity, destination, release, and 2 s stability"
        expected = {"target": expected_target, "region": "red_region"}
        recorder.event("eval.start", expected, hold=.6)
        # Expected identity is an explicit evaluator label, never inferred from
        # the model's plan and never passed to the student modules.
        actual = evaluate_actual(oracle, expected, result.outcome,
                                 spy.grasp_records, result.clarifications)
        recorder.stage = "DONE / " + ("PASS" if result.claimed_success and actual.actual_success else "FAIL")
        recorder.message = f"Visual claim: {result.claimed_success} | Independent actual success: {actual.actual_success} | {actual.checks.get('stability_detail', '')}"
        recorder.event("eval.result", actual, hold=1.5)
        recorder.capture(snapshot="final.png")
        observations = store.flush()
        recorder.close()
        summary = recorder.save({
            "schema": "abc-demo-v1", "scenario": scenario, "instruction": instruction,
            "expected": expected, "max_action_attempts": max_action_attempts,
            "scene_config": config, "modules": labels, "student_modules": list(students),
            "fault_injection": "First GRASP returns GRASP_MISSED before motion" if scenario == "retry" else None,
            "claimed_success": result.claimed_success, "actual_success": actual.actual_success,
            "outcome": result.outcome, "episode": result, "actual": actual,
            "grasp_records": spy.grasp_records,
            "observations": {key: {k: str(pathlib.Path(p).relative_to(out)) for k, p in files.items()}
                             for key, files in observations.items()},
        })
        return summary
    finally:
        world.recorder = None
        if recorder:
            recorder.close()
        world.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True, help="New/empty episode directory")
    parser.add_argument("--scenario", choices=["success", "retry"], default="success")
    parser.add_argument("--instruction", default="Move the stone to the red area.", help="Task instruction")
    parser.add_argument("--expected-target", choices=["stone", "stone2", "cube", "bottle"], default="stone",
                        help="Independent evaluator label; never sent to A/B/C")
    parser.add_argument("--attempts-per-action", type=int, choices=[1, 2], default=2,
                        help="Use 1 with --scenario retry to force a B replan after the injected grasp failure")
    parser.add_argument("--no-video", action="store_true", help="Write JSON, observations and snapshots without ffmpeg")
    parser.add_argument("--fps", type=int, default=10, choices=[5, 10, 20, 25])
    parser.add_argument("--student", action="append", choices=["A", "B", "C"], default=[],
                        help="Replace a demo with its implemented student module; repeat for multiple roles")
    parser.add_argument("--scene-trial", type=pathlib.Path,
                        help="Use a trial YAML's scene, instruction and target label; A/B/C choices stay explicit")
    args = parser.parse_args(argv)
    try:
        scene_config = None
        if args.scene_trial:
            import yaml
            from eval.runner import scene_config_from_spec
            trial = yaml.safe_load(args.scene_trial.read_text())
            if trial.get("clarification_responses") or trial.get("fault_injection"):
                raise ValueError("Recorded scene trials currently support direct instructions without scripted faults/clarifications")
            scene_config = scene_config_from_spec(trial["scene"])
            args.instruction = trial["instruction"]
            args.expected_target = trial["expected"]["target"]
        summary = run_episode(args.out, args.scenario, args.instruction, not args.no_video, args.fps, args.student,
                              args.expected_target, args.attempts_per_action, scene_config)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Demo setup/recording error: {exc}\n")
    print(f"{summary['outcome']}: claimed={summary['claimed_success']}, actual={summary['actual_success']}")
    print(f"Replay: {args.out.resolve() / 'index.html'}")
    return 0 if summary["claimed_success"] and summary["actual_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
