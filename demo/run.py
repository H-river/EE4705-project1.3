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


def run_episode(out, scenario="success", instruction="Move the stone to the red area.", video=True, fps=10, students=()):
    components, labels = load_components(students, retry=scenario == "retry")
    out = pathlib.Path(out).resolve()
    # An episode is an evidence bundle: never mix a new run with old artifacts.
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}. Choose a new --out directory.")
    world = RecordedWorld()
    recorder = None
    try:
        config = demo_scene()
        world.reset(config)
        world.step(500)  # settle for 1 s before the recorded episode starts
        store = ObservationStore(persist_dir=out / "observations")
        env = RobotEnv(world, store=store)
        recorder = Recorder(world, out, instruction, scenario, fps=fps, video=video)
        recorder.module_modes = " | ".join(f"{r}: {'student' if r in students else 'demo'}" for r in "ABC")
        world.recorder = recorder
        perception = ObservedPerception(components["A"], recorder, store)
        planner = ObservedPlanner(components["B"], recorder)
        executor = ObservedExecutor(components["C"], recorder)
        oracle = EvalOracle(world)  # evaluator only; never given to A/B/C
        spy = GraspSpyExecutor(executor, oracle)
        recorder.capture(hold=.5, snapshot="start.png")
        result = DemoOrchestrator(perception, planner, spy, env, NoClarification(),
                                  config=OrchestratorConfig(max_replans=1, max_search_attempts=1),
                                  store=store, recorder=recorder).run(instruction)
        recorder.stage = "EVALUATOR / CHECK"
        recorder.message = "Independent truth check: object identity, destination, release, and 2 s stability"
        recorder.event("eval.start", {"target": "stone", "region": "red_region"}, hold=.6)
        # This intentionally remains a STONE demo. A changed instruction cannot
        # silently redefine the expected object to agree with a wrong planner.
        actual = evaluate_actual(oracle, {"target": "stone", "region": "red_region"}, result.outcome,
                                 spy.grasp_records, result.clarifications)
        recorder.stage = "DONE / " + ("PASS" if result.claimed_success and actual.actual_success else "FAIL")
        recorder.message = f"Visual claim: {result.claimed_success} | Independent actual success: {actual.actual_success} | {actual.checks.get('stability_detail', '')}"
        recorder.event("eval.result", actual, hold=1.5)
        recorder.capture(snapshot="final.png")
        observations = store.flush()
        recorder.close()
        summary = recorder.save({
            "schema": "abc-demo-v1", "scenario": scenario, "instruction": instruction,
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
    parser.add_argument("--instruction", default="Move the stone to the red area.", help="Stone-to-red-area wording; evaluator target remains stone")
    parser.add_argument("--no-video", action="store_true", help="Write JSON, observations and snapshots without ffmpeg")
    parser.add_argument("--fps", type=int, default=10, choices=[5, 10, 20, 25])
    parser.add_argument("--student", action="append", choices=["A", "B", "C"], default=[],
                        help="Replace a demo with its implemented student module; repeat for multiple roles")
    args = parser.parse_args(argv)
    try:
        summary = run_episode(args.out, args.scenario, args.instruction, not args.no_video, args.fps, args.student)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Demo setup/recording error: {exc}\n")
    print(f"{summary['outcome']}: claimed={summary['claimed_success']}, actual={summary['actual_success']}")
    print(f"Replay: {args.out.resolve() / 'index.html'}")
    return 0 if summary["claimed_success"] and summary["actual_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
