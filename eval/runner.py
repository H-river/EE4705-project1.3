# Owner: backbone (ALL)
"""Trial runner.

Usage:
    python -m eval.runner --mode {grounding,planning,manipulation,e2e} \
        --trials eval/trials/smoke [--mock-all] [--out runs]

Modes select which Student modules are REAL (the others are backbone
mocks):
    grounding     real A (perception) + mock B/C
    planning      mock A + real B (planner) + mock C
    manipulation  mock A/B + real C (executor)
    e2e           real A + real B + real C

``--mock-all`` explicitly replaces all three real modules with mocks; the
results are then labeled infrastructure checks.  WITHOUT this flag, a mode
that needs an unimplemented Student module fails clearly (exit code 2,
naming the missing module) — mocks are never substituted silently.

Exit codes: 0 = all trial expectations met, 1 = one or more expectations
failed (or a trial crashed), 2 = configuration error.

Oracle-only ``expected`` labels never reach perception/planner/executor/
orchestrator; scripted clarifications flow only through the
ClarificationProvider interface.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
import time
from typing import Any, Optional

import yaml

from core.env import RobotEnv
from core.interfaces import Executor, Perception, Planner, RobotEnvProtocol
from core.mocks import GTPerception, RulePlanner, ScriptedClarifier, TeleportExecutor
from core.obs_store import ObservationStore
from core.oracle import EvalOracle
from core.orchestrator import Orchestrator, OrchestratorConfig
from core.types import (
    CONTRACT_VERSION,
    Action,
    ExecutionResult,
    SceneConfig,
    SceneObjectSpec,
    Skill,
    TrialOutcome,
    TrialRecord,
)
from core.world import SimWorld
from eval.criteria import evaluate_actual
from eval.logger import to_json_safe, write_trial_record

MODES = ("grounding", "planning", "manipulation", "e2e")
_REAL = {"grounding": ("A",), "planning": ("B",), "manipulation": ("C",), "e2e": ("A", "B", "C")}


class GraspSpyExecutor(Executor):
    """Eval-side wrapper: observes oracle state at each successful GRASP so
    the evaluator can detect wrong-object manipulation from the simulation
    itself (never from injected flags or config).  Transparent otherwise."""

    def __init__(self, inner: Executor, oracle: EvalOracle) -> None:
        self._inner = inner
        self._oracle = oracle
        self.grasp_records: list[dict] = []

    def reset(self) -> None:
        self._inner.reset()
        self.grasp_records = []

    def execute(self, action: Action, env: RobotEnvProtocol, perception: Perception) -> ExecutionResult:
        result = self._inner.execute(action, env, perception)
        if action.skill is Skill.GRASP and result.success:
            self.grasp_records.append({
                "held_gt_id": self._oracle.held_gt_id(),
                "sim_time": env.sim_time(),
                "action_target": action.target,
            })
        return result


def load_trials(trials_dir: pathlib.Path) -> list[dict]:
    paths = sorted(trials_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"no trial YAML files in {trials_dir}")
    trials = []
    for p in paths:
        spec = yaml.safe_load(p.read_text())
        spec["_path"] = str(p)
        trials.append(spec)
    return trials


def scene_config_from_spec(scene: dict) -> SceneConfig:
    objects = [
        SceneObjectSpec(name=o["name"], pos=tuple(float(v) for v in o["pos"]),
                        yaw=float(o.get("yaw", 0.0)))
        for o in scene.get("objects", [])
    ]
    return SceneConfig(seed=int(scene.get("seed", 0)), objects=objects,
                       robot_init={k: float(v) for k, v in (scene.get("robot_init") or {}).items()})


def _stub_check(name: str, cls: type) -> Optional[str]:
    if not getattr(cls, "IMPLEMENTED", True):
        return name
    return None


def build_modules(mode: str, mock_all: bool, world: SimWorld, oracle: EvalOracle,
                  fault: dict) -> tuple[Perception, Planner, Executor, dict[str, str]]:
    """Returns (perception, planner, executor, module_config).  Raises
    SystemExit(2) with a clear message when a required real module is an
    unimplemented stub."""
    real = () if mock_all else _REAL[mode]
    missing: list[str] = []
    module_config: dict[str, str] = {}

    if "A" in real:
        from perception.student_a import StudentAPerception

        if _stub_check("Student A perception (perception/student_a.py)", StudentAPerception):
            missing.append("Student A perception (perception/student_a.py)")
        perception: Perception = StudentAPerception()
        module_config["perception"] = "perception.student_a.StudentAPerception"
    else:
        perception = GTPerception(oracle)
        module_config["perception"] = "core.mocks.GTPerception (MOCK)"

    if "B" in real:
        from planner.student_b import StudentBPlanner

        if _stub_check("Student B planner (planner/student_b.py)", StudentBPlanner):
            missing.append("Student B planner (planner/student_b.py)")
        planner: Planner = StudentBPlanner()
        module_config["planner"] = "planner.student_b.StudentBPlanner"
    else:
        planner = RulePlanner()
        module_config["planner"] = "core.mocks.RulePlanner (MOCK)"

    if "C" in real:
        from executor.student_c import StudentCExecutor

        if _stub_check("Student C executor (executor/student_c.py)", StudentCExecutor):
            missing.append("Student C executor (executor/student_c.py)")
        executor: Executor = StudentCExecutor()
        module_config["executor"] = "executor.student_c.StudentCExecutor"
    else:
        executor = TeleportExecutor(
            world,
            seed=int(fault.get("seed", 0)),
            fail_prob=float(fault.get("fail_prob", 0.0)),
            wrong_object_prob=float(fault.get("wrong_object_prob", 0.0)),
        )
        module_config["executor"] = "core.mocks.TeleportExecutor (MOCK)"

    if missing:
        print(
            f"ERROR: mode '{mode}' requires unimplemented module(s):\n  - "
            + "\n  - ".join(missing)
            + "\nUse --mock-all to run an infrastructure check with mocks instead.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return perception, planner, executor, module_config


def check_expectations(trial: dict, record: TrialRecord) -> list[str]:
    """Smoke expectations: intermediate events, not only final outcomes.
    Returns a list of human-readable failures (empty = pass)."""
    expected = trial.get("expected", {}) or {}
    failures: list[str] = []
    events = record.events
    action_events = [e for e in events if e.get("type") == "action" and e.get("success")]
    skills_ok = {e.get("skill") for e in action_events}

    if not expected.get("feasible", True):
        if record.outcome != TrialOutcome.REFUSED.value:
            failures.append(f"expected refusal, got outcome {record.outcome}")
        return failures

    for skill in expected.get("events_required", []):
        if skill not in skills_ok:
            failures.append(f"required intermediate event {skill} did not occur successfully")

    if expected.get("search_required"):
        if "SEARCH" not in skills_ok:
            failures.append("expected a successful SEARCH event")

    if expected.get("clarification_required"):
        if not record.clarifications:
            failures.append("expected a clarification exchange")
        elif all(c.get("response") is None for c in record.clarifications):
            failures.append("clarification was asked but no scripted response was consumed")

    if expected.get("claimed_success", True) and record.outcome != TrialOutcome.CLAIMED_SUCCESS.value:
        failures.append(f"expected CLAIMED_SUCCESS, got {record.outcome}")
    if record.actual_success is not True:
        failures.append(f"actual success (oracle) is {record.actual_success}")
    if record.claimed_success and record.actual_success is not True:
        failures.append("system claimed success but oracle disagrees (false claim)")
    return failures


def run_trial(trial: dict, mode: str, mock_all: bool, run_dir: pathlib.Path,
              world: SimWorld, oracle: EvalOracle) -> tuple[TrialRecord, list[str]]:
    trial_id = str(trial.get("id", pathlib.Path(trial["_path"]).stem))
    trial_dir = run_dir / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    expected = trial.get("expected", {}) or {}

    record = TrialRecord(trial_id=trial_id, instruction=str(trial.get("instruction", "")))
    record.extra["category"] = trial.get("category", "standard")
    record.extra["trial_path"] = trial["_path"]
    record.extra["expected_feasible"] = bool(expected.get("feasible", True))
    record.extra["clarification_required"] = bool(expected.get("clarification_required", False))

    fault = dict(trial.get("fault_injection", {}) or {})
    fault.setdefault("seed", trial.get("scene", {}).get("seed", 0))

    store = ObservationStore(persist_dir=trial_dir / "frames")
    env = RobotEnv(world, store=store)
    perception, planner, executor, module_config = build_modules(mode, mock_all, world, oracle, fault)
    spy = GraspSpyExecutor(executor, oracle)
    clarifier = ScriptedClarifier(list(trial.get("clarification_responses", []) or []))
    record.module_config = module_config
    record.infrastructure_check = any("(MOCK)" in v for v in module_config.values())

    world.reset(scene_config_from_spec(trial.get("scene", {}) or {}))
    world.step(int(round(1.0 / world.timestep)))  # settle objects onto their supports (1 s)

    orch = Orchestrator(perception, planner, spy, env, clarifier,
                        config=OrchestratorConfig(), store=store)
    t0 = time.monotonic()
    episode = orch.run(record.instruction)
    record.timings["wall_s"] = time.monotonic() - t0
    record.timings["sim_end_s"] = world.sim_time

    record.outcome = episode.outcome.value
    record.claimed_success = episode.claimed_success
    record.events = [to_json_safe(e) for e in episode.events]
    record.clarifications = [to_json_safe(c) for c in episode.clarifications]
    record.error = episode.error
    record.extra["instruction_history"] = episode.instruction_history
    record.extra["verification"] = to_json_safe(episode.verification)
    record.extra["unconsumed_clarifications"] = clarifier.remaining()

    actual = evaluate_actual(oracle, expected, episode.outcome, spy.grasp_records, episode.clarifications)
    record.actual_success = actual.actual_success
    record.extra["actual_checks"] = to_json_safe(actual.checks)
    record.extra["grasp_records"] = to_json_safe(spy.grasp_records)
    record.extra["wrong_object"] = actual.wrong_object
    record.extra["refusal_correct"] = actual.refusal_correct
    record.extra["clarification_correct"] = actual.clarification_correct

    store.flush()
    failures = check_expectations(trial, record)
    record.extra["expectation_failures"] = failures
    write_trial_record(record, trial_dir / "trial_record.json")
    return record, failures


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EE4705 backbone trial runner")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--trials", required=True, type=pathlib.Path)
    parser.add_argument("--mock-all", action="store_true",
                        help="replace ALL Student modules with backbone mocks (infrastructure check)")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"))
    args = parser.parse_args(argv)

    trials = load_trials(args.trials)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / f"{stamp}_{args.mode}{'_mockall' if args.mock_all else ''}"
    n = 1
    while run_dir.exists():  # unique run directories
        n += 1
        run_dir = args.out / f"{stamp}_{args.mode}{'_mockall' if args.mock_all else ''}_{n}"
    run_dir.mkdir(parents=True)

    # Probe module availability up front for a clear failure (exit 2).
    world = SimWorld()
    oracle = EvalOracle(world)
    try:
        build_modules(args.mode, args.mock_all, world, oracle, {})
    except SystemExit:
        world.close()
        raise

    import json

    meta = {
        "contract_version": CONTRACT_VERSION,
        "mode": args.mode,
        "mock_all": args.mock_all,
        "n_trials": len(trials),
        "trials_dir": str(args.trials),
        "started": stamp,
        "infrastructure_check": args.mock_all,
        "note": ("MOCK-ONLY RESULTS: infrastructure check, not system performance"
                 if args.mock_all else "results include real Student modules"),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    all_failures: dict[str, list[str]] = {}
    records = []
    try:
        for trial in trials:
            trial_id = str(trial.get("id", pathlib.Path(trial["_path"]).stem))
            try:
                record, failures = run_trial(trial, args.mode, args.mock_all, run_dir, world, oracle)
            except SystemExit:
                raise
            except Exception as exc:  # persist partial failure records
                import traceback

                record = TrialRecord(trial_id=trial_id, outcome=TrialOutcome.ERROR.value,
                                     error=traceback.format_exc())
                write_trial_record(record, run_dir / trial_id / "trial_record.json")
                failures = [f"trial crashed: {type(exc).__name__}: {exc}"]
            records.append(record)
            all_failures[trial_id] = failures
            status = "PASS" if not failures else "FAIL"
            print(f"[{status}] {trial_id}: outcome={record.outcome} "
                  f"claimed={record.claimed_success} actual={record.actual_success}")
            for f in failures:
                print(f"        - {f}")
    finally:
        world.close()

    from eval.metrics import compute_metrics

    metrics = compute_metrics([json.loads((run_dir / r.trial_id / "trial_record.json").read_text())
                               for r in records if (run_dir / r.trial_id / "trial_record.json").exists()])
    (run_dir / "metrics.json").write_text(json.dumps(to_json_safe(metrics), indent=2, allow_nan=False))

    failed = sum(1 for v in all_failures.values() if v)
    print(f"\nRun dir: {run_dir}")
    print(f"Trials: {len(trials)}  passed: {len(trials) - failed}  failed: {failed}")
    if args.mock_all:
        print("NOTE: --mock-all results are infrastructure checks, not system performance.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
