# Owner: backbone (ALL)
"""Ground-truth success criteria (ACTUAL success), judged by the evaluator
through the oracle — completely independent of the system's own claim.

Actual success for a feasible manipulation trial requires ALL of:
1. correct object: the ground-truth identity attached at the (first
   successful) grasp event equals ``expected.target`` — recorded by the
   runner's oracle spy AT the grasp event, so a wrong-object grasp is
   detected even if the wrong object is later placed correctly;
2. correct destination: the expected object's center lies inside the
   expected region's finite bounds (x/y within center±half_extents, z in
   [support_z - 0.005, support_z + 0.12]);
3. released: no attachment remains at the end of the trial;
4. stability: condition 2 is maintained for 2.0 s of simulation time,
   sampled every 0.1 s, with positional drift <= 0.02 m from the first
   sample and linear speed <= 0.05 m/s at EVERY sample (see
   core.oracle.check_stability).

``expected`` labels are oracle-side only: they must never reach the
planner, executor, or orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.oracle import EvalOracle
from core.types import TrialOutcome


@dataclass
class ActualResult:
    actual_success: Optional[bool]  # None when not applicable (e.g. refusal trials)
    checks: dict[str, Any] = field(default_factory=dict)
    grasped_gt_id: Optional[str] = None
    wrong_object: Optional[bool] = None
    refusal_correct: Optional[bool] = None
    clarification_correct: Optional[bool] = None


def evaluate_actual(
    oracle: EvalOracle,
    expected: dict,
    outcome: TrialOutcome,
    grasp_records: list[dict],
    clarifications: list,
) -> ActualResult:
    """Judge one finished trial.  ``grasp_records`` are the runner spy's
    observations of oracle.held_gt_id() at each successful grasp event."""
    feasible = bool(expected.get("feasible", True))
    res = ActualResult(actual_success=None)

    # Refusal correctness is tracked separately from manipulation.
    if not feasible:
        res.refusal_correct = outcome is TrialOutcome.REFUSED
        res.checks["refused"] = res.refusal_correct
        return res

    if expected.get("clarification_required"):
        res.clarification_correct = len(clarifications) > 0 and any(
            c.response is not None for c in clarifications
        )

    target = expected.get("target")
    region = expected.get("region", "red_region")
    if target is None:
        res.checks["error"] = "expected.target missing for feasible trial"
        res.actual_success = False
        return res

    # 1. correct object at the grasp event
    grasped = grasp_records[0]["held_gt_id"] if grasp_records else None
    res.grasped_gt_id = grasped
    correct_object = grasped == target
    res.wrong_object = grasped is not None and grasped != target
    res.checks["correct_object"] = correct_object
    res.checks["grasped_gt_id"] = grasped

    # 2. destination
    in_region = oracle.object_in_region(target, region)
    res.checks["in_region"] = in_region

    # 3. released
    released = oracle.held_gt_id() is None
    res.checks["released"] = released

    # 4. stability, maintained over 2 s (only worth running if placed)
    if in_region and released:
        stable, detail = oracle.check_stability(target)
    else:
        stable, detail = False, "skipped (not placed/released)"
    res.checks["stable"] = stable
    res.checks["stability_detail"] = detail

    res.actual_success = bool(correct_object and in_region and released and stable)
    return res
