# Owner: backbone (ALL)
"""Aggregate metrics over a run directory of TrialRecord JSON files.

Denominator policy (explicit):
* claimed/actual success rates: denominator = trials whose ``expected``
  marks them feasible manipulation tasks;
* refusal correctness: denominator = infeasible trials;
* clarification correctness: denominator = trials with
  clarification_required;
* any zero denominator yields the JSON value null (never 0.0 and never a
  division error), with the denominator reported alongside.

Claimed and actual success are always reported separately.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from typing import Any, Optional


def _rate(numer: int, denom: int) -> Optional[float]:
    return None if denom == 0 else numer / denom


def compute_metrics(records: list[dict]) -> dict[str, Any]:
    feasible = [r for r in records if r.get("extra", {}).get("expected_feasible", True)]
    infeasible = [r for r in records if not r.get("extra", {}).get("expected_feasible", True)]
    clarif = [r for r in records if r.get("extra", {}).get("clarification_required")]

    claimed = sum(1 for r in feasible if r.get("claimed_success"))
    actual = sum(1 for r in feasible if r.get("actual_success") is True)
    both = sum(1 for r in feasible if r.get("claimed_success") and r.get("actual_success") is True)
    false_claims = sum(1 for r in feasible if r.get("claimed_success") and r.get("actual_success") is not True)
    refused_ok = sum(1 for r in infeasible if r.get("extra", {}).get("refusal_correct") is True)
    clarif_ok = sum(1 for r in clarif if r.get("extra", {}).get("clarification_correct") is True)
    wrong_object = sum(1 for r in feasible if r.get("extra", {}).get("wrong_object") is True)
    outcomes = Counter(r.get("outcome", "?") for r in records)

    return {
        "n_trials": len(records),
        "n_feasible": len(feasible),
        "n_infeasible": len(infeasible),
        "claimed_success_rate": _rate(claimed, len(feasible)),
        "actual_success_rate": _rate(actual, len(feasible)),
        "claimed_and_actual_rate": _rate(both, len(feasible)),
        "false_claim_count": false_claims,
        "wrong_object_count": wrong_object,
        "refusal_correct_rate": _rate(refused_ok, len(infeasible)),
        "clarification_correct_rate": _rate(clarif_ok, len(clarif)),
        "outcome_histogram": dict(outcomes),
        "infrastructure_check_only": all(r.get("infrastructure_check", False) for r in records) if records else None,
    }


def metrics_for_run_dir(run_dir: pathlib.Path) -> dict[str, Any]:
    records = []
    for path in sorted(run_dir.glob("*/trial_record.json")):
        records.append(json.loads(path.read_text()))
    return compute_metrics(records)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate metrics for a run directory")
    parser.add_argument("run_dir", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(metrics_for_run_dir(args.run_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
