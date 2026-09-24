"""Final run stage T2: eval.runner --jobs N (eval.parallel) and its scoring."""
import json
import pathlib
import shutil
import subprocess
import sys

import yaml

from eval.final_table import expected_outcome, is_correct, is_false_claim
from eval.parallel import is_provider_error, split_round_robin

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_round_robin_split_keeps_every_trial_once():
    ids = [f"t{i}" for i in range(7)]
    lanes = split_round_robin(ids, 3)
    assert lanes == [["t0", "t3", "t6"], ["t1", "t4"], ["t2", "t5"]]
    assert sorted(sum(lanes, [])) == sorted(ids)


def test_scoring_rules():
    ok = {"claimed_success": True, "actual_success": True, "outcome": "CLAIMED_SUCCESS"}
    false = {"claimed_success": True, "actual_success": False, "outcome": "CLAIMED_SUCCESS"}
    assert is_correct(ok, "success") and not is_correct(false, "success")
    assert is_false_claim(false) and not is_false_claim(ok)
    assert is_correct({"outcome": "REFUSED"}, "reject")
    assert not is_correct(ok, "reject")
    assert not is_correct(ok, "clarify")  # no clarification exchange
    assert is_correct({**ok, "clarifications": [{"question": "which?", "response": "gray"}]}, "clarify")
    assert expected_outcome({"expected": {"feasible": False}}) == "reject"
    assert expected_outcome({"expected": {"clarification_required": True}}) == "clarify"
    assert expected_outcome({"expected": {"outcome": "success"}}) == "success"


def test_provider_errors_are_recognised():
    assert is_provider_error({"outcome": "ERROR", "error": "APIError: HTTP 503: busy"})
    assert is_provider_error({"outcome": "ERROR", "error": "HTTP 429: You have exceeded"})
    assert is_provider_error({"outcome": "ERROR", "error": "Temporary failure in name resolution"})
    assert not is_provider_error({"outcome": "ERROR", "error": "KeyError: 'x'"})
    assert not is_provider_error({"outcome": "FAILED", "error": "HTTP 503"})


def test_final50_trials_have_complete_expected_blocks():
    paths = sorted((ROOT / "eval/trials/final50").glob("*.yaml"))
    assert len(paths) == 50
    outcomes = []
    for p in paths:
        t = yaml.safe_load(p.read_text())
        e = t["expected"]
        outcomes.append(e["outcome"])
        if e["outcome"] == "reject":
            assert e["feasible"] is False
        else:
            assert e["target"] and e["region"] == "red_region" and e["feasible"] is True
    assert outcomes.count("reject") == 1 and outcomes.count("clarify") == 2


def test_jobs_two_on_mock_trials_merges_records(tmp_path):
    trials = tmp_path / "trials"
    trials.mkdir()
    for name in ("f01_smoke_1_standard", "f02_smoke_2_scene_variation", "f03_smoke_3_instruction_variation",
                 "f49_reject_stack"):
        shutil.copy(ROOT / "eval/trials/final50" / f"{name}.yaml", trials)
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-m", "eval.runner", "--mode", "e2e", "--mock-all",
                           "--trials", str(trials), "--jobs", "2", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "SCORE 4/4 false_claims=0" in proc.stdout
    assert (out / "job_0/job.log").exists() and (out / "job_1/job.log").exists()
    assert len(list((out / "merged").iterdir())) == 4
    rows = json.loads((out / "rows.json").read_text())
    assert rows["score"]["correct"] == 4 and "E2E table" in (out / "E2E_TABLE.md").read_text()


def test_trial_timeout_is_recorded(tmp_path):
    trials = tmp_path / "trials"
    trials.mkdir()
    shutil.copy(ROOT / "eval/trials/final50/f01_smoke_1_standard.yaml", trials)
    shutil.copy(ROOT / "eval/trials/final50/f02_smoke_2_scene_variation.yaml", trials)
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-m", "eval.runner", "--mode", "e2e", "--mock-all",
                           "--trials", str(trials), "--jobs", "2", "--trial-timeout", "0.2",
                           "--out", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert "SCORE 0/2" in proc.stdout
    rows = json.loads((out / "rows.json").read_text())["rows"]
    assert {r["outcome"] for r in rows} == {"TIMEOUT"}
