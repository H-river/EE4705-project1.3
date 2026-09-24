"""E1: B metrics v2 (conversions and memory usage) on synthetic audits."""
import json

from eval.b_metrics import conversions, memory_section, memory_usage


def audit(status, *, repair=0, accepted=True, memory=None, age=None, reason="r", normal=None):
    return {"compiled_plan": {"status": status, "reason": reason}, "repair_count": repair, "accepted": accepted,
            "used_memory_for": memory, "memory_age_frames": age, "input": {"instruction": "move"},
            "responses": [{"normalizations": normal or []}]}


def test_conversions_count_statuses_repairs_and_normalisations():
    c = conversions([audit("READY"), audit("REJECTED", repair=1, accepted=False),
                     audit("NEEDS_SEARCH", normal=[{"change": "READY with SEARCH -> NEEDS_SEARCH"}])])
    assert c["status:READY"] == 1 and c["status:REJECTED"] == 1
    assert c["repaired -> not accepted"] == 1 and c["contract failure -> REJECTED (replan)"] == 1
    assert c["READY with SEARCH -> NEEDS_SEARCH"] == 1


def test_memory_usage_splits_trial_outcomes(tmp_path):
    for tid, reason, ok in (("t1", "mem plan", True), ("t2", "fresh plan", False)):
        d = tmp_path / tid
        d.mkdir()
        (d / "trial_record.json").write_text(json.dumps({
            "trial_id": tid, "instruction": "move", "claimed_success": ok, "actual_success": ok,
            "events": [{"type": "plan", "reason": reason}]}))
    m = memory_usage([audit("READY", memory="a1", age=6, reason="mem plan"), audit("READY", reason="fresh plan")],
                     tmp_path)
    assert m["memory_calls"] == 1 and m["ages"] == [6]
    assert (m["memory_trials"], m["memory_trials_ok"], m["fresh_trials"], m["fresh_trials_ok"]) == (1, 1, 1, 0)
    assert "1/2 = 50.0%" in memory_section(m)
