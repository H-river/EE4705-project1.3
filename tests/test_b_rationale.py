"""E3: optional one-sentence rationale in B's output, kept in the audit and
in planner.rationales (the runner copies it to TrialRecord.extra)."""
from core.llm_client import validate_schema
from planner.contract import WIRE_SCHEMA
from planner.fixtures import INSTRUCTION, example_response, example_scene, fixture_planner


def test_rationale_is_optional_and_bounded():
    reply = example_response()
    validate_schema(reply, WIRE_SCHEMA)
    validate_schema({**reply, "rationale": "Both goal members are localized."}, WIRE_SCHEMA)
    assert "rationale" not in WIRE_SCHEMA["required"]
    assert WIRE_SCHEMA["properties"]["rationale"]["maxLength"] == 1000


def test_rationale_reaches_audit_and_planner_log(tmp_path):
    reply = {**example_response(), "rationale": "Stone and region are unique and localized, so READY."}
    planner = fixture_planner([reply], tmp_path)
    plan = planner.plan(INSTRUCTION, example_scene())
    assert plan.status.value == "READY"
    assert planner.last_diagnostics["rationale"].startswith("Stone and region")
    assert planner.rationales == [{"call": 1, "status": "READY", "rationale": reply["rationale"],
                                   "used_memory_for": None}]


def test_missing_rationale_is_recorded_as_none(tmp_path):
    planner = fixture_planner([example_response()], tmp_path)
    planner.plan(INSTRUCTION, example_scene())
    assert planner.last_diagnostics["rationale"] is None and planner.rationales[0]["rationale"] is None
