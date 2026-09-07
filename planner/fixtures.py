"""Hand-written examples for offline interface tests, NEVER Qwen predictions."""
from core.llm_client import FakeTransport, LLMClient, LLMConfig
from core.types import GroundedObject, GroundStatus, SceneDescription
from planner.config import QwenPlannerConfig
from planner.contract import WIRE_VERSION
from planner.student_b import StudentBPlanner

INSTRUCTION = "Move the stone beside the blue cube to the red area."


def example_scene():
    return SceneDescription(objects=[
        GroundedObject("p0", "stone", GroundStatus.LOCALIZED, pos_world=(.4, -.15, .875),
                       attributes={"color": "gray"}, confidence=.95, source="offline_fixture", frame_id=7),
        GroundedObject("p1", "cube", GroundStatus.LOCALIZED, pos_world=(.4, .15, .875),
                       attributes={"color": "blue"}, confidence=.95, source="offline_fixture", frame_id=7),
    ], regions=[GroundedObject("p2", "red_region", GroundStatus.LOCALIZED, kind="region",
                               pos_world=(.4, .3, .853), region_half_extents_xy=(.08, .08),
                               confidence=.95, source="offline_fixture", frame_id=7)], frame_id=7)


def wire_action(skill, target="", *, object="", region="", condition=""):
    return dict(skill=skill, target=target, object=object, region=region, condition=condition)


def example_response(object_id="p0", region_id="p2"):
    return {"schema_version": WIRE_VERSION, "status": "READY",
            "goal": {"object_id": object_id, "object_name": "stone", "object_color": "",
                     "region_id": region_id, "region_name": "red_region"},
            "actions": [wire_action("APPROACH", object_id), wire_action("GRASP", object_id),
                        wire_action("MOVE_TO", region_id), wire_action("PLACE", region_id, object=object_id),
                        wire_action("VERIFY", object=object_id, region=region_id, condition="object_in_region"),
                        wire_action("STOP")],
            "reason": "Move the stone; the cube is only a reference object.", "clarification_question": ""}


def fixture_planner(responses, audit_dir):
    import json

    llm = LLMConfig(model="offline-fixture-NOT-Qwen", base_url="https://fixture.invalid/v1",
                    api_key="fixture-only-placeholder", max_retries=0, retry_backoff_s=0,
                    structured_output_mode="json_schema")
    transport = FakeTransport([FakeTransport.completion(json.dumps(r) if isinstance(r, dict) else r)
                               for r in responses])
    return StudentBPlanner(QwenPlannerConfig(llm, str(audit_dir)),
                           client=LLMClient(llm, transport=transport))
