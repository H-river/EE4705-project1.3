"""Inspect B offline, check Qwen configuration, or plan from a saved A scene."""
import argparse
import json
from pathlib import Path

from core.types import GroundedObject, GroundStatus, SceneDescription
from planner.config import QwenPlannerConfig
from planner.contract import WIRE_SCHEMA
from planner.fixtures import INSTRUCTION, example_response, example_scene, fixture_planner
from planner.prompts import json_value
from planner.student_b import StudentBPlanner


def load_scene(path):
    raw = json.loads(Path(path).read_text())

    def instances(items):
        return [GroundedObject(**{**g, "status": GroundStatus(g["status"])}) for g in items]

    return SceneDescription(objects=instances(raw.get("objects", [])), regions=instances(raw.get("regions", [])),
                            caption=raw.get("caption", ""), ambiguities=raw.get("ambiguities", []),
                            frame_id=raw.get("frame_id", -1), sim_time=raw.get("sim_time", 0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline-demo", action="store_true", help="Canned output; no network or Qwen accuracy test")
    mode.add_argument("--check-config", action="store_true", help="Validate environment variables; no request")
    mode.add_argument("--scene", type=Path, help="A SceneDescription JSON; sends a configured Qwen request")
    parser.add_argument("--instruction", help="Required with --scene")
    parser.add_argument("--out", type=Path, default=Path("runs/qwen_b_offline"), help="New/empty output folder")
    args = parser.parse_args(argv)
    try:
        if args.check_config:
            print(json.dumps(QwenPlannerConfig.from_env().public_settings(), indent=2))
            print("Configuration parsed only. Endpoint access and model support are NOT tested.")
            return 0
        if args.scene and not args.instruction:
            parser.error("--scene requires --instruction")
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("--out must be new or empty; use a different folder")
        scene = example_scene() if args.offline_demo else load_scene(args.scene)
        instruction = INSTRUCTION if args.offline_demo else args.instruction
        planner = (fixture_planner([example_response()], args.out / "audit") if args.offline_demo
                   else StudentBPlanner())
        args.out.mkdir(parents=True, exist_ok=True)
        for name, value in (("scene.json", scene), ("input.json", {"instruction": instruction, "scene": scene})):
            (args.out / name).write_text(json.dumps(json_value(value), indent=2) + "\n")
        plan = planner.plan(instruction, scene)
        (args.out / "plan.json").write_text(json.dumps(json_value(plan), indent=2) + "\n")
        parsed = planner.last_diagnostics["responses"][-1]["response"]["parsed"]
        (args.out / "response.json").write_text(json.dumps(parsed, indent=2) + "\n")
        (args.out / "schema.json").write_text(json.dumps(WIRE_SCHEMA, indent=2) + "\n")
        (args.out / "diagnostics.json").write_text(json.dumps(planner.last_diagnostics, indent=2) + "\n")
        print(f"{plan.status.value}: " + " -> ".join(a.skill.value for a in plan.actions))
        print(f"Source: {planner.client.response_source}; outputs: {args.out.resolve()}")
        if args.offline_demo:
            print("OFFLINE FIXTURE ONLY: no Qwen request; language accuracy and physical success are not tested here.")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Planner error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
