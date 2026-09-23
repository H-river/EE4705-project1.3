# Student Guide: Build the Robot Task Together

**Watch an example first:** the [offline demo guide](DEMO_README.md) has
commands for a recorded A → B → C workflow and a browser replay. Student B
can also run their own planner with demo A and C before the other modules
are ready. The demo's A and B use rules; they do not complete the AI tasks.

Our goal is to make a simulated robot follow an instruction such as:

> "Find the stone, pick it up, and put it in the red area."

We split this into three jobs:

| Student | Your job in simple words | Example output | Main file to edit |
| --- | --- | --- | --- |
| **A — Task 2: vision** | Look at a camera image. Find the objects and the red area. Answer questions about what is visible. | "This is stone `p0`, and it is here." | `perception/student_a.py` |
| **B — Task 3: planning (your role)** | Read the instruction and A's object list. Decide which actions the robot should take. | "Approach `p0`, grasp it, move to `p1`, place it, check, stop." | `planner/student_b.py` |
| **C — Task 4: execution** | Make the robot carry out each action. Check whether it worked and report failures. | "GRASP succeeded" or "GRASP failed: nothing was attached." | `executor/student_c.py` |

All three students also share the environment/report work in Task 1 and final integration in Task 5. See the [assignment PDF](../EE4705_Project1.3%20S1%20AY2627.pdf), pages 3–5.

**You can start separately.** A does not need to write robot motion code. B can develop with a small object list before A is ready. C can reuse the existing motion functions before B is ready.

The project already provides the simulator, cameras, robot controller, shared Python data types, and a program that connects A, B, and C. We call this shared code the **backbone**. All three student interfaces now have starting implementations. C has a tested simulation executor; A has a Qwen-VL adapter with offline tests but no live visual accuracy result yet. B has a live Qwen interface with a 32/32 result on a predefined planning set. Use the detailed [A guide](STUDENT_A_README.md), [B guide](STUDENT_B_README.md), and [C guide](STUDENT_C_README.md). The [C/A report](validation/AC_REFINEMENT.md) separates tested behavior from remaining work.

Start with [setup](#1-start-the-project), then read your section: [Student A](#2-student-a-help-the-robot-see), [Student B](#3-student-b-turn-instructions-into-actions), or [Student C](#4-student-c-make-the-robot-act). Use the [technical reference](#7-technical-reference-use-when-needed) when you need exact fields or units.

## 1. Start the project

### 1.1 Open a terminal in the project folder

On the current machine:

```bash
cd /home/jiamo/EE4705/project1.3
pwd
ls pyproject.toml core perception planner executor
```

On another computer, open the folder containing `pyproject.toml` and run the remaining commands there. The commands use a Linux terminal. All command paths below are relative to this project folder.

### 1.2 Install once, if needed

Skip this step if the project already has a working `.venv` and the robot assets have been downloaded.

```bash
# Python 3.10 or newer is required.
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
bash scripts/fetch_menagerie.sh
```

`.venv` is this project's Python environment. Using `.venv/bin/python` selects the correct packages without needing to activate it. The fetch script downloads robot assets and needs internet access. The starter exercises below do not need a model API key.

### 1.3 Check that the shared project works

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m eval.runner --mode e2e --trials eval/trials/smoke --mock-all
```

`pytest` runs automatic checks. The saved baseline is **121 passed, 1 skipped**; the skip is an experimental physical-grasp test. See the [saved validation record](validation/student_handoff/REPORT.md).

The second command runs five small example tasks. Look for:

```text
Trials: 5  passed: 5  failed: 0
```

A **mock** is a prepared replacement for a student module. `--mock-all` uses replacements for all three students, so this checks the shared setup. It does not test your real model or robot strategy.

These commands normally run without opening a simulator window. Images and results are saved to files. No window appearing is normal.

### 1.4 Understand the basic workflow

```text
User instruction + camera image
          |
          v
A: object list and positions
          |
          v
B: ordered list of actions
          |
          v
C: execute one action and report its result
          |
          v
Take another image, check, and update the plan if needed
```

An **episode** is one complete attempt at a user task. A **trial** is an episode or component example used in an experiment. **Grounding** means connecting a phrase such as "the stone" to a particular object in the image.

## 2. Student A: help the robot see

### What you need to build

Your module receives an image and a question. It should identify objects, describe the scene, and locate the requested object. A vision-language model (VLM) is an AI model that accepts images and text. You may use a VLM or another suitable visual AI model allowed by the assignment.

For "Where is the stone?", return the stone's ID, its box in the image, and its estimated position. If the stone is missing, report that. If two stones look equally likely, report the ambiguity instead of guessing.

**First milestone:** save one camera image, run your chosen visual model on it, and return a useful object list. You do not need to move the robot to reach this milestone.

### First exercise: save a camera image

Copy the whole block into a terminal, including the final `PY` line:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core.env import RobotEnv
from core.types import SceneConfig, SceneObjectSpec
from core.world import SimWorld

out = Path("runs/student_a_intro")
out.mkdir(parents=True, exist_ok=True)
world = SimWorld()
env = RobotEnv(world)
try:
    world.reset(SceneConfig(seed=11, objects=[
        SceneObjectSpec("stone", (0.40, -0.15, 0.88)),
        SceneObjectSpec("cube", (0.40, 0.15, 0.88)),
        SceneObjectSpec("bottle", (0.55, -0.35, 0.915)),
    ]))
    env.step(500)  # Let the objects settle on the table.
    obs = env.get_obs("head")
    Image.fromarray(obs.rgb).save(out / "head_rgb.png")
    np.save(out / "head_depth.npy", obs.depth)
    metadata = {
        "frame_id": obs.frame_id,
        "sim_time": obs.sim_time,
        "camera_name": obs.camera_name,
        "intrinsics": obs.intrinsics.tolist(),
        "t_world_camera": obs.t_world_camera.tolist(),
    }
    (out / "head_meta.json").write_text(json.dumps(metadata, indent=2))
    print("RGB:", obs.rgb.shape, obs.rgb.dtype)
    print("Depth:", obs.depth.shape, obs.depth.dtype)
    print("Saved files to", out.resolve())
finally:
    world.close()
PY
```

Open `runs/student_a_intro/head_rgb.png` in your editor or image viewer. You also have a depth array and camera information from the same capture. Depth tells you how far a visible surface is from the camera.
Expected shapes are RGB `(480, 640, 3)` and depth `(480, 640)`. Rerunning this exercise replaces these three example files.

### Your Python interface: input and output

Start from `perception/student_a.py`, which now implements the Qwen-VL adapter. Keep the class name `StudentAPerception`. The [A guide](STUDENT_A_README.md) gives the offline command and API setup.

| Method | Input | Output |
| --- | --- | --- |
| `describe(obs, query=None)` | Camera data, plus an optional question | `SceneDescription`: object list, region list, and a text answer/description |
| `ground(obs, target)` | Camera data and a phrase such as "the gray stone" | One `GroundedObject`, or `None` when no target is visible |
| `reset()` | Called at the start of a new episode | Clear object tracking from the previous episode |

These Python data types are in `core/types.py`. You do not need to invent a new format between students.
An example output object has these fields. The numbers are illustrative, not fixed answers:

```text
instance_id: p0
name: stone
kind: object
bbox_xyxy: [280, 220, 315, 255]
pos_world: [0.40, -0.15, 0.88]
status: LOCALIZED
confidence: 0.91
frame_id: the input image's frame_id
```

The box gives left, top, right, and bottom pixel coordinates. `pos_world` gives the estimated position in meters. Use `caption` to answer scene questions, and fill `regions` as well as `objects`: B and C need the destination too.

### Build it in this order

1. Send an image and question to your model. Save the original response so you can inspect mistakes.
2. Convert the response into objects and image boxes. If you resized the image, convert boxes back to the original 640 × 480 image.
3. Use depth and camera information to estimate positions. See the reference for the formula. For unreliable positions, use `UNLOCALIZED` with `pos_world=None`.
4. Give each object a stable ID. Two stones can both have `name="stone"`, but need different IDs. Keep the same ID when the same object moves or the camera view changes.
5. Handle a missing target and at least one confusing case, such as two similar objects. Clear tracking in `reset()`.

### Test A

Run the camera and model-client checks now:

```bash
.venv/bin/python -m pytest -q tests/test_env.py tests/test_cam_sync.py tests/test_llm_client.py
```

These test shared tools. The existing `tests/test_student_a.py` covers the parser, bad depth, missing objects, ambiguous references, stable IDs and reset. Run:

```bash
.venv/bin/python -m pytest -q tests/test_student_a.py
```

The adapter already has `IMPLEMENTED = True`. After configuring and checking the real API, you can exercise it through the workflow:

```bash
.venv/bin/python -m eval.runner --mode grounding --trials eval/trials/smoke --out runs/student_a
```

This uses your A with prepared B/C replacements. Do not add `--mock-all`, because that would replace your A too.

For the report, evaluate **at least 20 trials** with different objects, positions, and viewing directions. Include missing or confusing targets, and demonstrate scene description and visual question answering.
Save images, queries, predictions, correct answers, and response times. Report target-selection accuracy and discuss missed, falsely detected, or invented objects. State which cases count toward accuracy; report missing-target and ambiguity handling separately if needed.

### What to give B and C

Give them a sample `SceneDescription` and explain your ID rules. Confirm that it contains the stone **and** the region, positions in meters, and the input image's `frame_id` and `sim_time`.
Tell B how ambiguity appears. Tell C what happens when a previously tracked object is no longer visible.

## 3. Student B: turn instructions into actions

### What you need to build

Your module reads the user's request and A's scene description. It returns a short action list. You choose **what should happen next**; C handles motor control.
If A says `p0` is a stone and `p1` is the red area, your plan can say:

```text
APPROACH p0 → GRASP p0 → MOVE_TO p1 → PLACE p0 in p1 → VERIFY → STOP
```

You can start before A or C is ready. Use a scene written by hand to test the planner. You do not need camera rendering or robot joint calculations for this first step.

### First exercise: create and check a plan without a model

This teaches the required Python output format. It does **not** implement your language model.

```bash
.venv/bin/python - <<'PY'
import json
from dataclasses import asdict
from pathlib import Path
from core.types import (
    Action, ExecutionContext, GroundedObject, GroundStatus,
    Plan, PlanStatus, SceneDescription, Skill,
)
from core.validation import validate_plan

scene = SceneDescription(
    objects=[GroundedObject(
        "p0", "stone", GroundStatus.LOCALIZED,
        pos_world=(0.40, -0.15, 0.88), frame_id=0,
    )],
    regions=[GroundedObject(
        "p1", "red_region", GroundStatus.LOCALIZED,
        pos_world=(0.40, 0.30, 0.85), kind="region", frame_id=0,
        region_half_extents_xy=(0.08, 0.08),
    )],
    frame_id=0, sim_time=0.0,
)
plan = Plan(status=PlanStatus.READY, actions=[
    Action(Skill.APPROACH, "p0"),
    Action(Skill.GRASP, "p0"),
    Action(Skill.MOVE_TO, "p1"),
    Action(Skill.PLACE, "p1", {"object": "p0"}),
    Action(Skill.VERIFY, params={
        "condition": "object_in_region", "object": "p0", "region": "p1",
    }),
    Action(Skill.STOP),
])
errors = validate_plan(plan, ExecutionContext(scene=scene))
print("Validation errors:", errors)  # Expected: []
assert not errors
out = Path("runs/student_b_intro")
out.mkdir(parents=True, exist_ok=True)
for name, value in (("scene.json", scene), ("plan.json", plan)):
    (out / name).write_text(json.dumps(asdict(value), default=lambda x: x.value, indent=2))
print("Saved examples to", out.resolve())
PY
```

Open `runs/student_b_intro/scene.json` and `plan.json`. These show the data you receive and return. Rerunning the exercise replaces these files.
Later, your model should generate the plan from the instruction and scene rather than use this fixed action list.

### Your Python interface: input and output

The Qwen adapter is implemented in `planner/student_b.py`. Keep the class name `StudentBPlanner`. Follow the [B guide](STUDENT_B_README.md) for its offline run, exact JSON format and API setup.

| Method | Input | Output |
| --- | --- | --- |
| `plan(instruction, scene)` | User text and A's scene description | A `Plan` |
| `replan(instruction, scene, history, context, clarification=None)` | Updated scene, previous action results, current holding state, and possibly an answer from the user | A new `Plan` replacing the remaining actions |
| `reset()` | Called at the start of an episode | Clear information from the previous task |

`history` tells you what happened. `context.held_instance_id` tells you which object the system believes it holds. If GRASP already succeeded, a revised plan must not blindly grasp again.

### Build it in this order

1. Connect an LLM, VLM, or VLA-based planner. The shared model client may help; see Section 5.
2. Give the model the instruction, visible objects/regions, allowed actions, and output format. Convert its JSON into `Plan` and `Action` objects. Convert strings such as `"GRASP"` into `Skill.GRASP`.
3. Call `validate_plan`. Handle invalid output with a limited retry or clear failure, not an endless loop.
4. Handle different ways of asking for the same task, then add search, clarification, and refusal.
5. Implement `replan()` using the current holding state and latest scene.

| Situation | What B should return |
| --- | --- |
| A unique target is visible and located | `READY` with actions |
| The target is not currently visible | `NEEDS_SEARCH` with SEARCH; plan again after a new view |
| "The stone" could mean either of two stones | `NEEDS_CLARIFICATION`, a question, and no actions |
| The request is unsupported or clearly impossible | `INFEASIBLE`, a reason, and no actions |

For SEARCH, `target` is a word such as `stone`. For actions on a visible object, use A's ID such as `p0`. The Qwen response uses IDs; B's Python compiler fills positions from A. C can refresh them from a newer observation before moving.

**A valid format does not prove a correct plan.** A plan can pass `validate_plan` while choosing the wrong object. Your evaluation must check the instruction's meaning too.

### Test B

Run the existing planning and model-client checks:

```bash
.venv/bin/python -m pytest -q tests/test_validation.py tests/test_llm_client.py
```

`tests/test_student_b.py` now contains offline contract and simulation integration tests. Run them, then add real-model instruction tests with independently labelled answers:

```bash
.venv/bin/python -m pytest -q tests/test_student_b.py
```

B already has `IMPLEMENTED = True`. The live adapter passed 32 predefined B cases, but new instructions and A/C integration still need testing. After configuring Qwen as described in the [B guide](STUDENT_B_README.md), run:

```bash
.venv/bin/python -m eval.runner --mode planning --trials eval/trials/smoke --out runs/student_b
```

This uses prepared A/C replacements, so you can test integration without waiting for teammates.
For B's meaning and action-order accuracy, run the separately labelled benchmark in
[Section 6 of the B guide](STUDENT_B_README.md#6-measure-b-without-waiting-for-a-or-c).

For the report, use **at least 20 natural-language instructions**, including paraphrases and invalid/infeasible requests. Label each instruction's intended object, destination, expected status, and required action order before evaluating the model.
Count a plan as correct only when its format, references, action dependencies, and meaning are correct. Different valid sequences can count as correct; they do not need to match one answer word for word.

Save the instruction, input scene, raw model response, parsed plan, validation errors, correctness label, and response time. Report **Action Planning Accuracy = correct plans / evaluated instructions**, and show representative failures.

### Your coordination with A and C

Ask A for a real example scene early and replace your hand-written fixture with it. Agree on how missing and ambiguous objects are represented.
Give C one simple, validated plan early. Start with one visible stone and one red region. Once that works, add SEARCH, clarification, and recovery.
Keep model prompts and plan parsing in B; let C report movement failures through `ExecutionResult`.

## 4. Student C: make the robot act

### What you need to build

Your module receives **one action at a time**. It moves the robot, checks the result, and returns success or a useful error.

Reuse the motion functions in `core/skills.py` to get started. The project already calculates the arm joints needed to reach a position. You do not need to write that calculation from scratch. The robot's base slides around the table; learning to walk is outside this setup.

Start with default **weld attachment**: a simulator constraint makes an object follow the gripper. The assignment allows grasping or attaching. Contact-based physical grasping is experimental and is not needed for your first working task.

### First exercise: move the arm and save before/after pictures

This uses the real simulated controller, with a known target point. It needs no A, B, or model API.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image
from core import skills
from core.env import RobotEnv
from core.types import SceneConfig, SceneObjectSpec
from core.world import SimWorld

out = Path("runs/student_c_intro")
out.mkdir(parents=True, exist_ok=True)
world = SimWorld()
env = RobotEnv(world)
try:
    world.reset(SceneConfig(seed=11, objects=[
        SceneObjectSpec("stone", (0.40, -0.15, 0.88)),
        SceneObjectSpec("cube", (0.40, 0.15, 0.88)),
        SceneObjectSpec("bottle", (0.55, -0.35, 0.915)),
    ]))
    env.step(500)
    Image.fromarray(env.get_obs("head").rgb).save(out / "before.png")
    target = np.array([0.36, -0.20, 0.95])
    result = skills.reach(env, target)
    print("Reached:", result.success, "Error:", result.error_code.value)
    print("Position error (m):", float(np.linalg.norm(env.get_ee_pos() - target)))
    env.stop_motion()
    env.step(250)  # Give the simulated robot time to settle.
    Image.fromarray(env.get_obs("head").rgb).save(out / "after.png")
    assert result.success, result
    print("Saved pictures to", out.resolve())
finally:
    world.close()
PY
```

Expected: `Reached: True Error: NONE`, a position error below about 0.012 m, and two pictures in `runs/student_c_intro/`. Open them to compare the arm. Rerunning the exercise replaces them.
The **end effector**, or **TCP**, is the point near the gripper fingers that the controller moves to the requested position.

### Your Python interface: input and output

Start from `executor/student_c.py`, which selects the working `executor/closed_loop.py` implementation. Keep the class name `StudentCExecutor`. The [C guide](STUDENT_C_README.md) gives recording commands, recovery details and measured results.

```text
execute(action, env, perception) -> ExecutionResult
```

| Input or output | Meaning |
| --- | --- |
| `action` | One step, for example `Action(Skill.GRASP, target="p0")` |
| `env` | Methods to read cameras, move the robot, and check holding state |
| `perception` | A's interface; use it to locate targets in new images |
| `ExecutionResult.success` | Whether this one action achieved its goal |
| `error_code` | A failure such as `GRASP_MISSED`, `TARGET_LOST`, or `TIMEOUT` |
| `post_frame_id` | ID of the image taken after the action |
| `recovery_attempted` / `info` | Whether you tried recovery, plus useful details |

### Build it in this order

1. Get a new scene with `perception.describe(env.get_obs())`. Use `resolve_action_position(action, scene)` from `core/action_targets.py` to turn the target ID into a position.
2. Start with APPROACH and REACH, using `skills.approach` and `skills.reach`.
3. Add GRASP with `skills.grasp`, then check `env.is_attached()`. Sending a close command alone does not prove success.
4. Add MOVE_TO with `skills.move_to`. Check that the robot arrived and still holds the object.
5. Add PLACE: move to a suitable release height, release, let the object settle, then check its position. `skills.place(env)` only releases and waits; it does not move to the region or prove a correct placement.
6. Add SEARCH, VERIFY, and STOP. Reuse `skills.search` and `core.verification.verify_placement` where useful. STOP calls `env.stop_motion()`; use `env.step()` afterward to observe settling if needed.
7. Add at least one recovery or failure-reporting behavior. For example, observe again and retry a grasp once, then return a clear failure. Limit retries and simulated execution time.

PLACE resolves to the region's surface center. **Do not drive the gripper directly into that surface.** Choose a release height that accounts for the object and its offset from the gripper.

The motion functions return `SkillResult`. Your executor must return `ExecutionResult`, including the action and image after it. Use this pattern **inside your implementation**, after running a primitive:

```python
post = env.get_obs()
return ExecutionResult(
    action=action,
    success=primitive_result.success,
    error_code=primitive_result.error_code,
    post_frame_id=post.frame_id,
    info=primitive_result.info,
)
```

This is only a conversion pattern. Add any extra checks needed for that action before setting success.

### Test C

Run the existing controller checks:

```bash
.venv/bin/python -m pytest -q tests/test_g1_control.py tests/test_gripper.py tests/test_student_handoff.py
```

To explore which positions the arm can reach:

```bash
.venv/bin/python scripts/ik_reach_test.py
```

Read the printed output path and open the plot and CSV under `runs/ik_reach/`. Start near working points before trying difficult positions.

The existing `tests/test_student_c.py` checks individual actions and a real simulated transfer. It includes failed grasps, lost targets, stalled motion, stopping and release. Extend it when you add behavior. Run:

```bash
.venv/bin/python -m pytest -q tests/test_student_c.py
```

C already has `IMPLEMENTED = True`. Test it with prepared A/B inputs:

```bash
.venv/bin/python -m eval.runner --mode manipulation --trials eval/trials/student_c --out runs/student_c
```

This uses prepared A/B replacements with your real executor. `TeleportExecutor` is a replacement that skips physical movement; do not use it as your real implementation.

For the report, run **at least 10 trials** with different starting robot and object positions. Save results for grasp, transport, place, and stop, including retries and failures.
Report correct-object grasp successes / grasp attempts, correct-place successes / place attempts, and complete manipulation successes / trials. Also report final placement error or an equivalent measure, such as horizontal distance from object center to region center in meters.
Explain denominators; if no PLACE was attempted, its rate is N/A, not 100%.

### What to give A and B

Give B examples of success and failure results, including which failures should cause replanning. Tell A when you need a fresh view and what position accuracy works in your grasp tests.

## 5. Connect a model: shared notes for A and B

The project provides `LLMClient` in `core/llm_client.py`:

| Call | Use |
| --- | --- |
| `call_llm(prompt, system=None, json_schema=None)` | Text input, useful for B |
| `call_vlm(image_png_bytes, prompt, system=None, json_schema=None)` | Image and text input, useful for A |

Both return `LLMResponse`, with text, parsed JSON when requested, the original response, time, and token counts. After the camera exercise, A can read the PNG bytes with `Path("runs/student_a_intro/head_rgb.png").read_bytes()`.

This example demonstrates JSON handling **without calling a real service**:

```bash
.venv/bin/python - <<'PY'
from core.llm_client import FakeTransport, LLMClient, LLMConfig

client = LLMClient(
    LLMConfig(model="example", base_url="https://example.invalid/v1", api_key="test-only"),
    transport=FakeTransport([FakeTransport.completion('{"answer": "stone"}')]),
)
response = client.call_llm(
    "Return an object name.",
    json_schema={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
)
print(response.parsed)  # Expected: {'answer': 'stone'}
PY
```

For a real model, provide the actual `model`, `base_url`, and `api_key` in `LLMConfig` and use the real transport. Keep credentials out of source files and Git. The example address above is deliberately not a service address.

The runner creates student classes with **no constructor arguments**. If you add `__init__(client=None)` for testing, make sure the no-argument path loads your documented configuration. The runner has no `--model` or `--base-url` options, and `LLMConfig` does not load environment variables automatically.

Use prepared responses for parser/error tests and your actual chosen model for reported AI experiments. Record the model/version, prompt, output format, image processing, time, call count, and cache use. A cached reply's response time is not a live model response time.

## 6. Put the three parts together

### Which command tests which student?

| Mode | A | B | C |
| --- | --- | --- | --- |
| `grounding` | Your real A | Prepared replacement | Prepared replacement |
| `planning` | Prepared replacement | Your real B | Prepared replacement |
| `manipulation` | Prepared replacement | Prepared replacement | Your real C |
| `e2e` | Your real A | Your real B | Your real C |

All modes run the same overall workflow. They test how a module connects to the others; they are **not separate automatic scorers for the course's A/B/C metrics**.
All three adapters now have `IMPLEMENTED = True`. A/B still require API configuration. If a module is marked unimplemented during development, commands report that error rather than substituting a mock.

Once A, B, and C work individually, run all three together:

```bash
.venv/bin/python -m eval.runner --mode e2e --trials eval/trials/smoke --out runs/team
```

### Read the output

The command prints a new run directory. Each run has:

| File | What to look for |
| --- | --- |
| `run_meta.json` | Mode, shared interface version, and trial source |
| `<trial_id>/trial_record.json` | Which student modules ran, action results, errors, and success checks |
| `<trial_id>/frames/` | Saved camera images, depth, and camera information |
| `metrics.json` | Summary of the metrics currently implemented |

After the team command above, print the latest team's metrics with:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

files = sorted(Path("runs/team").glob("*/metrics.json"))
if not files:
    raise SystemExit("No team metrics yet. Finish the team run first.")
print(files[-1])
print(files[-1].read_text())
PY
```

`claimed_success` means the system thinks it succeeded. `actual_success` means the evaluator checked the simulator's true state. Read both: the system can be wrong about its own success.
Trials using replacements have `infrastructure_check: true` in their records. State which modules were real when reporting an experiment.

### Add your own trial files

The five smoke trials are starting examples. To create a separate test set:

```bash
mkdir -p eval/trials/team
cp -n eval/trials/smoke/smoke_1_standard.yaml eval/trials/team/team_01.yaml
```

Open `eval/trials/team/team_01.yaml` in your editor. Change `id` to `team_01`. It has three main parts:

- `instruction`: what the user says.
- `scene`: starting object/robot positions and a seed used to repeat the setup.
- `expected`: the correct target/destination and expected behavior, used only by the evaluator.

For a first small variation, keep the stone task and change its starting X coordinate from `0.40` to `0.42`. Run:

```bash
# Check the new setup with replacements first.
.venv/bin/python -m eval.runner --mode e2e --trials eval/trials/team --mock-all --out runs/team_setup

# After all real modules are implemented, test the real team.
.venv/bin/python -m eval.runner --mode e2e --trials eval/trials/team --out runs/team
```

Add more YAML files with different IDs and meaningful variations. Do not pass `expected` labels to A, B, or C. A seed makes a chosen setup repeatable; changing only the seed does not automatically move the objects.

### What is still needed for final submission?

| Work | Course minimum / output |
| --- | --- |
| A's evaluation | At least 20 perception/grounding trials; grounding accuracy and failures |
| B's evaluation | At least 20 instructions, including paraphrases and invalid/infeasible requests; planning accuracy |
| C's evaluation | At least 10 starting configurations; grasp/place success and placement error or equivalent |
| Team evaluation | At least 20 randomized full-system trials, changing objects, object positions, robot starts, target areas, and wording |
| Team report/demo | Task 1 review (within 2 pages), setup description, contributions, results, and a 3–5 minute uncut demo with success and a recovery/failure case |

Two shared items need further work before final team evaluation. First, current trial YAML cannot move or resize the target region: it supports objects, robot start, and seed. Add region configuration in shared scene code, or support explicitly selected alternative scene files. Changing `expected.region` alone does not change the physical scene.

Second, `eval/metrics.py` does not calculate all course metrics yet. It reports actual/claimed success and several failure/refusal measures. Keep separate A/B/C evaluation records and add missing aggregation, including average completion time and results under different variations. State whether time means simulation time or real elapsed time, and whether averages include only successful tasks.

Each student should include the model/strategy, commands, raw results, and representative failures in their contribution. Share one working setup and record the Git commit used for final experiments.

## 7. Technical reference: use when needed

### Shared files and code boundaries

| File | Purpose |
| --- | --- |
| `core/types.py` | Shared observations, scenes, plans, and results |
| `core/interfaces.py` | Methods A/B/C must implement |
| `core/validation.py` | Check a plan's format and action requirements |
| `core/action_targets.py` | Resolve a target ID into a position |
| `core/env.py` | Public camera and robot methods |
| `core/skills.py` | Reference motion functions C can reuse |
| `core/verification.py` | Check placement using perception and holding state |
| `core/llm_client.py` | Model requests, parsing, cache, and test transport |

Keep real student code in its own directory. Exchange shared data types rather than importing another student's implementation.
Do not import `core.oracle` or `core.mocks` in real student modules, read simulator object truth through private fields, or use teleport. The simulator's exact answers are **ground truth** and belong in test/evaluation code. The standalone exercises create the world to set up a test; C's real `execute()` receives `env` from the runner.

### Observation fields and units

| Field | Format / meaning |
| --- | --- |
| `rgb` | `uint8[480,640,3]`, red/green/blue image |
| `depth` | `float32[480,640]`, distance along the camera's forward axis in meters; invalid values are NaN |
| `intrinsics` | 3 × 3 camera matrix, often called K |
| `t_world_camera` | 4 × 4 matrix converting a camera-relative point to a world position |
| `frame_id`, `sim_time`, `camera_name` | Image ID, capture time in simulated seconds, and camera name |

World positions use meters with Z up. Angles use radians. Public camera axes are X right, Y down, Z forward. For pixel `(u,v)` with valid depth `d`:

```text
p_camera = d * inverse(K) @ [u, v, 1]
p_world  = (T_world_camera @ [p_camera.x, p_camera.y, p_camera.z, 1])[:3]
```

An image-box center may land on background or an object's front surface. Filter bad depth and explain how you estimate the object center from visible surfaces.
`env.get_obs()` defaults to the head camera (`onboard` is its alias). C can request `left_wrist` and `right_wrist`. `env.get_obs_multi(["head", "left_wrist", "right_wrist"])` captures all three at the same simulation time.

### Scene fields and IDs

`SceneDescription` contains `objects`, `regions`, `caption`, `ambiguities`, `frame_id`, and `sim_time`. Each object/region has:

| Field | Meaning |
| --- | --- |
| `instance_id` | A's stable ID, such as `p0`; not a simulator body ID |
| `name` | Standard class from `assets/objects.yaml`, such as `stone` |
| `kind` | `object` or `region` |
| `bbox_xyxy` | Left/top/right/bottom pixels in the original image |
| `pos_world` | Estimated object center, or region surface center, in world meters |
| `confidence`, `source`, `frame_id` | Confidence 0–1, source label, and input frame ID |
| `region_half_extents_xy` | Optional region half-widths in world X/Y; `(0.08,0.08)` means a 16 cm × 16 cm region |

Use `LOCALIZED` for a reliable 3D position, `UNLOCALIZED` for visible but unreliable position, `AMBIGUOUS` when the target cannot be chosen, and `NOT_FOUND` for no visual evidence. `ground()` normally returns `None` for a missing target.
Do not replace a missing ID with a different same-name object. Keep scene/image timestamps consistent.

### Action reference

| Skill | Input and behavior |
| --- | --- |
| SEARCH | `target="stone"` or another search word; only in `NEEDS_SEARCH` |
| APPROACH | Located ID or `params.pos`; move the base near the target |
| REACH | Located ID; move the gripper to its position; explicit position override allowed |
| GRASP | Located object ID; hand must be empty; grasp and check attachment |
| MOVE_TO | Located ID or `params.pos`; region ID resolves to surface center + 0.18 m |
| PLACE | Region ID while holding; optional `params.object` must match the held ID; resolves to surface point, then C chooses a gripper release pose |
| VERIFY | `params.condition` is `holding`, `object_visible`, or `object_in_region`; the last needs `params.object` and `params.region` IDs |
| STOP | No params; last action if present; cancel movement without automatically releasing |

`params.pos=[x,y,z]` is a world position in meters and overrides position lookup. Missing/ambiguous targets should trigger a new observation or failure, not a guessed replacement.
`resolve_action_position` only resolves coordinates; it does not check all holding conditions or prove movement success.

`set_arm_target`, `set_base_target`, `set_gripper`, and `stop_motion` update commands and return immediately. Time and movement advance through `env.step()`. Gripper opening is 0 closed to 1 open. `env.get_robot_state()` reports measured robot state and attachment, without scene-object truth.

### Placement checks and evaluation details

System placement checks require exact object/region IDs, current 3D evidence, release, region containment, and limited movement between observations 0.2 simulated seconds apart. Action-level VERIFY success does not replace the final check.
The region surface point and half-widths define bounds. For unchanged `red_region`, omitted half-widths use known `(0.08,0.08)` m from the vocabulary. Supply correct sizes for changed/new regions; unknown sizes fail. Current bounds align with world X/Y axes.

The evaluator separately checks the correct grasped object, release, its center within horizontal region bounds, height within `[-0.005,0.12]` m of the surface, and stability for 2 seconds with drift at most 2 cm and speed at most 0.05 m/s. These are project measurement rules, not extra course pass marks.

For A's independent tests, `EvalOracle.associate(scene, camera=obs.camera_name)` matches predicted object boxes to true boxes using overlap (IoU), with threshold 0.30 by default. Call it from evaluation code before stepping the simulation after perception. It associates objects only; evaluate regions separately. Never give true labels to a student module.

Current shared interface version: `CONTRACT_VERSION = 2`. Preserve it in experiment records. More background is in the root [README](../README.md) and [design decisions](DECISIONS.md).

## 8. Common problems

| What you see | What to do |
| --- | --- |
| `.venv/bin/python: No such file` | Check the project folder, then follow installation |
| Missing Python package | Use `.venv/bin/python`; rerun `.venv/bin/python -m pip install -e '.[dev]'` |
| Missing robot asset | Run `bash scripts/fetch_menagerie.sh` from the project folder |
| No simulator window | Normal; open saved PNG files |
| Rendering/EGL error | Tested setup uses Linux and NVIDIA EGL. If OSMesa is installed, try `MUJOCO_GL=osmesa .venv/bin/python -m pytest -q tests/test_env.py`. Otherwise share the full error with the team before changing student code |
| "requires unimplemented module(s)" / exit code 2 | Implement the named module and set `IMPLEMENTED = True`; use `--mock-all` only for setup checks |
| API/configuration error | Check model name, service address, credentials, and no-argument constructor configuration |
| GRASP fails on a visible target | Check A's position and C's reach/attachment result; visible does not necessarily mean reachable |
| Actions succeed but the task fails | Inspect `trial_record.json`, saved images, object identity, release, and final region position |

Owner: backbone (ALL). Updated 2026-09-07.
