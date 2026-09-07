# Watch the A → B → C workflow

This demo moves the gray stone into the red area. You can watch the robot,
see what A detected, read B's plan, and inspect each result from C.
You do not need an API key or a model server for the default demo.

The scene contains a stone, a blue cube, a green bottle, and a red area.
Only the stone is the target of this first demo. The bottle is outside the
first head-camera image, but is visible in the observer view.

## 1. Watch the two recorded examples

From the project directory, start a local web server:

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory runs/abc_workflow
```

Open these addresses in a browser:

- Normal success: <http://127.0.0.1:8765/success/>
- One failed grasp, then a successful retry: <http://127.0.0.1:8765/retry/>

Press **Play**. Click any event on the right to jump to it. You can pause,
seek, change playback speed, and read the JSON below the video.
Press **Ctrl+C** in the server terminal when finished.

These files are local run outputs, not Git-tracked assets. If they are
missing on your computer, use section 2 to generate your own recordings
and open the matching URLs given there.
You can also open an `episode.mp4` directly in a video player.

## 2. Record a new episode

Use the existing project environment and assets. See the root README if
you have not installed them yet. Video recording also needs `ffmpeg` on
your PATH. No new Python dependency is needed for this demo.

```bash
cd /home/jiamo/EE4705/project1.3

# Normal run. The output directory must be new or empty.
.venv/bin/python -m demo.run --scenario success --out runs/my_demo_success

# Explicitly reject the first grasp, then let the backbone retry it.
.venv/bin/python -m demo.run --scenario retry --out runs/my_demo_retry

# Try another supported way to ask for the same task.
.venv/bin/python -m demo.run \
  --instruction "Put the rock in the red zone." \
  --out runs/my_demo_paraphrase

# Without ffmpeg: save observations, JSON and two screenshots, but no video.
# This still needs a working MuJoCo rendering backend.
.venv/bin/python -m demo.run --no-video --out runs/my_demo_data
```

A passing run prints `CLAIMED_SUCCESS: claimed=True, actual=True` and
returns exit code `0`. A task that fails returns `1`; a setup error returns
`2`. The runner refuses to overwrite a nonempty output directory.
Choose a new directory when rerunning a command.

To view your new recording:

```bash
.venv/bin/python -m http.server 8766 --bind 127.0.0.1 --directory runs
```

Then open <http://127.0.0.1:8766/my_demo_success/>.
If a port is already in use, choose another port in the command and URL.

## 3. What each person does

| Person | Simple job | Input | Output | Demo implementation |
| --- | --- | --- | --- | --- |
| A | Find the visible objects and estimate where they are. | Head-camera RGB, depth, camera calibration. | `SceneDescription`: names, perceived IDs, image boxes, world positions, confidence and frame ID. | `perception/demo_rgbd.py` |
| B | Turn the instruction into an ordered list of actions. | Instruction text and A's scene. On replanning: action results and held-object state. | `Plan`: status, actions, target IDs and parameters. | `planner/demo_rules.py` |
| C | Carry out each action and say whether it worked. | One `Action`, the public `RobotEnv`, and A's perception interface. | `ExecutionResult`: success, error code, new observation ID and details. | `executor/demo_skills.py` |

For this example, B produces:

```text
APPROACH stone → GRASP stone → MOVE_TO red area
              → PLACE → VERIFY → STOP
```

The actual plan uses A's IDs, such as `p0` and `p2`. These are not MuJoCo
body IDs. A gives positions in the world frame, in meters. B provides a
transport waypoint above the red area and a support position for placing.
C turns those targets into bounded base and arm movement.

After grasping, C lifts the stone. After releasing it, C lifts the empty
gripper and returns the base to its initial observation position. This
helps A see the placed stone during verification.

The backbone handles plan validation, action retries and final visual
verification. A separate evaluator checks the actual object identity,
region bounds, release, and stability for 2 seconds. Its ground truth
never goes into A, B or C. It is possible for the visual claim and the
actual result to disagree; the recording shows both.

In the retry example, the first `GRASP` deliberately returns
`GRASP_MISSED` **before attempting motion**. The backbone runs the action
again, and C uses normal simulated control. This tests the error and
retry path; it is not evidence of detecting a real contact-grasp failure.

## 4. For Student B: try your planner with working demo A and C

You can develop B without waiting for the other students. Start by opening
`episode.json` and looking at `A.scene`, `B.plan` and `C.end` in `events`.
This gives you real example inputs and outputs for your planner.

Implement `plan()` and `replan()` in `planner/student_b.py`. Keep their
existing interfaces. Set `StudentBPlanner.IMPLEMENTED = True` only when
you have implemented them. Configure any model client in your own module;
this demo does not create an API connection for you.

Then run:

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m demo.run --student B --out runs/my_b_plan
```

This uses **demo A + your B + demo C**, with the same video and JSON
recording. The runner checks the stub flag and reports an error instead
of silently falling back to the rule planner.

Look for these signs:

1. `backbone.plan_invalid`: the action sequence or parameters broke the contract.
2. A failed `C.end`: inspect the error, action target and position.
3. `backbone.final_verification`: the final visual check.
4. `actual.checks`: the evaluator's separate checks. A valid plan alone
   does not prove that the robot can execute it successfully.

The expected object stays **stone**, even if your planner chooses another
object. This prevents a wrong-object plan from passing just because it
placed something in the red area.

The other students can use the same switch:

```bash
.venv/bin/python -m demo.run --student A --out runs/my_a_perception
.venv/bin/python -m demo.run --student C --out runs/my_c_skills
.venv/bin/python -m demo.run --student A --student B --student C --out runs/my_abc
```

Each selected student module must already be implemented. The demo calls
its default constructor. The injected `retry` scenario only works with
demo C. This first runner has no interactive clarification dialog; if a
planner asks for clarification, the episode ends with that need unresolved.

## 5. Files and tests

Each output directory contains:

| File | What to use it for |
| --- | --- |
| `index.html` | Browser replay, event navigation, A/B/C data panels. |
| `episode.mp4` | Annotated robot video; omitted with `--no-video`. |
| `episode.json` | Configuration, module labels, all A/B/C and backbone events, both clocks, and final checks. |
| `start.png`, `final.png` | Quick visual inspection. |
| `observations/*_rgb.png` | Actual camera images used as evidence. |
| `observations/*_depth.npy` | Depth arrays in meters; invalid depth is `NaN`. |
| `observations/*_meta.json` | Camera calibration, frame ID and simulation time. |
| `ffmpeg.log` | Video encoder diagnostics. |

Run the demo regression tests and the full backbone suite:

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m pytest -q tests/test_demo_workflow.py
.venv/bin/python -m pytest -q
```

The demo tests check image-based localization after an object position
change, missing visual evidence, plan references, missing/ambiguous
targets, refusal, replanning while holding, and actual simulated success
for normal and retry episodes. They also check the saved records.

## 6. What this version does and does not establish

The default A uses color thresholds, depth geometry, known object sizes
and a known tabletop workspace. It supports one visible object per
class/color. It is not a general detector or tracker. The red region is
static: A may reuse its last fully observed center when it is partly
occluded. Confidence numbers are heuristic, not calibrated probabilities.

The default B handles a small set of positive stone-moving instructions
with rules. It does not call an LLM and should not be used to evaluate
general language understanding. Unknown or complex instructions need a
proper planner. The included scene and successful examples are fixed;
they do not establish robustness to randomized scenes.

C uses continuous simulation and the public motion API, with the
backbone's weld attachment. It does not teleport objects. It does not
validate friction-only grasping, walking, or real robot hardware.

The replay contains display pauses so that outputs can be read. Its
`sim_time` and `video_time` are separate. This is not the required
3–5 minute uncut assessment demo, and these two episodes do not replace
the course's trial counts or AI-model requirements. Use the
[student guide](STUDENT_README.md) for the full student tasks and tests.
