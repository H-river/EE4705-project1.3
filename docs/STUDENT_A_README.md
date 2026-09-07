# Student A: turn camera images into useful objects

Your job is to tell B and C what the robot can see and where it is. Start
with one saved image. There is no need to move the robot or finish B/C first.

A working interface is now in `perception/student_a.py`. It sends **RGB only**
to Qwen-VL. The model returns object names, image boxes and a text answer.
Python uses depth and camera parameters to estimate 3D positions and assigns
stable IDs. Missing objects are not copied from an old frame into a new scene.

**Current status:** the interface and offline tests pass. The real Qwen-VL
API and visual accuracy have not been tested in this refinement session.
`IMPLEMENTED = True` means the adapter exists, not that Task 2 is complete.

## 1. Try the interface without an API key

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m pytest -q tests/test_student_a.py

a_out="runs/my_a_offline_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m perception.run --offline-demo --out "$a_out"
```

Open `$a_out/overlay.png` to see the boxes and IDs. Read `scene.json` and
`grounded.json` for the output sent to the other students. `summary.json`
will say `source: fixture` and `real_api_requests: 0`.

This uses a real simulated RGB-D image with **hand-labelled model responses**.
It checks parsing, depth conversion and handoff. It does not measure how well
Qwen detects objects or answers questions. The fixed target is the gray stone;
the bottle is outside this camera view and is not returned.

## 2. Configure real Qwen-VL

Use the OpenAI-compatible Qwen endpoint, not the Anthropic-compatible one.
The vision adapter has separate model settings from B:

```bash
export EE4705_VLM_BASE_URL='https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
export EE4705_VLM_MODEL='qwen3-vl-plus'
```

If `DASHSCOPE_API_KEY` or `EE4705_QWEN_API_KEY` is already exported in this
terminal, A can reuse it. Otherwise enter the key without displaying it:

```bash
read -rsp 'Qwen API key: ' DASHSCOPE_API_KEY
export DASHSCOPE_API_KEY
printf '\n'
```

Do not put a key in a Python file or commit it. `EE4705_VLM_API_KEY` can be
used instead if A needs a separate credential. The adapter requires an
explicit base URL and does not change providers automatically.

Qwen-VL uses `json_object` output and `enable_thinking=False`, with local
schema checks. It does **not** inherit B's text model or
`EE4705_QWEN_OUTPUT_MODE`. These settings follow the provider's
[structured output documentation](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)
and [vision documentation](https://www.alibabacloud.com/help/en/model-studio/vision).
Availability still needs a real call using your account and region.

## 3. Test one real camera image

First run the camera exercise in the [student guide](STUDENT_README.md#first-exercise-save-a-camera-image).
It writes three matching files in `runs/student_a_intro/`. Then run:

```bash
.venv/bin/python -m perception.run \
  --rgb runs/student_a_intro/head_rgb.png \
  --depth runs/student_a_intro/head_depth.npy \
  --meta runs/student_a_intro/head_meta.json \
  --query 'What objects are visible, and what colors are they?' \
  --target 'the gray stone' \
  --out "runs/my_a_live_$(date +%Y%m%d_%H%M%S)"
```

This makes two model requests: scene description and target grounding. A
response repair or transient HTTP retry may add requests. Original responses,
usage, errors and the camera inputs are kept under `audit/`. An API error
returns exit code 2 and `error.json`; it is not reported as “object missing”.
Use a new output directory for every run so previous evidence stays intact.

## 4. Input and output

| Interface | Input | Output |
| --- | --- | --- |
| `describe(obs, query=None)` | `Observation` plus an optional image question | `SceneDescription`: objects, regions, caption/answer, ambiguities, frame ID and time |
| `ground(obs, target)` | The same camera input plus a phrase or known perceived ID | One `GroundedObject`; `None` if missing; `AMBIGUOUS` if multiple candidates match |
| `reset()` | New episode | Clear IDs and image-response reuse |

`Observation` contains RGB `uint8 (480,640,3)`, metric optical-axis depth
`(480,640)`, camera intrinsics `K (3,3)`, camera-to-world transform `(4,4)`,
`frame_id`, `sim_time` and `camera_name`. All files must come from the same
capture. The saved metadata files record these values.

A model response looks like this:

```json
{
  "detections": [
    {"name": "stone", "color": "gray", "bbox": [800, 410, 910, 550], "confidence": 0.9}
  ],
  "selected": [0],
  "answer": "A gray stone is visible."
}
```

Boxes use **0..1000 normalized coordinates**. Python converts them to pixels.
`selected` contains zero-based detection indices, not tracking IDs. It can be
empty or contain multiple entries. The model never supplies world coordinates.

The resulting object has an ID such as `a2`, a pixel box and, if reliable,
`pos_world` in meters. The position is the object's center; for a region it is
a support-surface point. `frame_id` always belongs to the current observation.

| Status | Meaning for B/C |
| --- | --- |
| `LOCALIZED` | Visible with a usable 3D position |
| `UNLOCALIZED` | Visible, but depth/geometry/confidence is insufficient; do not reach or grasp |
| `AMBIGUOUS` | The target phrase or cross-frame identity cannot be resolved; do not guess |
| `None` from `ground` | No detection matches the target phrase |

## 5. What is implemented, and what needs your work

The parser checks required fields, classes, box bounds, confidence, and
selection indices. It allows at most one output repair. Reusing a response
for identical RGB/query does not reuse old depth: geometry and frame IDs are
computed from the current observation again.

Geometry currently uses known colors and object sizes from this project.
Color masks only refine pixels **inside model boxes**; they cannot create new
detections. Bad depth, clipped boxes or unreliable geometry produce
`UNLOCALIZED`. This baseline is not designed for arbitrary objects, large
occlusion, unknown dimensions or different lighting.

IDs are stable for unique class/color objects within an episode. When several
objects share class/color, association needs separated 3D positions (15 cm
matching gate, 3 cm separation margin); uncertain matches stay ambiguous.
Unique class/color association assumes the same physical instance remains in
the scene. Replacing it with an identical object can violate that assumption.

Your next steps:

1. Run the real API on one image and inspect its raw boxes and answer.
2. Collect **at least 20 labelled trials** with varied objects, positions and
   camera directions. Include missing and ambiguous targets and scene questions.
3. Record target selection, box quality, 3D error and response time separately.
   Count API failures and invalid responses; do not silently remove them.
4. Improve box prompting, occlusion handling and identity tracking using a
   development set, then evaluate on fresh labelled images.
5. Connect A to demo B and Student C only after single-image tests are useful:

```bash
.venv/bin/python -m demo.run --student A --student C \
  --out "runs/my_ac_$(date +%Y%m%d_%H%M%S)"
```

That command uses live A, rule-based demo B and Student C. It can make many
vision requests as C re-observes during execution. Full live ABC adds
`--student B`; neither live combination has been validated in this session.

`eval.runner --mode grounding` can exercise A through the existing workflow,
but its final task success is not a stand-alone target-grounding accuracy
metric. Do not use those two scores interchangeably.

## 6. Prepare and score 20 labelled images

You can prepare an evaluation dataset without a key:

```bash
cd /home/jiamo/EE4705/project1.3
a_dataset="runs/my_a_dataset_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m eval.grounding capture --out "$a_dataset"
```

This captures 20 images and matching depth/camera files. `dataset.json` stores
queries, truth labels, and hashes of the input files. Labels come from the
evaluation oracle in the same capture: 10 unique targets, 7 missing targets
and 3 ambiguous targets. No model is called by `capture`.

After setting the Qwen-VL endpoint and key, evaluate A:

```bash
.venv/bin/python -m eval.grounding run \
  --dataset "$a_dataset/dataset.json" \
  --out "runs/my_a_evaluation_$(date +%Y%m%d_%H%M%S)"
```

This makes one scene-description and one grounding request per case: normally
40 requests, plus any bounded retries/repairs. It verifies capture hashes
before sending requests. A receives the camera input and query; it does not
receive expected labels, simulator names or truth coordinates.

`summary.json` reports target-selection accuracy with **all 20 cases** in the
denominator, and separate unique/missing/ambiguous counts. Selection is matched
to the actual object by image-box IoU (at least 0.30), not by ID string. The
3D error is reported only for correctly selected, localized unique targets,
with its sample count. API/parse errors remain failed cases. Every case has
its own JSON and raw model audit. Exit code 1 means some cases did not pass.

Scene answers are saved for human review; this tool does not score VQA answers.
It resets A's tracking for each case, so cross-frame identity tracking needs
separate episode tests. Keep this dataset as a development set if you use its
mistakes to improve prompts; collect fresh images for a final accuracy claim.
