# Student A: turn camera images into useful objects

Your job is to tell B and C what the robot can see and where it is.

**Current status:** the interface is implemented, live-tested against the real
Qwen-VL API, and merged into the full `e2e` branch alongside Student B and C's
work (22 tracked commits across `perception/`, `executor/`, and the shared
backbone — see [`docs/night_run/CHANGES.md`](night_run/CHANGES.md) for the full
list with owners and rationale). Task 2's required deliverables (interface
design, VLM connection, grounding with both failure cases, and a ≥20-trial
accuracy evaluation) are complete, with results and open limitations documented
in §5–6 below. `IMPLEMENTED = True` reflects a tested, integrated module, not
just an adapter that exists.

## 1. Try the interface without an API key

```bash
cd ~/EE4705-project1.3
source .venv/bin/activate
sudo apt install ffmpeg                 #Need to install this for rendering MuJoCo backend
.venv/bin/python -m pip install -e ".[dev]"    # requires Python >=3.10; see README.md
.venv/bin/python -m pytest -q tests/test_student_a.py

a_out="runs/my_a_offline_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m perception.run --offline-demo --out "$a_out"
```

Open `$a_out/overlay.png` to see the boxes and IDs. Read `scene.json` and
`grounded.json` for the output sent to the other students. `summary.json`
will say `source: fixture` and `real_api_requests: 0`.

The full test suite now includes dedicated tests for every fix in §5:
`test_student_a_bbox.py`, `test_student_a_palette.py`,
`test_student_a_off_table.py`, `test_student_a_partial_region.py`,
`test_student_a_partial_object.py`, `test_student_a_phantom_track.py`, and
`test_tracking_hint.py`.

## 2. Configure real Qwen-VL

Use the OpenAI-compatible Qwen endpoint, not the Anthropic-compatible one:

```bash
export EE4705_VLM_BASE_URL='https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
export EE4705_VLM_MODEL='qwen3-vl-plus'
read -rsp 'Qwen API key: ' DASHSCOPE_API_KEY
export DASHSCOPE_API_KEY
printf '\n'
```

Do not put a key in a Python file, commit it, or paste it into any chat or
log — treat a leaked key as compromised and rotate it immediately.
`EE4705_VLM_API_KEY` can be used instead if A needs a separate credential.

## 3. Test one real camera image

```bash
.venv/bin/python -m perception.run \
  --rgb runs/student_a_intro/head_rgb.png \
  --depth runs/student_a_intro/head_depth.npy \
  --meta runs/student_a_intro/head_meta.json \
  --query 'What objects are visible, and what colors are they?' \
  --target 'the gray stone' \
  --out "runs/my_a_live_$(date +%Y%m%d_%H%M%S)"
```

Use a new output directory for every run so previous evidence stays intact.

## 4. Input and output (contract v4)

| Interface | Input | Output |
| --- | --- | --- |
| `describe(obs, query=None, *, hint=None)` | `Observation`, an optional question, and an optional `TrackingHint` | `SceneDescription`: objects, regions, caption/answer, ambiguities, frame ID and time |
| `ground(obs, target)` | The same camera input plus a phrase, a known instance ID, or a `"<color> <class>"` phrase | One `GroundedObject`; `None` if missing; `AMBIGUOUS` if multiple candidates match |
| `recall(name, color=None)` | A class name | Best-effort memory-only lookup (see §5); **not** part of the `Perception` contract — never a substitute for a live call |
| `reset()` | New episode | Clears IDs, image-response reuse, and memory |

`TrackingHint` (contract v3, restored in round 7): the orchestrator passes
`held_object_id`/`held_pos_world` and a released instance's expected position
across a GRASP→PLACE cycle, so a released object is re-identified at its known
drop point rather than risking a fresh, disconnected ID. A keeps the most
recent hint in force across calls that don't supply one.

A `"<color> <class>"` grounding query (e.g. `"dark_red stone"`) is matched on
**both** fields directly in A's own code, not left to the model's own
selection — the model was observed selecting every stone regardless of color
9 times in a row in one trial before this fix.

| Status | Meaning for B/C |
| --- | --- |
| `LOCALIZED` | Visible with a usable 3D position (may carry `attributes['pos_basis']`: `'depth_patch_partial'` for an edge-clipped box, or `'held_hint'` while grasped) |
| `UNLOCALIZED` | Visible, but depth/geometry/confidence is insufficient; do not reach or grasp |
| `AMBIGUOUS` | The target phrase or cross-frame identity cannot be resolved; do not guess |
| `None` from `ground` | No detection matches the target phrase |

## 5. What is implemented, and what was tried and reverted

**Geometry and detection filtering:**
- Edge-clipped boxes (region or object) are localized from the visible part of
  the depth patch when ≥50% of it is valid, instead of a flat `UNLOCALIZED` — a
  correct PLACE was previously unverifiable whenever the target region sat at
  the image boundary after the robot repositioned to check it. **This is
  experimental for regions** (biased toward the visible side) and still needs
  explicit sign-off before being treated as fully trusted.
- A detection whose 3D center falls outside a plausible table-height band is
  dropped before tracking — across 1,495 archived frames this removed 331
  phantom "stones"/"regions" hallucinated on the floor or the robot's own
  shadow.
- A detection with a class/color pair that cannot physically exist (checked
  against the real published palette in `assets/objects.yaml`, not a hardcoded
  guess) is dropped or corrected — this is the fix for the recurring "red cube"
  and stone/region color-conflation failures described below.
- A single detection of a class in one frame is no longer forced `AMBIGUOUS`
  just because a stale, possibly-hallucinated track exists from an earlier
  frame.

**Robustness** (same output for a valid input; a previously-fatal bad input is
now handled instead of crashing the episode):
- A response with a bounding box spanning nearly the whole frame, or a
  degenerate/placeholder box — **including one the model selected as its
  actual answer**, which used to make SEARCH fatal — is dropped and `selected`
  is re-mapped, instead of raising and burning the one allowed repair.
- A provider content-filter refusal (e.g. an HTTP 400 "inappropriate content")
  is returned as a typed, uncached result: `describe()` returns an empty frame
  instead of raising, so a genuinely bad single frame doesn't fail the whole
  episode.

**Memory** (this author's addition, not yet promoted to a full interface field
— deliberately left as an open interface decision rather than merged
silently):
- `_tracks` records each instance's last confirmed position and timestamp.
  `attributes['memory_last_localized_*']` is attached only to an object still
  detected this frame but currently `UNLOCALIZED` — it never changes `status`
  or `pos_world`, so it cannot cause a guess.
- `recall(name, color=None)` is a standalone, non-contract lookup for the same
  data, for a caller that explicitly wants a best-effort last-known position
  (e.g. biasing where SEARCH looks first) while still requiring a live
  re-confirmation before acting on it.

**Tried and reverted, with reasons** (kept here because a negative result is
still a result):
- A grounding-prompt reframe (*"identify which of your own detections matches
  this description"* instead of *"find this named target"*) produced a large,
  clean improvement on missing-target accuracy in isolation (50%→80% on one
  paired test). It was **not kept** in the final merge: the team's code-level
  fix (`colour_class_query`, `normalise_class_color`) addressed the same
  failure mechanism more reliably and deterministically, and further
  prompt-only additions (a duplicate-instance-count instruction, an
  identical-bbox guard) measured **no net accuracy gain** across repeated runs
  on the same dataset and introduced new instability in two previously-stable
  cases. Longer, denser system prompts appear to reduce reliability on
  unrelated cases even while fixing a targeted one — documented here as a
  real, citable limit of prompt-only iteration, not an unfinished task.
- A same-response cross-check (comparing `ground()`'s "missing" verdict
  against its own detection list before returning `None`) was implemented and
  verified correct in isolation, but did not make it into the merge once
  `colour_class_query` made the underlying disagreement it was designed to
  catch far less frequent.

**Known, characterized, unresolved limitations:**
- A region/object pair sharing near-identical image coordinates at the image
  boundary (a "gray stone" hallucinated at the exact pixel box of the actual
  red region) reproduced in **5 of 5** repeated live runs on one specific
  scene — a stable, deterministic model limitation, not fixed by any prompt or
  code change attempted so far.
- The model misses one of two same-class objects in a scene roughly **20%** of
  the time on duplicate-instance tests, despite an explicit system-prompt
  instruction to report all instances — a detection-completeness limit that no
  selection-time logic (prompt or code) can correct after the fact.
- **Response non-determinism:** the identical scene, prompt, and dataset
  produced a different pass/fail verdict on at least 2 cases across
  otherwise-identical repeated runs, at `temperature=0`. Treat any single
  evaluation run as one sample, not a stable ground truth, without repetition.

## 6. Evaluate perception and grounding

Prepare a balanced, labelled dataset without a key:

```bash
a_dataset="runs/my_a_dataset_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m eval.grounding capture --out "$a_dataset"
```

This captures 30 images (**10 unique, 10 missing, 10 ambiguous** — rebalanced
from an original 10/7/3 split specifically to give the missing/ambiguous cases
enough raw material to be statistically meaningful; the previous split
under-tested both). No model is called by `capture`.

Score it against the real API:

```bash
.venv/bin/python -m eval.grounding run --dataset "$a_dataset/dataset.json" \
  --out "runs/my_a_evaluation_$(date +%Y%m%d_%H%M%S)"
```

Selection is matched to the actual object by image-box IoU (≥0.30), not by ID
string. Reported Target Grounding Accuracy across repeated live runs on this
dataset has ranged **76.7%–86.7%**, depending on prompt/code version and
run-to-run model variance — see the "tried and reverted" notes in §5 before
citing a single number as final.

**Additional metrics**, each reusing an existing `eval.grounding` run wherever
possible (no extra model calls) rather than recapturing:

```bash
.venv/bin/python -m eval.a_scene_description --eval-dir <existing eval run> \
  --dataset "$a_dataset/dataset.json"
```
Scene Description Accuracy: exact object-mention recall + hallucination check
against ground truth. Has surfaced the same region/stone confusion found in
grounding failures, independently, across a larger sample.

```bash
.venv/bin/python -m eval.a_bbox_precision --eval-dir <existing eval run> \
  --dataset "$a_dataset/dataset.json"
```
Bounding Box Precision: IoU between predicted and ground-truth boxes on
correctly-selected unique targets. Consistently ≥0.80 mean IoU, comfortably
above a 0.5 "tight box" threshold.

```bash
.venv/bin/python -m eval.a_spatial capture <dir> \
  --n-containment 20 --n-distance 20
.venv/bin/python -m eval.a_spatial run --dataset <capture>/dataset.json --out <dir>
```
Spatial Reasoning Accuracy: containment and relative-distance VQA questions
with geometrically-computed ground truth. Only tested on well-separated,
unambiguous placements so far — not yet on boundary/near-tie cases.

```bash
.venv/bin/python -m eval.a_vqa_check --eval-dir <existing eval run> \
  --dataset "$a_dataset/dataset.json"
```
Tests the exact three example VQA questions from the project brief (*"What
objects are nearby?"*, *"Where is the stone?"*, *"Which object is inside the
red area?"*). Found that semantically equivalent phrasings of the same
question can disagree by 10–25 percentage points, and that `ground()`'s
structured answer and `describe()`'s free-text answer can disagree on the same
underlying fact — an adaptability limitation worth reporting alongside the
headline accuracy number, not papered over by it.

Keep a dataset as a development set once you've used its failures to change
anything; capture a fresh one before citing a final accuracy claim, since
`eval.grounding capture`'s scene layouts are otherwise deterministic
(identical seed → identical scenes, confirmed by matching dataset hashes
across "fresh" captures) and reusing one after tuning against it overstates
the result.
