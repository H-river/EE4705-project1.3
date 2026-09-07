# C refinement and A preparation — 7 September 2026

Student C now has a working executor with feedback checks and bounded recovery.
Its ten development trials completed successfully. After freezing the control
code, ten additional layouts completed **9/10**, with the failed bottle case
retained. A now has a Qwen-VL adapter, offline tests and twenty labelled camera
samples; **no real A model accuracy is claimed**.

All evidence is under:

```text
/home/jiamo/EE4705/project1.3/runs/c_refine_20260907/
```

The local replay page is [http://127.0.0.1:8768/](http://127.0.0.1:8768/).
Start it again with:

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m http.server 8768 --bind 127.0.0.1 \
  --directory runs/c_refine_20260907
```

## 1. Changes to C

The previous student C stub is now `StudentCExecutor`, backed by
`executor/closed_loop.py`. Demo C shares this implementation, with demo-only
failure injection in its wrapper.

- **SEARCH / APPROACH / REACH:** require fresh localization, bound scanning
  and movement, stop residual commands, and check actual TCP arrival.
- **GRASP:** re-observe the exact ID, open first, allow one local retry after
  a miss, then verify a 12 cm lift and continued attachment.
- **MOVE_TO:** check arrival and holding; after timeout/unreachable motion,
  try one base re-parking and another reach. Failures stay visible to B.
- **PLACE:** account for the grasp offset, open before detaching, retreat,
  and check released placement on two fresh observations. Try at most two
  nearby camera headings when the exact object/region is not visible.
- **VERIFY / STOP:** distinguish visibility from valid depth, check tracked
  holding identity, and measure settling before reporting STOP success.

A failed GRASP or PLACE can already have attached or released the object.
The backbone now records `partial_action_state`, updates B's held/released
context, and replans instead of retrying an action with obsolete preconditions.
Each C action records measured error, attachment state, elapsed simulation
time and recovery details. The evaluator records final object coordinates
and XY distance from the region center after its separate stability check.

The contract-v2 dataclasses, 12 mm TCP tolerance, visual placement geometry,
and oracle success criteria are unchanged. A/B/C do not gain oracle access.
B's model, prompt and compiler were not changed by this task.

## 2. C measurements

Both suites use **ground-truth mock A + rule mock B + Student C**. The arm
moves through MuJoCo's controller; grasping uses weld attachment. These are
C component experiments, not live ABC, contact-only or hardware experiments.
The generic runner marks records containing mocks as infrastructure/component
checks. That label does not mean C was teleported: C is the student executor.

| Metric | Development layouts | Additional frozen layouts |
| --- | ---: | ---: |
| Trials completed, claimed and actual | 10/10 | 9/10 |
| Completed without any failed action or internal recovery | 7/10 | 4/10 |
| GRASP action attempts passed | 10/11 (90.9%) | 12/15 (80.0%) |
| MOVE_TO action attempts passed | 10/11 (90.9%) | 11/19 (57.9%) |
| PLACE action attempts passed | 10/10 (100%) | 9/11 (81.8%) |
| STOP action attempts passed | 11/11 | 11/11 |
| Mean final XY distance, all ten cases | 1.17 cm | 2.59 cm |
| Maximum final XY distance, all ten cases | 2.71 cm | 16.79 cm |
| False success claims | 0 | 0 |

Every action attempt, including failed attempts and retries, stays in the
skill denominator. Local recovery is reported separately. A placement action
can fail verification even after releasing the object, so its action score
and final physical task result need not be equal. The failed additional case
is included in the XY mean and maximum even though it never released.
Object height above support is stored separately and is not called XY error.

The development layouts were used to improve C. The additional layouts were
generated from seed `47050709` after those changes. The control files and
YAML labels were hashed in `c_evaluation_freeze.json` before the additional
run. C was not changed after seeing its result. Ten additional examples are
a small estimate; they do not establish a general 90% execution guarantee.

### Development cases

| Case | Actual result | Final XY distance (cm) | Failed actions | Internal recovery |
| --- | --- | ---: | --- | --- |
| c_01_stone | Pass | 0.40 | None | None |
| c_02_stone | Pass | 0.49 | None | None |
| c_03_stone | Pass | 0.41 | None | PLACE |
| c_04_stone | Pass | 2.61 | MOVE_TO: TIMEOUT | MOVE_TO twice across two action attempts |
| c_05_stone | Pass | 0.52 | GRASP: TIMEOUT | None; backbone retry |
| c_06_cube | Pass | 2.71 | None | None |
| c_07_cube | Pass | 0.62 | None | None |
| c_08_cube | Pass | 0.81 | None | None |
| c_09_bottle | Pass | 1.46 | None | None |
| c_10_bottle | Pass | 1.73 | None | None |

### Additional cases

| Case | Actual result | Final XY distance (cm) | Failed action types |
| --- | --- | ---: | --- |
| c_eval_01_stone | Pass | 0.90 | PLACE_FAILED; GRASP TIMEOUT |
| c_eval_02_cube | Pass | 0.80 | None |
| c_eval_03_bottle | Pass | 1.00 | GRASP TIMEOUT |
| c_eval_04_stone | Pass | 0.47 | None |
| c_eval_05_cube | Pass | 2.20 | None |
| c_eval_06_bottle | Pass | 2.37 | None |
| c_eval_07_stone | Pass | 0.45 | None |
| c_eval_08_cube | Pass | 0.32 | PLACE TIMEOUT |
| c_eval_09_bottle | Fail | 16.79 | GRASP TIMEOUT; MOVE_TO UNREACHABLE on 8 attempts |
| c_eval_10_stone | Pass | 0.58 | None |

`c_eval_09_bottle` first needed a grasp retry, then held the correct bottle.
The collision-aware IK rejected transport. Repeated parking converged to
essentially the same base configuration and did not resolve the collision.
The task ended holding the object, outside the region, with no success claim.
This is a retained failure, not a removed outlier. A useful next C change is
to choose a genuinely different parking heading after repeated failure and
test it on fresh layouts. The current `base_repositioned` diagnostic means
that the parking primitive reached its target; it does not prove the new
pose differs substantially from the previous attempt.

## 3. Retained evidence and recordings

| Evidence under the run root | Result / purpose |
| --- | --- |
| `manipulation_smoke/20260907_144648_manipulation` | 4/5 before camera-view recovery; retained failure |
| `manipulation_smoke_refined/20260907_144902_manipulation` | 5/5 after camera-view recovery |
| `variation_r1/20260907_145245_manipulation` | 9/10 before transport recovery; retained stalled-arm failure |
| `variation_r2/20260907_145646_manipulation` | 10/10 after transport recovery |
| `final_with_geometry/20260907_151251_manipulation` | 10/10, final per-action and oracle geometry records |
| `additional_evaluation/20260907_152449_manipulation` | 9/10 on frozen additional layouts, including the failed bottle |
| `episodes/stone_final/` | Standard stone transfer with final C code |
| `episodes/cube_final/` | Blue-cube transfer with final C code |
| `episodes/c04_rgbd_final/` | Changed stone layout; a transport failure followed by recovery; claimed/actual pass |
| `episodes/bottle_edge_rgbd/` | Same edge-layout bottle with RGB-D demo A; physical placement passed, but the workflow did not claim success |

Recordings use **RGB-D demo A + rule-based demo B + Student C**, so they are
separate from the ground-truth-mock-A component suites above. The bottle
recording is a useful false-negative example: the final visual check lost
sight of an exact instance, demo B tried another grasp, and C rejected the
now-unlocalized target. The independent evaluator nevertheless found the
correct bottle released and stable in the region. Its result is
`claimed=False, actual=True`; it must not be described as a successful ABC
workflow. Live Qwen B was not used in these new recordings.

Each episode directory contains `episode.mp4`, `episode.json`, RGB-D inputs,
snapshots and a browser replay. Earlier recordings and failed trial bundles
remain in place. Video decode checks and visual inspection were performed.

## 4. Work completed for A

`StudentAPerception` now implements `describe`, `ground` and `reset` using:

1. Qwen-VL image boxes, target selections and scene answers.
2. Strict local parsing, one bounded output repair, and explicit API errors.
3. RGB-D world-coordinate fitting within the model boxes, using known project
   colors and class dimensions. Invalid geometry remains UNLOCALIZED.
4. Local perceived IDs, current frame/time propagation, missing-object
   handling and conservative association for similar objects.
5. Input/response audit files, source labels, usage and sanitized errors.
   Identical-image reuse still recomputes geometry using current depth.

Default visual model: `qwen3-vl-plus`, JSON-object output, thinking disabled.
A has separate model settings from B. Model boxes use 0..1000 coordinates;
Python produces pixel boxes and world coordinates. The model never receives
truth positions or expected answers.

The offline CLI sample in `a_offline_verified/` detects two objects and one
region from **hand-labelled responses**, performs the real depth conversion,
and reports zero real API requests. It is a parser/geometry example, not a
model accuracy result. The fixed-sample localization test checks error below
4 mm for the two object centers and region support point.

`a_dataset_20/dataset.json` contains 20 captured images with hashes and
independent same-capture oracle labels: 10 unique, 7 missing and 3 ambiguous
target cases. Capture required zero model calls. `eval.grounding run` can
score real A on these files with all cases in the denominator, box-based
identity matching, per-status scores, and 3D error sample counts. Scene
answers are retained for manual review; VQA accuracy is not automatically scored.

The earlier temporary Qwen credential handoff had been cleaned up, and no
new credential file was available in this session. Therefore the real A API,
its detection/grounding accuracy and full live ABC were **not tested**.
No rule model was substituted and called a real VLM.

## 5. Reproduce and continue

Final offline regression: **221 passed, 1 skipped in 42.83 s**. The skip is
the explicitly experimental contact-only grasp check. The full log is
`runs/c_refine_20260907/pytest_release.txt`. Tests cover new C postconditions
and partial-action handoff, A parsing/tracking/depth/provenance, and evaluator
denominators, fixture rejection and capture-hash validation. The final C
source hashes still match the pre-evaluation freeze.

Start with the plain-English [C guide](../STUDENT_C_README.md) or
[A guide](../STUDENT_A_README.md). Both include input/output tables and commands.

```bash
cd /home/jiamo/EE4705/project1.3

# Offline implementation and integration checks
.venv/bin/python -m pytest -q

# C development and additional layouts (use separate output roots)
.venv/bin/python -m eval.runner --mode manipulation \
  --trials eval/trials/student_c --out runs/my_c_development
.venv/bin/python -m eval.runner --mode manipulation \
  --trials eval/trials/student_c_evaluation --out runs/my_c_additional

# Use the exact run directory printed by either runner
.venv/bin/python -m eval.skill_metrics runs/my_c_development/REPLACE_WITH_PRINTED_RUN

# A offline adapter exercise, without a model request
.venv/bin/python -m perception.run --offline-demo \
  --out "runs/my_a_offline_$(date +%Y%m%d_%H%M%S)"

# Prepare labelled A inputs; this also makes no model request
.venv/bin/python -m eval.grounding capture \
  --out "runs/my_a_dataset_$(date +%Y%m%d_%H%M%S)"
```

The supplied assignment asks A for visual AI, scene/VQA/grounding behavior
and at least 20 varied trials; C needs execution skills, feedback,
recovery/failure reporting and at least 10 varied configurations. The current
C baseline and measurements support that component work. A still needs real
model evaluation. These component experiments do not replace Task 5's
randomized full-system evaluation and uncut 3–5 minute demonstration.
Source: the supplied `EE4705_Project1.3 S1 AY2627.pdf`, Task 2 and Task 4 sections.

Remaining limitations: weld attachment, sliding-base approximation, limited
motion planning, incomplete recovery for some collision configurations,
known-class/color assumptions in A, and unverified live A / full live ABC.
