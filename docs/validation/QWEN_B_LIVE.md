# Student B: live Qwen validation

Date: September 7, 2026. Workspace: `/home/jiamo/EE4705/project1.3`.
The requested target was **B planning accuracy above 90%**, meaning the correct
target, destination, status and action order. Work started at 05:40 UTC.

**Result: 32/32 (100%) on the predefined evaluation set.** Every stage's first
response also passed. This is a measured result on 32 curated cases with a small
task vocabulary; it does not establish a 90% success guarantee on new scenes or
arbitrary instructions. Student A/C implementations and hardware are outside
this evaluation.

## Measured results

| Run | Final accepted plans | First responses | Use |
| --- | --- | --- | --- |
| Development, prompt v2 | 11/13 (84.6%) | 10/13 (76.9%) | Diagnose failures |
| Development, prompt v3 | 13/13 (100%) | 12/13 (92.3%) | Check the refinement |
| Evaluation, frozen prompt v3 | **32/32 (100%)** | **32/32 (100%)** | Independent set, run once |

The final evaluation used **38 live model calls**, including initial plans for
six recovery cases. It had **0 output repairs, 0 duplicate normalizations and
0 cache hits**. Usage was 132,796 prompt + completion tokens. Mean time per
case was 21.39 s; median was 18.73 s. Two-stage cases include both calls in their
time. These are planning times, not robot execution times.

| Evaluation category | Correct / total |
| --- | --- |
| Standard transfers | 2/2 |
| Paraphrases | 3/3 |
| Chinese instruction | 1/1 |
| Color selection | 1/1 |
| Distractors | 2/2 |
| Spatial relations | 3/3 |
| Missing/unlocalized targets | 5/5 |
| Ambiguity/unspecified destination | 3/3 |
| Unsupported requests | 6/6 |
| Execution-state replanning | 4/4 |
| Clarification followed by replanning | 1/1 |
| Search followed by replanning | 1/1 |

Other checks were kept separate:

| Check | Result | Meaning |
| --- | --- | --- |
| Live Qwen + existing planning smoke trials | 5/5 | Real B with ground-truth mock A and teleport mock C |
| Final-code recorded transfers | 3/3 | Demo RGB-D A + real Qwen B + demo motion C in MuJoCo |
| Full offline regression suite | 182 passed, 1 skipped in 31.22 s | Includes fixed-response B, compiler, scorer, recovery and simulation tests |
| Six MP4 files | All decode without ffmpeg errors | Checked by full decode, not just file existence |

The skipped test is `tests/test_gripper.py:122`: contact-only physical grasp
mode is experimental. The videos and physical integration tests use the demo's
weld attachment. The final and failed replan images were also inspected: the
fixed run leaves the stone within the red region; the earlier failed run leaves
it outside the accepted placement area. The report page showed all 32 cases
and six episode links.

## What the ground truth means

The canonical test inputs are
[`development.json`](../../eval/trials/student_b/development.json) and
[`evaluation.json`](../../eval/trials/student_b/evaluation.json).
They contain prepared `SceneDescription` inputs and predefined expected labels.
These are not measurements from a finished Student A model. Perceived IDs are
remapped to avoid relying on the offline demo's `p0/p1/p2` numbering.

For each request, `expected` gives the correct status and relevant object/region
IDs, or requires a role to remain unbound when ambiguous. Search labels specify
allowed class names. The evaluator also checks action dependencies: approach
before grasp, transport of the intended held object, placement in the intended
region, final verification, then stop. It accepts equivalent valid sequences;
it does not compare one exact JSON string.

Expected answers never enter the model input. A self-consistent plan for the
wrong object fails the external labels even if B's compiler accepts it. Offline
tests explicitly cover that case and check that a private expected marker never
reaches the request. Unsupported and clarification cases require a nonempty
reason/question; the evaluator does not grade the quality of that prose.

Each case counts once. A recovery case must pass both initial and final stages.
API failures and invalid outputs remain in the denominator. No failed case was
removed or rerun within the final evaluation. The run is marked `complete: true`.
Development and physical episodes are not added to its denominator.

The evaluation started at `2026-09-07T05:59:46.383595+00:00` after freezing B.
Its manifest records hashes and copies of the suite, prompt, compiler, adapter,
configuration, LLM client and scorer. Those source hashes still matched after
evaluation. The later C release fix and report/document edits did not change B
or the scoring code. Future tuning after inspecting this set makes it a
regression set; use newly labelled unseen cases for another independent claim.

## Configuration actually used

| Setting | Value |
| --- | --- |
| Model | `qwen3.7-plus-2026-05-26` |
| Base URL | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| Protocol | OpenAI-compatible Chat Completions |
| Output mode | `json_schema` |
| Prompt / compiler | `qwen-b-v3` / `student-b-compiler-v2` |
| Wire schema | `student-b-plan-v1` |
| Temperature / max output tokens | 0 / 2048 |
| Thinking setting | Unset; provider flag omitted |
| HTTP timeout / extra transport retries | 60 s / 1 |
| Maximum output repairs | 1 |
| Parallel cases | 2 |
| Cache | Disabled for the benchmark |

The development runs used a 30 s HTTP timeout; generation settings were the
same. Raw usage and timing are kept per call. Credentials are not part of the
report or source snapshots.

## Failures retained and changes made

1. **Original API failure:** Qwen selected the correct stone and red region, but
   repeated IDs in unused action fields. The old compiler rejected both its
   first answer and repair. The new compiler clears only allowed duplicates
   that exactly match the target/goal. It records the change, preserves raw JSON,
   and still rejects conflicts, unknown fields and invented coordinates. Both
   original responses now pass the offline replay through the compiler.
2. **Development `dev_07`:** two stones were visible and the request did not
   select one. Qwen guessed instead of asking. Prompt v3 now counts scene-derived
   candidates and forbids defaulting to the first or gray stone.
3. **Development `dev_11`:** a cube was held while the accepted goal remained
   the stone. Qwen called the task infeasible. Prompt v3 distinguishes a held
   object conflict, which needs clarification, from an unsupported task.
4. **Development `dev_05`:** missing-region handling required one repair in both
   rounds. In round 2 the first response used class name `red_region` as an ID;
   the compiler rejected it and the repair succeeded. This is why development
   first-response accuracy remains 12/13 even when accepted accuracy is 13/13.
5. **Recorded placement failure:** B correctly replanned after an injected grasp
   miss, but C detached the weld while the gripper was still closed. The restored
   contacts could eject the stone. C now stops motion, opens the gripper, waits
   at most two simulated seconds for public opening feedback, then detaches.
   Position logic and success tolerances were not loosened. The old failed video
   remains alongside the successful rerun.

The diagnostics folder also preserves the placement trace used to diagnose C.
That trace used an evaluator-only oracle observer and was never supplied to B
or C. Exploratory pose-adjustment changes were removed; the final C change is
only the gripper-opening order and bounded feedback wait.

## Recorded episodes

All six episodes use uncached live Qwen responses. Their `B.model` events retain
the input, raw output, validation and timing. `B.plan` shows the compiled plan.

| Folder under `episodes/` | Outcome | What it demonstrates |
| --- | --- | --- |
| `live_success` | Claimed and actual success | Standard stone transfer before the C fix |
| `live_replan` | Failed; actual false | Correct B recovery followed by C placement failure; retained |
| `live_success_fixed` | Claimed and actual success | Stone transfer with final C code |
| `live_replan_fixed` | Claimed and actual success | First `GRASP_MISSED`, second live B plan, successful placement |
| `cube_transfer` | Claimed and actual success | Blue cube selected and independently checked as the target |
| `missing_target` | `CLARIFICATION_EXHAUSTED`; actual false | Dark-red stone absent; B searches and then asks without moving the gray stone |

The grasp miss is deliberately injected before motion. It tests the error and
replanning path, not detection of a naturally occurring physical grasp failure.
The missing-target episode is an incomplete task, not a member of the 32-case
accuracy set. C currently searches by class only, so it reports the gray stone
when B searches for the absent dark-red stone. B preserves the color constraint;
after repeated searches it asks for clarification. The demo has no interactive
answer provider, so it ends without a false success claim.

Videos are annotated workflow replays with display pauses, about 12–25 s each.
They are not the assignment's uncut final assessment recording. A's color/geometry
baseline and C's simulated motion are teaching implementations; real Student A
and Student C remain stubs.

## Open or reproduce

The local evidence folder is:
`/home/jiamo/EE4705/project1.3/runs/qwen_b_refine_20260907`.
It is ignored by Git; the code, fixtures, labels and this report are versioned.

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m eval.b_report runs/qwen_b_refine_20260907
.venv/bin/python -m http.server 8767 --bind 127.0.0.1 \
  --directory runs/qwen_b_refine_20260907
```

Open [the local result and video page](http://127.0.0.1:8767/).
If port 8767 is already serving this report, use that existing page.

| Evidence | Relative path inside the evidence folder |
| --- | --- |
| Frozen final inputs, settings and source | `evaluation/suite.json`, `evaluation/manifest.json`, `evaluation/source/` |
| Final scores and all returned plans | `evaluation/summary.json` |
| Each independent expected answer | `evaluation/test_XX/case.json` |
| Each raw request/response | `evaluation/test_XX/audit/<episode-id>/001.json` |
| Original rejected output | `historical_failure/original_audit.json` |
| Development failures | `development_r1/dev_07/`, `development_r1/dev_11/` |
| Development repair evidence | `development_r2/dev_05/` |
| Repaired original live request | `repaired_live_request/` |
| Planning smoke results | `smoke/20260907_140655_planning/` |
| Videos, observations and event streams | `episodes/<name>/` |
| Physical diagnosis | `placement_diagnostic/trace.json` |
| Regression / video checks | `pytest.txt`, `video_validation.json` |

For a fresh measurement, set the API environment variables using
[the B guide](../STUDENT_B_README.md#5-connect-qwen), then use new output folders:

```bash
.venv/bin/python -m eval.b_benchmark \
  --suite eval/trials/student_b/evaluation.json \
  --out runs/my_b_evaluation --workers 2

.venv/bin/python -m demo.run --student B --scenario retry \
  --attempts-per-action 1 --out runs/my_b_replan

.venv/bin/python -m demo.run --student B \
  --instruction 'Move the blue cube to the red area.' \
  --expected-target cube --out runs/my_b_cube

.venv/bin/python -m eval.runner --mode planning \
  --trials eval/trials/smoke --out runs/my_b_smoke

.venv/bin/python -m pytest -q -rs
```

API commands make paid requests. Offline pytest uses fixed responses and does
not require a key. See the B guide for every input/output file and how to add
new labels without revealing expected answers to the model.
