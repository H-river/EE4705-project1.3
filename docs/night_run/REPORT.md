# Night run REPORT — branch `e2e` (2026-09-23, 00:26 → ~01:50)

Live API budget used: **535 / 600** (A/VLM 397, B/LLM 138; cache hits A 72, B 10), counted from the audit files (HTTP attempts, de-duplicated). Counter: `runs/night/CALLS.txt`. All raw outputs are under `runs/night/` (git-ignored); copies of the text files are in `docs/night_run/`.

## 1. Stages

| stage | result |
|---|---|
| 0 env | PASS (after credentials arrived). EGL render ok; offline demo ok; live B ok; live A ok (but that episode was claimed=False / actual=True) |
| 1 merge + tests | PASS: A (`Task_a_perception_4`) and C (`task_c_moveit`) merged cleanly; 221 passed, 1 skipped |
| 2 fixes | PASS: 2.1–2.7 committed; 225 passed, 1 skipped. Later 2.8 + round 1 → 233 passed, 1 skipped |
| 3 planning | PASS: 31/32 on `eval/trials/student_b/evaluation.json` (run via `eval.b_benchmark`, because `eval.runner` only reads YAML). Only failure test_26 = contract rule (see §5) |
| 4 manipulation | PASS (9/10 on student_c, 9/10 on student_c_v2), **but both failures are regressions from fixes 2.4 and 2.5** (baseline 10/10 on both; verified by reverting each one) |
| 5 A demo | done. No final_verification event: PLACE failed because the red region was UNLOCALIZED ("box touches image boundary") at both verification frames (7, 8), then GRASP gave TARGET_LOST. Evaluator: actual success |
| 6 e2e smoke | FAIL: smoke_1/3 ERROR (A raises on invalid VLM reply; 2 reruns each, same). smoke_4 took the NEEDS_SEARCH path ✔ but SEARCH_EXHAUSTED. smoke_5 clarification asked and answered ✔, then LIMIT_EXCEEDED in a SEARCH loop. smoke_2 REFUSED with actual success |
| 7 e2e main | e2e_v2: claimed 0/10, actual 3/10, 0 ERROR. e2e_eval NOT run (needs ≥150 calls left; 90 were left) |
| fix loop | 1 round (K1, C PLACE view retry): 3/4 affected trials → CLAIMED_SUCCESS. Stopped at budget < 80. K3 prepared as a patch, not committed |
| 8 report | this file |
| 9 B material | see §7 (appended) |

## 2. Fix list (paste-ready for teammates)

**For Student C (`executor/student_c.py`)**
- `8a930b3` [C-fix] **SEARCH no longer monkeypatches `perception.describe`.** `_remember_search_result`/`_install_search_memory` replaced A's describe() with a wrapper that re-inserted remembered instances into every later scene and stamped them with the new frame_id, so downstream checks saw objects the current frame did not contain. The call site was removed; the functions were kept. Test: tests/test_student_c.py (no dedicated test). ⚠ This removal exposed the object/region ping-pong (K5) the patch was hiding. That is really A's/B's problem to solve.
- `069559a` [C-fix] **A failed post-grasp retreat no longer turns a verified grasp into a failure.** The retreat result goes under `info["return_to_initial_pose"]`; success stays True. Test: tests/test_student_c.py (no dedicated test).
- `e8181b6` [C-fix] **`_arm_tucked` is reset after every REACH/GRASP/PLACE**, so the next APPROACH/SEARCH tucks again. ⚠ **REGRESSION**: tucking from the post-GRASP/post-PLACE pose often fails (IK "intersects robot or scene geometry"). It caused manip c_05 to fail (10/10 → 9/10) and the K2 failures in c_2_01/02/08 and smoke_2. Recommend: revert, or on IK failure skip the tuck when the hand is already clear of the table.
- `11204e1` [C-fix] **APPROACH/SEARCH contact recovery returns TIMEOUT (contact_recovery=True) instead of success.** ⚠ **REGRESSION**: in c_2_05 (manip, 10/10 → 9/10 on v2) the "successful" backed-off pose used to be good enough to grasp from; now B replans the same APPROACH into the same contact until the limit. The report is now honest, but a real recovery (a different approach heading) is missing.
- `2fe1fc8` [C-fix] round 1: **PLACE retries verification from the last localized viewpoint** when the result is "reliable 3D grounding is required", not only "not visible". Order: yaw offset 0 at `_view_pose`, then ±0.2 rad. Result: c_2_01, c_2_02, smoke_2 went from REFUSED (placed correctly but unverified) to CLAIMED_SUCCESS. Test: tests/test_student_c_view_recovery.py (3 tests).

**For Student A (`perception/vision_contract.py`)**
- `61327b7` [A-fix] **A >600×600 (near full-frame) detection is dropped instead of rejecting the whole frame**; `selected` is re-mapped. Test: tests/test_student_a_bbox.py.
- `3258089` [A-fix] 2.8: **An unselected detection with a degenerate bbox is dropped** (x1≥x2, y1≥y2, or a side < 2 in 0..1000 units; e.g. qwen's `[0,0,0,0]` "not visible" placeholder with confidence 0.0). A degenerate box that is itself `selected` still raises, so the one repair is used; this keeps the existing test_bad_wire_rejected case valid. Test: tests/test_student_a_bbox.py (live-reply fixture + 4 cases). The smoke_1/3 rerun after 2.8 is still pending (budget).
- Also for A: commit `038ec54` on your branch broke the key-entry line in docs/STUDENT_A_README.md (`read -rsp '…' https://ws-…` is not valid bash, and it exposes a workspace endpoint). Please restore `read -rsp 'Qwen API key: ' DASHSCOPE_API_KEY`.

**For Student B (me)**
- `dcb962a` [B] planning_input sends only `attributes.color` to the model; A's new memory_* tracker fields no longer leak into the prompt. Tests: test_student_b, test_b_benchmark.

**Chore**: `5686bd7`: .vscode untracked; eval/trials/my_student_c deleted (byte-identical to student_c); my_student_c_copy → **eval/trials/student_c_v2**.

## 3. E2E summary

Wall times of reruns (fix_r1) are shorter partly because A's replies for unchanged frames came from the cache.

### e2e_v2 (10 trial runs)

| | actual=True | actual=False |
|---|---|---|
| claimed=True | 0 | 0 |
| claimed=False | 3 | 7 |

- false-claim rate (#claimed∧¬actual / #claimed): n/a (0 claimed)
- missed-success rate (#¬claimed∧actual / #actual): 3/3 = 1.00
- outcomes: {'REFUSED': 3, 'LIMIT_EXCEEDED': 4, 'SEARCH_EXHAUSTED': 3}
- last-failed-action error_code histogram: {'UNREACHABLE': 3, 'SEARCH_NOT_FOUND': 4, 'TIMEOUT': 1, '-': 2}
- wall time of actual successes: mean 139.4 s, median 121.4 s (n=3)
- live calls in this set: 318
- attribution: {'A': 3, 'unknown': 6, 'C': 1}

### fix_r1 (4 trial runs)

| | actual=True | actual=False |
|---|---|---|
| claimed=True | 3 | 0 |
| claimed=False | 1 | 0 |

- false-claim rate (#claimed∧¬actual / #claimed): 0/3 = 0.00
- missed-success rate (#¬claimed∧actual / #actual): 1/4 = 0.25
- outcomes: {'CLAIMED_SUCCESS': 3, 'REFUSED': 1}
- last-failed-action error_code histogram: {'-': 3, 'UNREACHABLE': 1}
- wall time of actual successes: mean 55.8 s, median 22.8 s (n=4)
- live calls in this set: 25
- attribution: {'-': 3, 'A': 1}

### e2e_smoke (9 trial runs)

| | actual=True | actual=False |
|---|---|---|
| claimed=True | 0 | 0 |
| claimed=False | 1 | 8 |

- false-claim rate (#claimed∧¬actual / #claimed): n/a (0 claimed)
- missed-success rate (#¬claimed∧actual / #actual): 1/1 = 1.00
- outcomes: {'ERROR': 6, 'REFUSED': 1, 'SEARCH_EXHAUSTED': 1, 'LIMIT_EXCEEDED': 1}
- last-failed-action error_code histogram: {'-': 7, 'UNREACHABLE': 1, 'SEARCH_NOT_FOUND': 1}
- wall time of actual successes: mean 98.4 s, median 98.4 s (n=1)
- live calls in this set: 143
- attribution: {'A': 7, 'C': 1, 'unknown': 1}
**Latest run of each of the 15 distinct e2e trials** (smoke 5 + v2 10, after round 1): claimed∧actual 3, ¬claimed∧actual 1, ¬claimed∧¬actual 11, **false claims 0**. The system never claimed success when the task wasn't done. Every miss was either a missed success (verification could not localize the region) or a failure to act.

Live calls: 535 in total (env 10, planning 39, A demo 0, smoke 143, v2 318, round 1 25). Cache hits: A 72, B 10.

## 4. Attribution table

See `E2E_TABLE.md` (copied in full below). Rule-based attribution follows the plan's rules; the "triage cause" column refers to the cause table in PROGRESS.md (K1–K7).

| set | trial | claimed | actual | outcome | error_code | replans | llm_calls | wall_s | attribution (rule) | triage cause | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2e_smoke | smoke_1_standard | False | False | ERROR |  | 1 | 20 | 70.3 | A | K7 VLM invalid reply → A raises (A) | error: core.llm_client.SchemaError: VLM output invalid after one repair: $.detections[2].color: 'empty' not in enum ['gray', 'dark_red', 'blue', 'green', 'red', ''] |
| e2e_smoke | smoke_2_scene_variation | False | True | REFUSED | UNREACHABLE | 3 | 11 | 98.4 | A | K1 + K2 (C); fixed by round 1 | task achieved but not claimed (verification/perception) |
| e2e_smoke | smoke_3_instruction_variation | False | False | ERROR |  | 1 | 3 | 44.1 | A | K7 (A) | error: core.llm_client.SchemaError: VLM output invalid after one repair: $.detections[2].bbox: violates minItems=4 |
| e2e_smoke | smoke_4_search | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 1 | 44 | 115.3 | C | K6 contact→TIMEOUT then SEARCH_NOT_FOUND (C) | failed codes TIMEOUT,SEARCH_NOT_FOUND,SEARCH_NOT_FOUND +contact |
| e2e_smoke | smoke_5_clarification | False | False | LIMIT_EXCEEDED |  | 9 | 15 | 243.6 | unknown | K5-like: ground() found stone, describe() did not (A); clarification itself OK | last events: [{"type": "search_found", "sim_time": 3.689999999999815, "target": "stone"}, {"type": "limit", "sim_time": 3.689999999999815, "which": "max_total_plans"}, {"type": "safe_stop", "sim_time"… |
| e2e_smoke | smoke_1_standard | False | False | ERROR |  | 1 | 1 | 5.7 | A | K7 VLM invalid reply → A raises (A) | error: core.llm_client.SchemaError: VLM output invalid after one repair: $.detections[2].bbox: violates minItems=4 |
| e2e_smoke | smoke_3_instruction_variation | False | False | ERROR |  | 1 | 1 | 5.2 | A | K7 (A) | error: core.llm_client.SchemaError: VLM output invalid after one repair: $.detections[2].bbox: violates minItems=4 |
| e2e_smoke | smoke_1_standard | False | False | ERROR |  | 5 | 41 | 154.9 | A | K7 VLM invalid reply → A raises (A) | error: core.llm_client.SchemaError: VLM output invalid after one repair: bbox must have positive width and height |
| e2e_smoke | smoke_3_instruction_variation | False | False | ERROR |  | 5 | 7 | 95.5 | A | K7 (A) | error: core.llm_client.SchemaError: VLM output invalid after one repair: bbox must have positive width and height |
| e2e_v2 | c_2_01_stone | False | True | REFUSED | UNREACHABLE | 3 | 11 | 117.1 | A | K1 PLACE unverifiable + K2 tuck blocked (C); fixed by round 1 | task achieved but not claimed (verification/perception) |
| e2e_v2 | c_2_02_stone | False | True | REFUSED | UNREACHABLE | 3 | 13 | 121.4 | A | K1 + K2 (C); fixed by round 1 | task achieved but not claimed (verification/perception) |
| e2e_v2 | c_2_03_stone | False | False | LIMIT_EXCEEDED | SEARCH_NOT_FOUND | 9 | 106 | 416.3 | unknown | K5 object/region ping-pong (A/B) | last events: [{"type": "search_found", "sim_time": 41.59000000000486, "target": "stone"}, {"type": "limit", "sim_time": 41.59000000000486, "which": "max_total_plans"}, {"type": "safe_stop", "sim_time"… |
| e2e_v2 | c_2_04_stone | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 3 | 39 | 180.1 | unknown | K5 ping-pong + K4 edge box (A/B/C) | last events: [{"type": "action", "sim_time": 16.496000000001736, "skill": "SEARCH", "target": "stone", "success": false, "error": "SEARCH_NOT_FOUND", "attempt": 1, "frame_id": 37}, {"type": "search_ex… |
| e2e_v2 | c_2_05_stone | False | False | LIMIT_EXCEEDED | TIMEOUT | 9 | 39 | 339.5 | C | K6 APPROACH torso-table contact loop (C, fix 2.5) | failed codes TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT +contact |
| e2e_v2 | c_2_06_cube | False | False | LIMIT_EXCEEDED |  | 9 | 17 | 170.0 | unknown | K3 "red cube" grounding accepted by SEARCH (A) | last events: [{"type": "search_found", "sim_time": 4.155999999999764, "target": "cube"}, {"type": "limit", "sim_time": 4.155999999999764, "which": "max_total_plans"}, {"type": "safe_stop", "sim_time":… |
| e2e_v2 | c_2_07_cube | False | False | LIMIT_EXCEEDED |  | 9 | 16 | 263.9 | unknown | K3 "red cube" grounding accepted by SEARCH (A) | last events: [{"type": "search_found", "sim_time": 3.689999999999815, "target": "cube"}, {"type": "limit", "sim_time": 3.689999999999815, "which": "max_total_plans"}, {"type": "safe_stop", "sim_time":… |
| e2e_v2 | c_2_08_cube | False | True | REFUSED | UNREACHABLE | 5 | 14 | 179.8 | A | K1 + K2 (C); round 1 not enough (region never localized) → K2 | task achieved but not claimed (verification/perception) |
| e2e_v2 | c_2_09_bottle | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 2 | 33 | 111.3 | unknown | K4 bottle only seen as edge box, SEARCH never centres it (C/A) | last events: [{"type": "action", "sim_time": 16.026000000001996, "skill": "SEARCH", "target": "bottle", "success": false, "error": "SEARCH_NOT_FOUND", "attempt": 1, "frame_id": 33}, {"type": "search_e… |
| e2e_v2 | c_2_10_bottle | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 1 | 30 | 100.4 | unknown | K4 bottle only seen as edge box (C/A) | last events: [{"type": "action", "sim_time": 15.556000000001864, "skill": "SEARCH", "target": "bottle", "success": false, "error": "SEARCH_NOT_FOUND", "attempt": 1, "frame_id": 29}, {"type": "search_e… |
| fix_r1 | c_2_01_stone | True | True | CLAIMED_SUCCESS |  | 0 | 6 | 25.3 | - | - |  |
| fix_r1 | c_2_02_stone | True | True | CLAIMED_SUCCESS |  | 1 | 6 | 20.3 | - | - |  |
| fix_r1 | c_2_08_cube | False | True | REFUSED | UNREACHABLE | 5 | 7 | 157.9 | A | K1 + K2 (C); round 1 not enough (region never localized) → K2 | task achieved but not claimed (verification/perception) |
| fix_r1 | smoke_2_scene_variation | True | True | CLAIMED_SUCCESS |  | 0 | 6 | 19.7 | - | - |  |

## 5. Needs a human decision (ranked)

1. **Fixes 2.4 and 2.5 are regressions.** Manip went from 10/10 to 9/10 on both sets. Verified: reverting 2.4 fixes c_05, reverting 2.5 fixes c_2_05, reverting both gives 10/10 + 10/10. 2.4 is also K2 in e2e (tuck IK blocked after PLACE/GRASP). The plan only asked for a revert at ≤ 8/10, so both are still in. Recommendation: revert 2.4 (`git revert e8181b6`), or make the tuck fall back to "skip if hand already above table + margin". For 2.5, keep the honest TIMEOUT but give C a real alternative approach heading, or revert.
2. **Object/region ping-pong (K5) and ground()/describe() disagreement**: c_2_03, c_2_04, smoke_5 and the SEARCH loops in smoke_1/3. From many viewpoints only one of {target, red region} is LOCALIZED. C's search-memory patch hid this by faking fresh detections (removed in 2.2). A real fix has to surface A's tracker memory to B without calling it LOCALIZED (core/types.py says LOCALIZED needs 2D evidence), which is an interface decision (R3), for daytime.
3. **Apply K3** (`runs/night/pending_K3_class_color.patch`, tests included, suite green): drop impossible class/colour detections (a "red cube" is the red square) and normalise stone/red → dark_red, using the public palette in assets/objects.yaml. It targets c_2_06/c_2_07, whose SEARCH "found" a red cube 9× in a row. Untested live; rerun c_2_06/c_2_07 after applying.
4. **Edge-box targets (K4)**: A marks any box touching the image border UNLOCALIZED, and C's SEARCH sweep never re-centres it (bottle in c_2_09/10: 42 empty + 10 edge-box groundings). This is the main remaining e2e blocker. Options: C turns toward the side the box touches; or A localizes from the visible part for small objects.
5. **Contract rejects a correct INFEASIBLE** (planning test_26 "inside the drawer"): the repaired reply was INFEASIBLE with goal.region_name='drawer' and was rejected with "Unknown region class". planner/contract.py is R3-frozen; decide whether INFEASIBLE goals may name unsupported classes.
6. **VLM schema-invalid replies (K7)** raise inside core/llm_client before A's validate_wire (color 'empty', bbox with <4 items), and the orchestrator turns that into ERROR. It needs either A-side normalisation before schema validation or a backbone change. Also: the first reply is cached even when it later fails validate_wire, so reruns replay it. Consider not caching replies that fail A's own validation.
7. **Deviation in 2.8**: a selected degenerate box still raises (see §2). Confirm or relax (relaxing requires changing test_bad_wire_rejected[mutation0]).
8. Rerun smoke_1/3 after 2.8 (skipped, 65 calls left < 100), and run `student_c_evaluation` e2e (never run).
9. Not done (per plan): moving A's memory attributes to a new field; C's `env._world` access in `_table_contacts` (needs a public RobotEnv API).

## 6. `git log --oneline main..e2e`

```
2fe1fc8 [C-fix] round 1: retry PLACE verification from the last localized view when unlocalized
3258089 [A-fix] 2.8: drop degenerate unselected bboxes instead of rejecting the frame
5686bd7 [chore] Untrack .vscode, drop duplicate trial set, rename C's new trials to student_c_v2
61327b7 [A-fix] Drop near-full-frame detections instead of rejecting the whole VLM frame
11204e1 [C-fix] Report APPROACH/SEARCH table-contact recovery as TIMEOUT, not success
e8181b6 [C-fix] Re-tuck the arm after every REACH/GRASP/PLACE attempt
069559a [C-fix] Keep a verified grasp successful when the post-grasp retreat fails
8a930b3 [C-fix] Stop SEARCH from monkeypatching perception.describe
dcb962a [B] Send only whitelisted perception attributes (color) to the planner model
8060727 Merge remote-tracking branch 'origin/task_c_moveit' into e2e
038ec54 Change Qwen API key input method to URL
d96b423 Revise capture_dataset for balanced case generation
3f4905f Add memory attributes to GroundedObject instantiation
18ed9ee Student C: added "collision detection and small retreat" function to prevent the robot from tilting due to hitting the table as it moves
c6afce6 Add VQA object-mention accuracy check script
790ae0a Enhance tracking with memory attributes and recall method
8f2e006 Student C: added "retreat" function after grasping target object
f1afc06 Student C: Arm tuck function starts before scanning to prevent outstretched arm moving objects at the start. SEARCH updated: Keeps record of previous scan to prevent function looping (exit SEARCH when relevant items are found)
15ae22c Student C: added "arm tuck" in APPROACH function trials "c_2_08_cube" and "c_2_10_bottle" adjusted to have all objects on table
b543a6c Created new set of 10 trials under "eval/trials/my_student_c_copy" with red stone included and various object placements
12e8ea0 Add ffmpeg installation step to README
95d51ef Enhance grounding query rules in vision contract
093d6d9 Update README with virtual environment activation
c616bfa Student C: STOP function added (Version 1.0) Shifted parameters in separate functions to top of _execute
915b6be Student C: VERIFY function added (Version 1.0)
6e6bbd4 Student C: SEARCH function added (Version 1.0)
52d8bbb Student C: PLACE function's test AssertionError resolved (Version 1.0.1)
0ae91eb Student C: PLACE function added (Version 1.0) PLACE: resolved AssertionError during tests/test_student_c.py GRASP: added "self._held_offset" for PLACE
1fdd346 Student C: MOVE_TO function added (Version 1.0)
ad404ec Student C: GRASP function added (Version 1.0)
4e614cf Student C: REACH functionadded (Version 1.0)
445ca38 Student C: APPROACH code added (Version 1)
```

## 7. STAGE 9 — B report material (appended)

- 9.1 `82557a9` [B] `eval/b_metrics.py` → `docs/validation/B_METRICS.md`. Over all 168 B planner calls tonight: first-pass-valid 163/168 (97.0%), repair 5/168, repaired and accepted 3/5, accepted 166/168. **Validator ablation: 5/168 first responses were rejected by the contract.** Without the validator the robot would have run them unchanged: 2 goal-colour mismatches, 1 target without a current 3D position, and 2 that named an unsupported class (these 2 are the contract's own false rejections, see §5 item 5). Normalisations: 0. Tokens/latency per call type are in B_METRICS.md (initial plan ≈ 2.5k prompt / 1.1k completion tokens, ~17 s; post-clarification ≈ 2.8k / 1.5k, ~22 s).
- 9.2 `eval/trials/student_b_variation/variation.json` (20 new cases): **19/20** live (runs/night/planning_variation). ambiguous 4/4, colour negation 4/4, paraphrase 4/4, relation 4/4, infeasible 3/4. The miss (var_15 "Move the purple sphere to the red area.") is the same contract rejection as test_26: the repaired answer was a correct INFEASIBLE but was rejected with "Unknown object class 'sphere'". This is now 2 independent cases for §5 item 5.
- Live calls for Stage 9: 21. **Final total: 556 / 600** (A 397, B 159).
