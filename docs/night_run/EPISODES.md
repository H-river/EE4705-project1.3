# Episode gallery (night run + rounds 2–5)

These are 10 e2e episodes: 5 where the robot claimed success and really achieved it, and 5 failures with different causes.

Since round 5, 9 of the 10 are **real episode videos**. Each was re-run live with the recorder attached:

```
python -m eval.runner --mode e2e --trials <dir with one trial YAML> --out <out> --video
```

`--video` attaches `demo/recording.py`'s Recorder, the same one `demo/run.py` uses, and the layout is unchanged. Physics is identical to a run without it: a mock-all smoke run gives bit-identical final positions, sim end time and event list with and without `--video`. The runner writes `<trial_id>/episode.mp4`, `episode.json` (recorder events, frames, holds) and `index.html` (replay page). The mp4 panels are:
- **Observer camera.** A live render every 0.1 s of sim time. It is not an A input.
- **A / last analyzed head-camera image.** A's boxes and instance ids for the last frame A described or grounded. The image is held until A runs again.
- **B / structured plan.** The current action list, with the executing action highlighted.
- **Stage banner.** The active module and skill, the result code and detail, sim time, and whether the object is attached.

The video clock includes labelled display holds (plan, action result, evaluator), so video time ≠ sim time. `episode.json` records both clocks. These are the round-5 live runs, **not** replays of the earlier runs. Round-3 prompt changes invalidated most of A's cache, so the VLM answered again and some episodes took a different path; the entries below describe the round-5 run. All mp4s are 1440×900, 24–58 s and < 2 MB, so no downscaling and no LFS were needed.

One episode, F5, is still the old **slideshow** from `scripts/render_episode.py` (saved frames only, 0 live calls). See F5 for why.

"run dir" is the folder under `runs/night/` (git-ignored) that holds the trial record, frames, episode.json and runner log.

## Successes (claimed ∧ achieved)

### S1. smoke_3_instruction_variation: paraphrased instruction
- **instruction:** "could you put that gray rock over on the red marker please?"
- **outcome:** CLAIMED_SUCCESS (claimed T / actual T), error_code –, 0 replans. Evaluator: 2.8 cm from the region centre
- **timeline:**
  1. t=1.0 B maps "gray rock" → stone a2 and "red marker" → red_region a1: READY APPROACH › GRASP › MOVE_TO › PLACE › VERIFY › STOP
  2. t=1.0–7.5 APPROACH a2 ok, GRASP a2 ok, MOVE_TO a1 ok
  3. t=11.3 PLACE ok after 2 view retries: "offset [-0.009, -0.031, 0.022], released and stable across 0.2 s"
  4. t=11.5 VERIFY passed; t=11.7 STOP
- **attribution:** – (all three modules worked). Same path as the round-2 run.
- **video:** `docs/night_run/episodes/ok_smoke_3_instruction_variation.mp4` (24 s; run dir `r5/smoke_3_instruction_variation`)

### S2. smoke_2_scene_variation: different layout
- **instruction:** "move the stone onto the red region"
- **outcome:** CLAIMED_SUCCESS (T/T), error_code –, 0 replans. Evaluator: 1.2 cm
- **timeline:**
  1. t=1.0 all targets LOCALIZED in the first frame → READY
  2. t=1.0–8.2 APPROACH, GRASP, MOVE_TO ok
  3. t=11.45 PLACE: the check from the retreat pose is a view problem. One view retry, back at the last localized viewpoint (round-1 fix #7), passes: offset 0.5 cm
  4. t=11.65 VERIFY passed
- **attribution:** –. Before round 1 this trial was REFUSED although the stone was placed correctly (fixes #7 and #9 in CHANGES.md).
- **video:** `docs/night_run/episodes/ok_smoke_2_scene_variation.mp4` (24 s; run dir `r5/smoke_2_scene_variation`)

### S3. smoke_5_clarification: ambiguous referent → clarification
- **instruction:** "move the stone onto the red region"
- **outcome:** CLAIMED_SUCCESS (T/T), error_code –, 1 replan (after the answer). Evaluator: 2.8 cm
- **timeline:**
  1. t=1.0 frame 0: A sees a gray stone and a dark_red stone (both LOCALIZED). B returns NEEDS_CLARIFICATION
  2. B asks "There are two stones: a gray one and a dark red one. Which stone should I move to the red region?" and gets the answer "the gray one"
  3. B replans READY on a3 (gray); APPROACH/GRASP/MOVE_TO ok; PLACE verified after 2 view retries (t=11.3)
  4. t=11.5 VERIFY passed (offset ≈3 cm, inside the 8 cm half extent)
- **attribution:** –. In the night run this trial ended LIMIT_EXCEEDED (ground()/describe() disagreement). Fixes #8 and #10 removed that.
- **video:** `docs/night_run/episodes/ok_smoke_5_clarification.mp4` (26 s; run dir `r5/smoke_5_clarification`)

### S4. c_2_08_cube: SEARCH first, then manipulate
- **instruction:** "Move the blue cube to the red area."
- **outcome:** CLAIMED_SUCCESS (T/T), error_code –, 1 replan. Evaluator: 0.3 cm
- **timeline:**
  1. t=1.0 the cube is not localized → NEEDS_SEARCH [SEARCH cube]
  2. t=3.7 SEARCH cube ok (cube a1 LOCALIZED) → READY
  3. t=3.7–11.8 APPROACH a1, GRASP a1, MOVE_TO a2 ok
  4. t=15.7 PLACE ok after 3 view retries (offset 1.7 mm); t=15.9 VERIFY passed
- **attribution:** –. This trial was REFUSED in the night run and after round 1 (the region stayed UNLOCALIZED at verification). The experimental partial-region fix (#9) made it pass.
- **video:** `docs/night_run/episodes/ok_c_2_08_cube.mp4` (31 s; run dir `r5/c_2_08_cube`)

### S5. c_2_06_cube: recovers from a provider content-filter refusal
- **instruction:** "Move the blue cube to the red area."
- **outcome:** CLAIMED_SUCCESS (T/T), error_code –, 0 replans. Evaluator: 0.7 cm
- **timeline:**
  1. t=1.0 READY; APPROACH a0, GRASP a0, MOVE_TO a1 ok
  2. PLACE releases the cube; the first check is a view problem, so C does a view retry
  3. t=11.47 frame 10 (view retry 1): **the provider refuses the image again** (A's audit: `status=content_filtered`). A returns an empty frame, and verification returns `passed=None`, which counts as a view problem
  4. view retry 2: cube and region LOCALIZED; t=12.07 PLACE verified (offset 0.9 cm); t=12.27 VERIFY passed
- **attribution:** –. This is the after-run for fix #14, and the round-5 run reproduced the refusal at the same frame. The same trial ended ERROR in round 2 (F5).
- **video:** `docs/night_run/episodes/ok_c_2_06_cube_content_filter_recovered.mp4` (25 s; run dir `r5/c_2_06_cube`)

## Failures

### F1. c_2_04_stone (round 5): dark-red stone placed on the red region, then UNLOCALIZED → LIMIT_EXCEEDED
- **cause class:** perception UNLOCALIZED (task achieved, not claimed)
- **replaces:** the old F1, c_2_01_stone (night run, region UNLOCALIZED → REFUSED). c_2_01 has been CLAIMED_SUCCESS since round 1, and it was again in round 5 (`r5/c_2_01_stone`, T/T, 0.7 cm). c_2_04 is the E2E_TABLE failure with the same cause (r2_v2: LIMIT_EXCEEDED / PLACE_FAILED, actual T, "task achieved but not claimed (verification/perception)"), and round 5 reproduced it.
- **instruction:** "Move the red stone to the red area."
- **outcome:** LIMIT_EXCEEDED (claimed F / **actual T**, 1.1 cm from the centre), last failed error_code PLACE_FAILED, 9 replans
- **timeline:**
  1. t=1.0 frame 0: dark_red stone a3 and red_region a2 LOCALIZED → READY
  2. t=1.0–7.8 APPROACH a3, GRASP a3, MOVE_TO a2 ok
  3. t=11.87 PLACE releases the stone in the region. Frame 11 shows it: A's caption reads "a dark_red stone on a red region", but the stone is **UNLOCALIZED**. The 3 view retries move the region to the image edge, and A stops reporting the stone at all → PLACE_FAILED "exact object or region instance is not visible"
  4. B: NEEDS_SEARCH (red stone). SEARCH stone "succeeds" 9× at the same sim time, because ground("stone") returns the **gray** stone a4, while B's next describe() has no dark_red stone → max_total_plans → LIMIT_EXCEEDED
- **attribution:**
  - A: a red object on the red region is not localized (low contrast), and it is dropped entirely from the retry views.
  - C/A/B: a SEARCH target is only a class, so SEARCH accepts any stone and never learns that the goal is the dark_red one. The pending K3 patch does not cover this; it only drops impossible class/colour pairs.
  - B: 9 replans all asked for the same SEARCH.
  - Still open.
- **video:** `docs/night_run/episodes/fail_c_2_04_placed_stone_unlocalized.mp4` (45 s; run dir `r5/c_2_04_stone`)

### F2. c_2_03_stone (round 5): SEARCH ping-pong, lost grasp target, then a refusal based on a caption
- **cause class:** search loop → unfounded refusal
- **instruction:** "Move the gray stone to the red area."
- **outcome:** REFUSED (F/F), last failed error_code TARGET_LOST, 9 replans. **Changed** from the gallery's round-2 run (LIMIT_EXCEEDED, no failed code); round 2's r2_fixB rerun also ended REFUSED/TARGET_LOST.
- **timeline:**
  1. t=1.0 frame 0: both stones are cut by the right image edge → UNLOCALIZED → NEEDS_SEARCH stone
  2. t=9.4 SEARCH stone NOT_FOUND; then the ping-pong: SEARCH red_region ok → "stone not visible" → SEARCH stone ok → "red region not visible" … (3 rounds, t=9.9–22.6)
  3. t=22.6 READY on a4; APPROACH ok; t=23.1 GRASP → TARGET_LOST "needs fresh, unambiguous 3D evidence" (frame 62: a4 box at the right edge, UNLOCALIZED)
  4. SEARCH stone ok, SEARCH red_region ok. At t=30.0 (frame 82) the stone fills the lower image, and A's caption says "**The gray stone is part of the table surface** (not a separate object)". B returns INFEASIBLE with exactly that reason → REFUSED
- **attribution:**
  - A/C: SEARCH accepts a target that the next describe() does not report as LOCALIZED (K5 in REPORT §5).
  - C: APPROACH leaves the stone at the image edge.
  - A: the caption contradicts its own detection list.
  - **B (mine):** INFEASIBLE from a caption claim, although a4 is in the object list. B should not refuse on free text.
  - Still open.
- **video:** `docs/night_run/episodes/fail_c_2_03_search_loop_refusal.mp4` (58 s; run dir `r5/c_2_03_stone`)

### F3. c_2_05_stone (round 5): torso–table contact on APPROACH
- **cause class:** scene obstacle
- **instruction:** "Move the red stone to the red area."
- **outcome:** LIMIT_EXCEEDED (F/F), last failed error_code TIMEOUT, 9 replans. The label matches the gallery, but the path changed.
- **timeline:**
  1. t=1.0 NEEDS_SEARCH; t=4.2 SEARCH stone ok → READY on the dark_red stone a3
  2. t=5.27 APPROACH a3 reports ok with "Table contact detected; backed off before the next planned action". GRASP → TARGET_LOST (frame 10: a3 at the right edge, UNLOCALIZED)
  3. t=9.3 SEARCH stone → TIMEOUT "Repeated table contact during SEARCH"; retry ok at t=11.0
  4. From frame 28 the view shows only the **gray** stone a4. SEARCH stone "succeeds" 8× at an unchanged sim time (it grounds a4), while B keeps asking for the red stone → LIMIT_EXCEEDED
- **attribution:**
  - C: the approach heading drives the torso into the table edge, and there is no alternative heading.
  - C/A: the class-only SEARCH acceptance, as in F1.
  - Still open.
- **video:** `docs/night_run/episodes/fail_c_2_05_table_contact.mp4` (41 s; run dir `r5/c_2_05_stone`)

### F4. c_2_09_bottle (round 5): GRASP loses the target after APPROACH
- **cause class:** grasp
- **instruction:** "Move the green bottle to the red area."
- **outcome:** **REFUSED** (F/F), last failed error_code SEARCH_NOT_FOUND, 2 replans. **Changed** from SEARCH_EXHAUSTED in round 2.
- **timeline:**
  1. t=1.0 READY (bottle a0 LOCALIZED in frame 0)
  2. t=3.9 APPROACH a0 ok, but at the approach pose the bottle box touches the right image edge (frame 6: [526, 232, 640, 375]) → UNLOCALIZED → GRASP → TARGET_LOST "needs fresh, unambiguous 3D evidence"
  3. SEARCH bottle: t=8.2 table contact (TIMEOUT), t=14.6 SEARCH_NOT_FOUND. The last view (frame 33) shows only the floor and the robot's shadow
  4. B: INFEASIBLE "could not be located after multiple search attempts" → REFUSED. Round 2 kept searching and ended SEARCH_EXHAUSTED
- **attribution:**
  - C: the approach pose puts a tall object at the image edge. GRASP needs fresh 3D evidence and has no re-centre step, and the SEARCH sweep looks at the floor.
  - A: a box clipped by the edge is UNLOCALIZED for objects (the partial fix #9 covers regions only).
  - Still open (REPORT §8 item 2).
- **video:** `docs/night_run/episodes/fail_c_2_09_grasp_target_lost.mp4` (28 s; run dir `r5/c_2_09_bottle`)

### F5. c_2_06_cube (round 2): provider content filter → ERROR — **slideshow**
- **cause class:** content filter
- **instruction:** "Move the blue cube to the red area."
- **outcome:** ERROR (claimed F / **actual T**), error_code INTERNAL_ERROR, 0 replans
- **timeline:**
  1. t=1.0 READY; APPROACH a0, GRASP a0, MOVE_TO a1 ok (frames 0–7)
  2. t=11.5 PLACE releases the cube inside the region. The verification frame (frame 11, red banner) is refused by the provider with HTTP 400 "Input data may contain inappropriate content"
  3. APIError propagates out of PLACE → INTERNAL_ERROR → the episode ends in ERROR. The evaluator finds the cube in the region
- **attribution:** provider (A's VLM call). Fixed in round 3 by #14.
- **why a slideshow:** the round-5 rerun of this trial is S5. The filter fired again, but fix #14 turned it into a view retry and the episode succeeded. By design this failure cannot be reproduced any more. E2E_TABLE has no other content-filter failure, so no same-cause replacement exists. The entry keeps its round-2 slideshow so the gallery still covers this cause.
- **video:** `docs/night_run/episodes/fail_c_2_06_content_filter.mp4` (**slideshow**, rendered by `scripts/render_episode.py`; run dir `r2_v2/c_2_06_cube`)

## Round 5 notes

- Live calls: 137 of 150 (A 111, B 26), for 10 runs: the 9 listed trials plus c_2_01 (confirmed that it now succeeds). The count comes from `runs/night/r5/audit_{a,b}` (HTTP attempts).
- A recurring pattern in F1 and F3: SEARCH("stone") grounds a stone of the **wrong colour** and reports success, and B keeps asking for the right one until max_total_plans.

## Final run (2026-09-24): re-recorded from tag `final-40`

These are 10 live re-runs of RUN 1 trials on the tagged code, with `--video`, cache off, B thinking off (`runs/final/video/`, 6/10 correct). The re-runs are new episodes, so their paths can differ from RUN 1. f23 failed in RUN 1 but succeeded here, which shows its carry contact is intermittent. "err" is the evaluator's xy distance from the region centre.

| file | trial | outcome | timeline |
|---|---|---|---|
| `final_ok_f04_smoke_4_search.mp4` | f04 search (robot starts facing away) | CLAIMED_SUCCESS, err 2.7 cm | NEEDS_SEARCH → SEARCH stone → READY APPROACH › GRASP › MOVE_TO › PLACE › VERIFY › STOP → final verification ✔ |
| `final_ok_f11_c_06_cube.mp4` | f11 blue cube | CLAIMED_SUCCESS, err 1.1 cm | READY, 6 actions, verification ✔ |
| `final_ok_f15_c_10_bottle.mp4` | f15 green bottle | CLAIMED_SUCCESS, err 0.2 cm | READY, 6 actions, verification ✔ (a bottle that stays in view after APPROACH) |
| `final_ok_f23_c_2_08_cube.mp4` | f23 cube with 2 stones present | CLAIMED_SUCCESS, err 4.8 cm | READY, 6 actions; no table contact this time (RUN 1: torso contact on both MOVE_TOs) |
| `final_ok_f49_reject_stack.mp4` | f49 "stack the cube on the bottle" | REFUSED (correct) | B: INFEASIBLE, stacking on objects is unsupported; no motion |
| `final_ok_f50_clarify_two_stones.mp4` | f50 two stones, no colour | CLAIMED_SUCCESS, err 2.7 cm | NEEDS_CLARIFICATION "Which stone …: the gray one or the dark red one?" → "the gray one" → READY, 6 actions, verification ✔ |
| `final_fail_f14_c_09_bottle.mp4` | f14 bottle | FAILED (claimed F, **actual T**, err 1.2 cm) | READY … PLACE ✔ but VERIFY ✘ ×2 (A never sees the exact bottle/region instances after the release) → re-grasp loop: SEARCH › APPROACH › GRASP TARGET_LOST ×4 |
| `final_fail_f19_c_2_04_stone.mp4` | f19 "red stone" | CLARIFICATION_EXHAUSTED | B asks "gray (a4) or dark red (a3)?" because 'red' ≠ A's colour `dark_red`; the trial has no scripted answer (B colour-vocabulary item) |
| `final_fail_f25_c_2_10_bottle.mp4` | f25 bottle at (0.35, −0.35) | LIMIT_EXCEEDED | SEARCH bottle → READY APPROACH › GRASP TARGET_LOST (the bottle left the head view after parking) → SEARCH … repeated to the plan limit |
| `final_fail_f34_c_eval_09_bottle.mp4` | f34 bottle at the far edge | SEARCH_EXHAUSTED | SEARCH red_region → SEARCH bottle ✘ ×2: the bottle is top-clipped in every view, which A rejects ("clipped object centre is below the table top") |
