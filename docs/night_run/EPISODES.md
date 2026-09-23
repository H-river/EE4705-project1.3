# Episode gallery (night run + rounds 2–3)

These are 10 e2e episodes: 5 where the robot claimed success and really achieved it, and 5 failures with different causes. Every video was rendered from data already on disk, with **0 live calls**:

```
python scripts/render_episode.py <trial_dir> <out.mp4>
```

What a video shows:
- **Frames.** Each slide is one head-camera observation. The frames are the ones the runner saved plus every frame Student A analysed in that episode. A's audit folder also keeps SEARCH views and frames the provider refused.
- **Boxes.** Boxes are A's own output for that exact image: green = LOCALIZED, yellow = UNLOCALIZED, magenta = AMBIGUOUS.
- **Captions.** A caption lists the events since the previous frame (plan, action result, clarification, verification).
- **Cards.** A title card opens the video and an outcome card closes it.

It is a slideshow of observations, not continuous video: the runner does not record between observations.

All videos are in `docs/night_run/episodes/` and each is < 0.6 MB, so no LFS is needed. "run dir" is the folder under `runs/night/` (git-ignored) that holds the trial record, frames and log.

## Successes (claimed ∧ achieved)

### S1. smoke_3_instruction_variation: paraphrased instruction
- **instruction:** "could you put that gray rock over on the red marker please?"
- **outcome:** CLAIMED_SUCCESS (claimed T / actual T), error_code –, 0 replans
- **timeline:**
  1. t=1.0 B maps "gray rock" → stone a2 and "red marker" → red_region a1: READY APPROACH › GRASP › MOVE_TO › PLACE › VERIFY › STOP
  2. t=3.5–7.5 APPROACH a2 ok, GRASP a2 ok, MOVE_TO a1 ok
  3. t=11.3 PLACE ok: "offset [-0.009, -0.031, 0.022], released and stable across 0.2 s"
  4. t=11.9 final verification passed from a fresh frame
- **attribution:** – (all three modules worked). The execution is identical to smoke_1_standard, which has the same scene.
- **video:** `docs/night_run/episodes/ok_smoke_3_instruction_variation.mp4` (run dir `r2_smoke/smoke_3_instruction_variation`)

### S2. smoke_2_scene_variation: different layout
- **instruction:** "move the stone onto the red region"
- **outcome:** CLAIMED_SUCCESS (T/T), error_code –, 0 replans
- **timeline:**
  1. t=1.0 all targets LOCALIZED in the first frame → READY
  2. APPROACH, GRASP, MOVE_TO ok
  3. PLACE: the first check from the retreat pose is a view problem. One view retry, back at the last localized viewpoint (round-1 fix #7), passes: offset 0.5 cm
  4. final verification passed
- **attribution:** –. Before round 1 this trial was REFUSED although the stone was placed correctly (fixes #7 and #9 in CHANGES.md).
- **video:** `docs/night_run/episodes/ok_smoke_2_scene_variation.mp4` (run dir `r2_smoke/smoke_2_scene_variation`)

### S3. smoke_5_clarification: ambiguous referent → clarification
- **instruction:** "move the stone onto the red region"
- **outcome:** CLAIMED_SUCCESS (T/T), error_code –, 1 replan (after the answer)
- **timeline:**
  1. t=1.0 frame 0: A sees a gray stone and a dark_red stone (both LOCALIZED). B returns NEEDS_CLARIFICATION
  2. B asks "There are two stones: a gray one and a dark red one. Which stone should I move to the red region?" and gets the answer "the gray one"
  3. B replans READY on a3 (gray); APPROACH/GRASP/MOVE_TO ok; PLACE verified after 2 view retries
  4. t=11.9 final verification passed (offset ≈3 cm, inside the 8 cm half extent)
- **attribution:** –. In the night run this trial ended LIMIT_EXCEEDED (ground()/describe() disagreement). Fixes #8 and #10 removed that.
- **video:** `docs/night_run/episodes/ok_smoke_5_clarification.mp4` (run dir `r2_smoke/smoke_5_clarification`)

### S4. c_2_08_cube: SEARCH first, then manipulate
- **instruction:** "Move the blue cube to the red area."
- **outcome:** CLAIMED_SUCCESS (T/T), error_code –, 1 replan
- **timeline:**
  1. t=1.0 the cube is not localized → NEEDS_SEARCH [SEARCH cube]
  2. t=3.7 SEARCH cube ok (search_found) → READY
  3. t=4.5–11.8 APPROACH a1, GRASP a1, MOVE_TO a2 ok
  4. t=15.7 PLACE ok (offset 0.5 mm); t=16.3 final verification passed
- **attribution:** –. This trial was REFUSED in the night run and after round 1 (the region stayed UNLOCALIZED at verification). The experimental partial-region fix (#9) made it pass.
- **video:** `docs/night_run/episodes/ok_c_2_08_cube.mp4` (run dir `r2_v2/c_2_08_cube`)

### S5. c_2_06_cube (round 3): recovers from a provider content-filter refusal
- **instruction:** "Move the blue cube to the red area."
- **outcome:** CLAIMED_SUCCESS (T/T), error_code –, 0 replans
- **timeline:**
  1. t=1.0 READY; APPROACH a0, GRASP a0, MOVE_TO a1 ok
  2. t=10.15 frame 9, first PLACE check: the cube box touches the image edge, so the cube is UNLOCALIZED and C does a view retry
  3. t=11.47 frame 10 (view retry 1): **the provider refuses the image** ("Input data may contain inappropriate content"). A returns an empty frame, and verification returns `passed=None`, which counts as a view problem
  4. t=11.87 frames 11–12 (view retry 2, +0.2 rad): cube and region LOCALIZED; PLACE verified (offset 0.9 cm)
  5. t=12.7 final verification passed
- **attribution:** –. This is the after-run for fix #14. The same trial ended ERROR in round 2 (F5).
- **video:** `docs/night_run/episodes/ok_c_2_06_cube_content_filter_recovered.mp4` (run dir `r3/c_2_06_cube`)

## Failures

### F1. c_2_01_stone (night run): placed correctly, region UNLOCALIZED → REFUSED
- **cause class:** perception UNLOCALIZED
- **instruction:** "Move the gray stone to the red area."
- **outcome:** REFUSED (claimed F / **actual T**), last error_code UNREACHABLE, 3 replans
- **timeline:**
  1. t=1.0 READY; APPROACH a2, GRASP a2, MOVE_TO a1 ok
  2. t=9.8 PLACE → PLACE_FAILED "reliable 3D grounding is required". Frames 10–11 show the stone inside the red region, but the region box touches the right image edge, so A marks it UNLOCALIZED (yellow)
  3. B replans NEEDS_SEARCH; the SEARCH arm tuck fails: TIMEOUT "Arm did not tuck", then UNREACHABLE ×3 "position-only IK intersects robot or scene geometry"
  4. t=14.8 B: INFEASIBLE "Cannot localize targets after repeated search failures" → REFUSED
- **attribution:**
  - A: the edge-clipped region is UNLOCALIZED.
  - C: the PLACE view retry covered only "not visible", and the tuck failure came from fix 2.4.
  - Fixed by round 1 (#7) and the revert of 2.4 (#3). This trial is CLAIMED_SUCCESS since round 1.
- **video:** `docs/night_run/episodes/fail_c_2_01_region_unlocalized.mp4` (run dir `e2e_v2/c_2_01_stone`)

### F2. c_2_03_stone (round 2): SEARCH loop until max_total_plans
- **cause class:** search loop
- **instruction:** "Move the gray stone to the red area."
- **outcome:** LIMIT_EXCEEDED (F/F), no failed action code (every SEARCH reports ok), 9 replans
- **timeline:**
  1. t=1.0 NEEDS_SEARCH stone. The sweep turns the head far past the table: frames 4–11 show only the floor and the robot's shadow, and the VLM reports AMBIGUOUS "gray stone" boxes on the shadows (magenta, frames 5, 10, 11)
  2. t=8.9 SEARCH stone ok → B: the red region is not localized → SEARCH red_region ok (t=9.9) → B: the stone is not visible … (ping-pong, 2 rounds)
  3. t=16.0 frames 36–52: the same pose shows the stone LOCALIZED and the bottle, but no red region. SEARCH red_region reports "ok" 7× in a row at an unchanged sim time, while B's next describe() has no localized red region
  4. max_total_plans → LIMIT_EXCEEDED
- **attribution:**
  - A/C: SEARCH accepts a target that the next describe() does not report as LOCALIZED. This is the ground()/describe() disagreement, K5 in REPORT §5.
  - C: the sweep spends about half its views looking away from the table.
  - Still open.
- **video:** `docs/night_run/episodes/fail_c_2_03_search_loop.mp4` (run dir `r2_v2/c_2_03_stone`). The night run's `e2e_v2/c_2_07_cube` is another search loop, a phantom "red cube" accepted 10×; fix #8 addressed it.

### F3. c_2_05_stone (night run): torso–table contact on every APPROACH
- **cause class:** scene obstacle
- **instruction:** "Move the red stone to the red area."
- **outcome:** LIMIT_EXCEEDED (F/F), last error_code TIMEOUT, 9 replans
- **timeline:**
  1. t=4.2 SEARCH stone ok → READY on the dark_red stone a3
  2. t=5.3, 5.8 APPROACH a3 → TIMEOUT "Table contact detected; backed off, approach not completed" (contact_recovery)
  3. B replans SEARCH; t=10.0 SEARCH also hits the table (TIMEOUT), then SEARCH stone/red_region ok ×7
  4. t=13.5, 14.1 APPROACH a3 → the same table contact again → LIMIT_EXCEEDED
- **attribution:** C. The approach heading drives the torso into the table edge, and there is no alternative heading. Fix 2.5 (#4) turned the backed-off "success" into TIMEOUT and so exposed it; after the revert, manipulation passes but e2e still fails (round 2: GRASP TARGET_LOST after the contact). Still open.
- **video:** `docs/night_run/episodes/fail_c_2_05_table_contact.mp4` (run dir `e2e_v2/c_2_05_stone`)

### F4. c_2_09_bottle (round 2): GRASP loses the target after APPROACH
- **cause class:** grasp
- **instruction:** "Move the green bottle to the red area."
- **outcome:** SEARCH_EXHAUSTED (F/F), last error_code SEARCH_NOT_FOUND, 4 replans
- **timeline:**
  1. t=1.0 READY (bottle a0 LOCALIZED in frame 0)
  2. t=3.9 APPROACH a0 ok, but at the approach pose (frame 3) the bottle box touches the right image edge → UNLOCALIZED → GRASP → TARGET_LOST "needs fresh, unambiguous 3D evidence"
  3. SEARCH bottle: table contact (TIMEOUT), then SEARCH_NOT_FOUND; most sweep views face the floor (frames 10–19, 27–35)
  4. t=17.1 SEARCH bottle ok, SEARCH red_region ok (the bottle is out of view again), then t=23.8 SEARCH bottle NOT_FOUND → SEARCH_EXHAUSTED
- **attribution:**
  - C: the approach pose puts a tall object at the image edge. GRASP needs fresh 3D evidence and has no re-centre step.
  - A: a box clipped by the edge is UNLOCALIZED for objects (the partial fix #9 covers regions only).
  - Still open (REPORT §8 item 2).
- **video:** `docs/night_run/episodes/fail_c_2_09_grasp_target_lost.mp4` (run dir `r2_v2/c_2_09_bottle`)

### F5. c_2_06_cube (round 2): provider content filter → ERROR
- **cause class:** content filter
- **instruction:** "Move the blue cube to the red area."
- **outcome:** ERROR (claimed F / **actual T**), error_code INTERNAL_ERROR, 0 replans
- **timeline:**
  1. t=1.0 READY; APPROACH a0, GRASP a0, MOVE_TO a1 ok (frames 0–7)
  2. t=11.5 PLACE releases the cube inside the region. The verification frame (frame 11, red banner) is refused by the provider with HTTP 400 "Input data may contain inappropriate content"
  3. APIError propagates out of PLACE → INTERNAL_ERROR → the episode ends in ERROR. The evaluator finds the cube in the region
- **attribution:** provider (A's VLM call). Fixed in round 3 by #14; see S5 for the after-run.
- **video:** `docs/night_run/episodes/fail_c_2_06_content_filter.mp4` (run dir `r2_v2/c_2_06_cube`)

## Episodes not re-run

None of these 10 episodes was re-run for the gallery. All of them had saved frames, and A's audit folders supplied the rest.
