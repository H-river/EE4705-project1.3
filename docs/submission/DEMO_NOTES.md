# Demo video notes

Video link: [video link]  (upload `docs/submission/EE4705_demo.mp4` as an unlisted YouTube / Google Drive video and paste the URL here)

File: `docs/submission/EE4705_demo.mp4` (3 min 46 s, 1440×900, 3.5 MB). Built by `scripts/make_demo_video.py`
from the clips in `docs/submission/episodes/`.

Each episode is one complete, unedited recording made by the project's Recorder: nothing inside an episode is cut,
sped up or reordered. Caption cards appear only between episodes. In every episode the left panel is an observer
camera (not an input to the system). The right panels show the head-camera image Student A last analysed and
Student B's current plan; the bottom bar shows the skill Student C is executing, then the independent evaluator's
verdict.

Episodes 1–6 are live end-to-end runs: Student A perception (Qwen-VL), Student B planner (Qwen, thinking off) and
Student C executor, on the submission code (branch `e2e`). They were re-recorded on 2026-09-27 with
`eval.runner --mode e2e --video --no-cache` on trials from `eval/trials/final50`, using 47 live API calls. The
system behaviour is identical to the headline run `final2-44`. Episodes 7–8 are the bonus: learned ACT policies
behind the same skill interface, in manipulation mode (ground-truth perception + rule planner, 0 API calls).

| Time | # | Trial | What it shows | Outcome |
|---|---|---|---|---|
| 0:12 | 1 | `f11_c_06_cube` | Standard task: perceive → plan (APPROACH, GRASP, MOVE_TO, PLACE, VERIFY, STOP) → execute → verify | PASS |
| 0:38 | 2 | `f14_c_09_bottle` | A different object (the tall bottle), first-attempt grasp and place | PASS |
| 1:04 | 3 | `f23_c_2_08_cube` | Scene variation: two stones also on the table; the cube instance is targeted | PASS |
| 1:32 | 4 | `f04_smoke_4_search` | Search: the robot starts facing away; B returns NEEDS_SEARCH, C turns until A grounds the stone, B re-plans | PASS |
| 2:09 | 5 | `f50_clarify_two_stones` | Clarification: two stones match "the stone"; B asks gray or dark red, the answer "the gray one" is used | PASS |
| 2:37 | 6 | `f49_reject_stack` | Infeasible request ("stack the cube on the bottle"): B refuses, no motion | REFUSED (correct) |
| 2:47 | 7 | `bonus_c1/c1_00` | Bonus: learned ACT grasp **and** learned ACT place | PASS |
| 3:12 | 8 | `bonus_c3/c3_01` | Bonus: learned ACT grasp (trained with bottle demos) on the bottle | PASS |
| 3:38 | — | — | Summary card: 44/50 final evaluation, 0 false claims; bonus learned grasp 47/50 | — |

Run folders (git-ignored): `runs/final/video2/r1` (f04, f49), `r2` (f14), `r3` (f11, f23, f50), `L1`, `L2` (bonus).
Checkpoints: `runs/bonus/best/act_2k_bottle` (grasp), `runs/bonus/best/place_act_wide` (place).

**Paragraph for the report, under the video link:** The video shows eight complete, unedited episodes. Episodes
1–6 are live end-to-end runs of the full system (Qwen-VL perception, Qwen planner and closed-loop executor) on
the submitted code. They cover a standard task, a different object, a cluttered scene, searching for a target
that starts out of view, asking a clarification question when two objects match, and refusing an infeasible
request. Episodes 7–8 show the bonus learned ACT policies driving the grasp and place skills behind the same
interface. Cards between episodes explain what to watch. The failure-reporting behaviour (the system reports
FAILED rather than claiming success when recovery runs out) is evaluated in the 50-trial results (0 false
claims) rather than shown in the video.
