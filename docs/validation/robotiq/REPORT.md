# Robotiq replacement validation

Owner: backbone (ALL)

Date: 2026-09-06, Asia/Singapore. Environment: project `.venv`, MuJoCo
3.12.0, EGL. These are infrastructure/simulation checks, not real-hardware
or Student A/B/C performance claims.

## Status and contract decisions

This is the historical gripper-replacement validation snapshot (v1).
The subsequent student-handoff fixes and current contract v2 are documented
in [the student guide](../../STUDENT_README.md) and DECISIONS §13; the
measurements below have not been relabeled as a new run.

Implementation and bounded evaluation are recorded here. **Weld remains
default; physical is experimental.** The following two contract discrepancies
were surfaced during the original replacement review:

1. Add `RobotEnv.get_robot_state() -> RobotState`. Neither existed at HEAD,
   although the request explicitly requires testing state reports. Proposed
   snapshot fields: sim_time, base_pose, ee_pos, ee_quat (wxyz), measured
   gripper_opening, attached, last_attach_reason. No object identity/pose or
   oracle/model handles are exposed. The snapshot is read under the world lock.
2. Reject a position-only IK endpoint with actual penetration using
   `ValueError`, allowing existing `move_to` to re-park the base. This replaces
   the interrupted implementation's broader downward-only gate. Original
   high-target acceptance assertions are restored; upward poses without
   collision remain accepted. Without this guard, the existing real
   pick/carry/place regression times out with 0.396 m EE error after the
   forearm hits the table during cross-table movement. No Student module or
   reference skill implementation was edited to conceal this failure.

Existing method signatures, CONTRACT_VERSION=1, nearest-within-0.05 m opaque
attachment returns and all five trial configurations are preserved. The
request's `AttachmentResult` was also absent at HEAD; the actual
`Optional[str]` return is retained.

## Baseline and final commands

Baseline was captured by Claude **before model replacement**, recovered from
its saved baseline2 logs and session transcript (22:06–22:07 local). It was
not mislabeled by rerunning the new model as a baseline. Baseline HEAD is
`f4df9ef`. Baseline outputs are copied here; stdout's partial pytest dot line
is exactly the predecessor's saved output. Exit codes for the four baseline
commands were 0 in that transcript.

| Command (project venv on PATH) | Baseline | Final | Final exit |
| --- | --- | --- | --- |
| `pytest -q` | 89 passed in 8.41 s | **98 passed, 1 skipped in 16.80 s** | 0 |
| `python -m eval.runner --mode e2e --trials eval/trials/smoke --mock-all` | 5/5; wall 2.36 s | **5/5; wall 2.500 s** | 0 |
| `python scripts/ik_reach_test.py` | 23/25 IK/usable/tracked | **21/25 IK/usable/tracked; task poses all pass** | 0 |
| `python scripts/cam_sync_check.py` | 125 batches; zero timestamp/state delta | **125 batches, 375 observations; zero failures/deltas** | 0 |
| `python scripts/grasp_test.py` | Not present | **stone 0/10, cube 7/10, bottle 1/10** | **1** |

The physical evaluator's exit 1 honestly indicates performance below 7/10;
it is not swallowed or treated as success. The skip is the explicitly marked
canonical physical cube test because the feature ships experimental. All
89 existing tests remain, with camera mounting expectations updated to the
new physical mounts; original control assertions are retained. Ten gripper
tests were added (nine execute, one experimental skip).

Raw logs and machine-readable commands:

- [baseline pytest](baseline_pytest.txt), [smoke](baseline_smoke.txt),
  [reach](baseline_reach.txt), [camera sync](baseline_camsync.txt)
- [final pytest](final_pytest.txt), [smoke](final_smoke.txt),
  [reach](final_reach.txt), [camera sync](final_camsync.txt),
  [grasp](final_grasp.txt), [command metadata](final_commands.json)

The interrupted tree initially failed pytest collection because
`PHYSICAL_GRASP_EXPERIMENTAL` was undefined. Restoring ordinary collision
masks exposed the 1 mm pad margin's premature closure; margin was reduced to
0.3 mm without changing stroke/time assertions. The direct `pytest` entry
point also lacked the repository on sys.path; pytest's pythonpath setting
now makes it work identically to `python -m pytest` for `tests.conftest` imports.

## Reach and orientation comparison

The declared 5×5 x/y grid, z=0.90 m, table top 0.85 m, base/waist/ready poses,
task/trial coordinates and tracking thresholds were not relaxed or shrunk.
No trial target positions changed; SEARCH and clarification cases remain.

| Model / orientation | Mathematical IK | Collision-free IK endpoint | Tracked |
| --- | --- | --- | --- |
| Old palm TCP, 20° tilt / 30° azimuth | 23/25 | 23/25 | 23/25 |
| New pinch TCP, old 20° / 30° orientation | 25/25 | 2/25 | 4/25 |
| New pinch TCP, 45° / 0° orientation | 21/25 | 21/25 | 21/25 |

The second row reports separate endpoint/trajectory metrics; these are not
interchangeable. Its four task poses all failed. The new approach axis lies
along the wrist/forearm rather than the former palm normal, so retaining
the old orientation creates physically colliding solutions. This experiment
justifies the documented orientation change; the reachable area is still
smaller than the old model, and no full-workspace success is claimed.

Final grid failures: (0.28,−0.14), (0.28,−0.08), (0.32,−0.14),
(0.32,−0.08). Approach/grasp/lift/place all track collision-free.
[Final CSV](reach/reach_results.csv), [summary](reach/summary.json),
[plot](reach/reach_grid.png); [old-orientation output](old_orientation.txt),
[CSV](old_orientation/reach_results.csv), [summary](old_orientation/summary.json).

## Physical grasp experiment

Canonical smoke_1 poses, seeded independent uniform x/y offsets ±1 cm,
yaw offsets ±10°, 10 attempts per object. The reference approach/reach
adapters move real actuators. Grasp requires both pads on the same free
body within the gap; lift is 0.15 m in 0.03 m increments and hold is 1 s.
No object mass, size or original contact parameter was changed. The scene-wide
elliptic/impratio change was removed; tuning is confined to pads. After the
bounded experiments fell below the threshold, further tuning stopped.

| Object | Held | Failures | Slip checkpoint |
| --- | --- | --- | --- |
| stone | 0/10 | 10 slipped | all at 0.06 m commanded lift |
| cube | 7/10 | 2 single_pad_contact; 1 closed_on_nothing | none |
| bottle | 1/10 | 9 unreachable:TIMEOUT during reach | none |

[Full CSV](grasp/grasp_results.csv), [summary](grasp/summary.json).
CSV `slip_height_m` is the commanded lift at first detection (3 cm resolution),
while `measured_lift_m` is actual object displacement at that instant. It is
not an exact continuous slip-onset estimate. Representative right-wrist
failures: [stone slip](grasp/failure_frames/stone_00_failure.png),
[cube single pad](grasp/failure_frames/cube_07_failure.png),
[cube empty closure](grasp/failure_frames/cube_08_failure.png),
[bottle reach timeout](grasp/failure_frames/bottle_00_failure.png).
All final failures also have images in
`runs/grasp_test/20260906_231702/failure_frames/` (ignored run directory).

## Frames and camera checks

I inspected the rendered right-wrist open/closed images: finger tips remain
at the bottom and the center is clear. Segmentation verifies zero gripper
pixels in the central 20% for both wrists in both states. Median central
depth is 2.830 m right, 0.772 m left. Plane unprojection checks for both
wrist cameras and head pass in the full suite; effective clipping is
0.005/10 m. [Measured camera checks](frames/camera_check.json).

- [Head at ready, open](frames/head_ready_open.png)
- [Right wrist at ready, open](frames/right_wrist_ready_open.png)
- [Right wrist at ready, closed](frames/right_wrist_ready_closed.png)
- [Left wrist at ready, open](frames/left_wrist_ready_open.png)
- [Left wrist at ready, closed](frames/left_wrist_ready_closed.png)
- [Overhead at ready, showing robot/grippers](frames/overhead_ready_open.png)
- [Head during reach, showing the right gripper](frames/head_reach_open.png)

**Delivery limitation:** head-ready contains zero gripper pixels because the
unchanged head optics and unchanged ready posture place them outside its
field of view. The requested head-ready view showing a gripper is therefore
not satisfied. The supplemental head reach image is accurately labeled and
does not pretend to be the ready posture. Neither head calibration nor ready
joints was silently changed to make a delivery image.

## Remaining limits

Physical mode is experimental and below threshold for two objects. Broader
layouts, continuous slip timing, other MuJoCo versions and real hardware
remain unverified. Zero camera timestamp error proves atomic capture, not
full visual alignment capability. Mock smoke success is an infrastructure
check only. Mathematical IK reachability does not prove collision-free
motion at arbitrary targets. Current interface behavior and student readiness
are recorded separately in the subsequent student-handoff guide.
