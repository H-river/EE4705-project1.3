# Student C: make each robot action work

Your job is to turn one action from B into robot movement, check the result,
and tell B what happened. For example: move near the stone, pick it up, carry
it to the red area, release it, and check that it stayed there.

There is now a working starting point. `executor/student_c.py` selects
`ClosedLoopExecutor` in `executor/closed_loop.py`. Improve that shared class
or override methods in `StudentCExecutor`. The demo uses the same control
code; only its optional failure injection lives in `executor/demo_skills.py`.

This is a **MuJoCo simulation with weld attachment**. The robot really moves
through the controller; objects are not teleported. It does not establish
contact-only grasping or real-robot performance.

## 1. Run a complete example

From a terminal:

```bash
cd /home/jiamo/EE4705/project1.3
c_episode="runs/my_c_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m demo.run --student C --out "$c_episode"
.venv/bin/python -m http.server 8768 --bind 127.0.0.1 --directory "$c_episode"
```

Open `http://127.0.0.1:8768/`. If that port is already serving the refinement
report, use 8769. Stop the server with Ctrl+C.

The example uses **RGB-D demo A + rule-based demo B + Student C**. Watch
`episode.mp4`, then inspect `episode.json`. `claimed_success` is the system's
visual judgement; `actual_success` is a separate simulator truth check.

To try a changed scene and record all intermediate actions:

```bash
.venv/bin/python -m demo.run --student C \
  --scene-trial eval/trials/student_c/c_04_stone.yaml \
  --out "runs/my_c_changed_scene_$(date +%Y%m%d_%H%M%S)"
```

`--scene-trial` reads the scene, instruction and evaluator target from YAML.
It does not change which A/B/C modules are selected. It currently accepts
simple trials without scripted clarification or injected faults.

## 2. Input and output

```python
result = executor.execute(action, env, perception)
```

| Value | Meaning |
| --- | --- |
| `action` | One `Action` from B, such as `Action(Skill.GRASP, "p0")` |
| `env` | Public camera, robot state, movement and attachment methods |
| `perception` | A's `describe` and `ground` methods, used on fresh images |
| `result.success` | Whether this action passed its own checks |
| `result.error_code` | `NONE`, or a reason B can use to recover |
| `result.post_frame_id` | Camera frame after the action |
| `result.recovery_attempted` | Whether C used an internal recovery |
| `result.info` | Actual TCP error, attachment state, elapsed simulation time and recovery details |

Positions use world coordinates in **meters**. IDs such as `p0` or `a2`
belong to A. They are not simulator object names. C must not import the oracle.

A failed action can still change the world. For example, grasping may attach
an object but the lift may fail; placement may release it but the camera may
not see the result. C reports `attached_after` and `held_instance_id`. The
backbone updates B's context and replans instead of blindly repeating that
partially completed action.

## 3. What each skill now checks

| Skill | Current behavior | Useful failure output |
| --- | --- | --- |
| SEARCH | Observe up to 13 views within 30 simulated seconds. Accept only a fresh, unambiguous 3D target. Stop base motion between views. | `SEARCH_NOT_FOUND`, `SEARCH_FATAL`, `TIMEOUT`, views checked |
| APPROACH | Park the base near the target, check its pose, stop and settle. | `TIMEOUT` |
| REACH | Move the TCP to the requested point and independently check its measured position. Tolerance remains 0.012 m. | `UNREACHABLE`, `TIMEOUT`, `ee_error_m` |
| GRASP | Re-observe the exact object, open the gripper, attempt attachment, lift 0.12 m and check holding. Retry a miss at most once with a fresh image. | `TARGET_LOST`, `GRASP_MISSED`, `ALREADY_HOLDING`; every local attempt |
| MOVE_TO | Require holding, check measured arrival and attachment. After a timeout/unreachable point, change the base parking pose once and retry reaching. | `NOT_HOLDING`, `TIMEOUT`, `motion_recovery` |
| PLACE | Account for the grasp offset, move above the support, open before releasing, retreat, and verify the exact object twice. If the result is outside the camera view, try at most two nearby headings. | `PLACE_FAILED`, `TIMEOUT`, visual check and view attempts |
| VERIFY | Check exact-ID visibility, tracked holding, or released/stable placement. A visible object may lack usable depth. | `VERIFY_FAILED` |
| STOP | Cancel motion, preserve attachment, and require two consecutive stable 0.1 s samples, within 2 s. | `TIMEOUT`, `NOT_HOLDING`, measured drift |

Motion primitives have bounded timeouts, normally 12 simulated seconds per
reach stage. A long reach can include a waypoint and a final point; a complete
action can therefore take longer than 12 s. C's local recovery budget and the
backbone's action retry/replan budgets are separate, and both are recorded.

Holding the right perceived ID is C's tracked belief. The independent
evaluator checks which physical object was actually attached. C cannot prove
physical identity from the attachment boolean alone.

## 4. Test C independently of A and B

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m pytest -q tests/test_student_c.py tests/test_student_handoff.py

c_test_root="runs/my_c_trials_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m eval.runner --mode manipulation \
  --trials eval/trials/student_c --out "$c_test_root"
```

The runner prints its final directory. Use that directory below, or resolve it
from the variable in the same terminal:

```bash
c_run=$(find "$c_test_root" -mindepth 1 -maxdepth 1 -type d -print -quit)
.venv/bin/python -m eval.skill_metrics "$c_run"
```

This mode uses **ground-truth mock A + rule mock B + Student C**. It isolates
C's execution. Do not add `--mock-all`: that replaces C with teleportation.

Inputs are the ten YAML files in `eval/trials/student_c/`. Their expected
object labels are read only by the evaluator. Outputs are:

| File | Read it for |
| --- | --- |
| `metrics.json` | Trial success, false claims and wrong-object counts |
| `skill_metrics.json` | Success per action attempt, recovery cases, final XY error |
| `<case>/trial_record.json` | Every action result, failure reason, state change and independent checks |
| `<case>/frames/` | RGB, metric depth and camera parameters for logged observations |

A grasp rate of 10/11 means one failed action is still in the denominator.
A task success rate of 10/10 can include retries. The error metric is the
object center's horizontal distance from the region center, measured by the
oracle after its stability check; object height is reported separately.

The current development result is **10/10 completed, 7/10 without retry or
recovery, grasp 10/11, place 10/10, mean XY error 1.17 cm**. The cases were
used during refinement, so these numbers are not an unseen success estimate.
A separate set of ten layouts, frozen after the control changes, completed 9/10; only 4/10 needed no retry/recovery. The retained bottle failure shows a collision configuration that repeated parking cannot resolve. See [the refinement report](validation/AC_REFINEMENT.md) for all cases and both sets of metrics.

## 5. What C should improve next

1. Reduce the two remaining first-attempt failures: transport in `c_04_stone`
   and grasp motion in `c_05_stone`. Inspect the TCP error and base pose, then
   test a new motion change on both the old suite and new scenes.
2. Test target positions and base headings outside the current ten examples.
   Keep labels fixed before running. Include both failed and successful trials.
3. Test with A's estimated positions and B's Qwen plans. Keep component scores
   separate from full ABC success; a perception error can look like a skill error.
4. Treat contact-only grasping as a separate experiment. Do not remove the weld
   and reuse the current success numbers.

To connect B after its key is available in this terminal:

```bash
.venv/bin/python -m demo.run --student B --student C \
  --out "runs/my_bc_$(date +%Y%m%d_%H%M%S)"
```

This calls the Qwen B API and still uses demo A. It has not been rerun with a
live key during the C/A refinement session.


To reproduce the additional layouts separately:

```bash
.venv/bin/python -m eval.runner --mode manipulation \
  --trials eval/trials/student_c_evaluation \
  --out "runs/my_c_additional_$(date +%Y%m%d_%H%M%S)"
```

If you tune C using those results, treat them as development cases from then
on and collect a new set for your next independent evaluation.
