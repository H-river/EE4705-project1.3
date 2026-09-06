# Backbone implementation decisions

Owner: backbone (ALL)

## 0. Missing design document `docs/backbone_design_v3.md`

At implementation time the work directory was empty: **no
`docs/backbone_design_v3.md` existed anywhere in the project or home
directory**.  The closest document found was
`~/Downloads/EE4705_backbone_design.md`, an earlier high-level plan (single
`SimEnv` exposing `get_state()`, no RobotEnv/EvalOracle split, no
`validate_plan`, no `GroundStatus`) — clearly pre-v3.

Resolution (recorded, not silent): the implementation prompt itself
specifies the v3 contract surface (SimWorld/RobotEnv/EvalOracle,
`validate_plan(plan, context) -> list[PlanError]`, `try_attach_near_ee()`,
`GroundStatus`, the 32-frame ObservationStore, the four eval modes, the
success criteria), so it was treated as the primary authority; the earlier
design doc supplied compatible details (the 8-skill enum, dataclass field
shapes, trial YAML shape, `objects.yaml` synonyms, `llm_client` call
surface).  Where both were silent, simple choices were made and recorded
below.  **If the real v3 document surfaces, place it in `docs/` and
reconcile against this file; contract renames then go through the normal
contract-evolution process (see README).**

## 1. Environment / dependencies

- Python 3.12.3 in a project-local venv (`.venv`), `pip install -e ".[dev]"`.
- Tested versions: **mujoco 3.12.0** (satisfies `>=3.1`), numpy 2.5.2,
  PyYAML 6.0.3, Pillow 12.3.0 (image writing), requests 2.34.2 (HTTP
  client), pytest 9.1.1.
- Headless rendering: **MUJOCO_GL=egl** (NVIDIA EGL verified on this
  machine).  `core/rendering.py` sets the variable before `mujoco` is
  imported; override with `MUJOCO_GL=osmesa` if needed.
- This machine has ROS Jazzy on `PYTHONPATH`; its pytest plugins are
  explicitly disabled in `pyproject.toml` (`-p no:launch_testing ...`)
  because they crash inside the isolated venv.

## 2. Contracts

- Public domain objects are **dataclasses** in `core/types.py` with
  boundary validation in `__post_init__` (`Observation`, `GroundedObject`).
  `CONTRACT_VERSION = 1`.
- Enums: `Skill` (8 values from the earlier design doc), `PlanStatus`
  (`READY`, `NEEDS_SEARCH`, `NEEDS_CLARIFICATION`, `INFEASIBLE`),
  `ErrorCode` (includes distinct `SEARCH_NOT_FOUND` vs `SEARCH_FATAL`),
  `GroundStatus` (`LOCALIZED`, `UNLOCALIZED`, `AMBIGUOUS`, `NOT_FOUND`),
  `TrialOutcome`.
- `GroundedObject.attributes` (e.g. `{"color": "gray"}`) was added so
  clarification responses ("the gray one") can be resolved without leaking
  ground-truth IDs.
- Executor results reference observations by `post_frame_id` (not by
  embedding the Observation) to keep records serializable; frames live in
  the ObservationStore.

## 3. Simulation

- **Angle units**: `scene.xml` declares `<compiler angle="radian"/>`.
  (MuJoCo's default is degrees; hinge ranges written in radians silently
  became ±3.2° limits — found and fixed during Step 9.)
- **Attachment** is a per-object predefined weld equality, activated with
  the object's current pose relative to `ee_body` written into `eq_data`
  (`[anchor(3), relpos(3), relquat(4), torquescale]`), verified empirically
  to give ~zero residual — so no snap.  `ATTACH_RADIUS = 0.08 m`,
  `MAX_ATTACHMENTS = 1`, ties broken by (distance, body id).  Custom
  `solref` on the welds was removed: a 4 ms time constant with a 5 ms
  timestep violates MuJoCo's stability bound and exploded.
- Arm links carry explicit masses (2.0/1.5/0.8 kg): with near-massless arm
  bodies the weld + stiff position actuators produced a limit-cycle bounce
  of the held object.
- Stones/bottle use `condim="6"` with rolling friction; without it the
  ellipsoid stone rolled forever (~1.5 cm/s) and failed the 2 s stability
  criterion.
- Unused manipulable objects are **parked** at (10+, 10, 0.05), far outside
  the workspace and every camera view; `stone2` visibility is therefore
  fully controlled by the trial config.
- `SimWorld.teleport_body` is an explicitly privileged method used only by
  the TeleportExecutor mock and tests; no RobotEnv path teleports objects.

## 4. RGB-D

- The installed `mujoco.Renderer`
  (`mujoco/rendering/classic/renderer.py:189-199`) already converts the
  z-buffer to **metric optical-axis depth in meters** — verified against a
  known-distance scene — so no second conversion is applied.  Depth
  ≥ 0.98·zfar is mapped to NaN (background/invalid).
- Public camera frame: +x right, +y down, +z forward;
  `T_world_camera = T_world_glcam · diag(1, −1, −1)`.
  Intrinsics: `f = H / (2·tan(fovy/2))`, principal point
  `(W/2 − 0.5, H/2 − 0.5)` (pixel centers at integer coordinates) —
  verified by unprojecting non-central table pixels onto the z = 0.40 plane.
- RGB and depth are two render passes over the same un-stepped `MjData`;
  arrays are copies (no renderer-buffer aliasing).

## 5. Plan validation

Per-skill requirements (v3 left them to the acceptance cases in the
prompt): see the docstring of `core/validation.py`, which is the normative
list.  Notable choices: SEARCH is legal only in `NEEDS_SEARCH` plans;
REACH/GRASP require `LOCALIZED` references; `VERIFY(object_visible)`
accepts `UNLOCALIZED`; positional params are validated for type, finiteness
and workspace bounds (±3 m, −0.5..2.5 m z); simulated held-state starts
from the live `ExecutionContext`.

## 6. Oracle association

Segmentation-render based: visibility = ≥ 30 unoccluded pixels (after a
4-of-8-neighbor denoise that removes stray mislabeled silhouette pixels the
renderer emits); association = greedy one-to-one bbox IoU with threshold
0.30; a perceived box whose top two IoUs are both ≥ 0.30 and within 0.10 is
reported ambiguous and left unmatched.  `associate()` refuses to run if the
world has stepped past the scene's `sim_time` (same-state guarantee).
Ground-truth and perceived IDs are never string-compared.

## 7. Orchestrator

Limits (all configurable in `OrchestratorConfig`): 2 search attempts, 2
clarifications, 4 replans, 2 attempts/action, 40 total actions, 10 total
plans — every branch is bounded.  A successful SEARCH invalidates the
remaining (blindly planned) actions and forces a replan.  Final
verification is vision-based (fresh observation; 3D region containment with
the region's known 0.08 m half-extent + 0.03 m slack, or 2D bbox-center
containment as fallback) and may reuse an in-plan
`VERIFY(object_in_region)` only if no state-changing action ran after it.

## 8. LLM client

One OpenAI-compatible `/chat/completions` adapter (`requests` transport
behind a `Transport` protocol; `FakeTransport` for tests).  Cache key =
SHA-256 over {cache format version, base URL, model, full messages
(including base64 image payloads), schema, temperature, max_tokens}; the
API key is never in keys, entries, or logs.  Retryable = HTTP
408/429/5xx + transport errors, bounded by `max_retries`.

## 9. Evaluation

- Stub detection: Student classes carry `IMPLEMENTED = False`; the runner
  refuses (exit 2, naming the module) to run a mode whose real module is a
  stub unless `--mock-all` is given.
- Wrong-object detection: the runner wraps the executor in
  `GraspSpyExecutor`, which records `oracle.held_gt_id()` at each
  successful GRASP — detection is purely from simulated state.
- Actual-success tolerances: region containment = x/y within the region's
  half-extents and z ∈ [support−0.005, support+0.12]; stability = 2.0 s,
  sampled every 0.1 s, drift ≤ 0.02 m from the first sample AND speed
  ≤ 0.05 m/s at every sample.
- Metrics denominators and zero-denominator behavior: see
  `eval/metrics.py` docstring (null, never 0/0).

## 10. Smoke scene contrivances

- SEARCH necessity (smoke_4): robot starts with yaw = 3.0 rad (facing away
  from the table), so nothing is initially visible.
- Clarification necessity (smoke_5): both stones (same class `stone`,
  different colors) are visible; the single scripted response
  "the gray one" resolves the ambiguity and is consumed exactly once.

## 11. Unitree G1 upgrade (2026-09-06)

Prerequisite note: `docs/backbone_design_v3.md` and `docs/RECONCILE_v3.md`
did not exist when this work started (see §0); the contract in force is the
code + this file.  All RobotEnv / EvalOracle signatures and semantics were
preserved; the only new public method is `RobotEnv.get_obs_multi`.
`CONTRACT_VERSION` stays 1.  `Observation` gained the optional
`capture_id: str = ""` field (additive, default-valued, needed for batch
identity); the pre-G1 `ATTACH_RADIUS` of 0.08 m was set to the specified
0.05 m.  The project was not a git repository; it was initialised with a
baseline commit of the pre-upgrade state so this task's commits are
separable.

### 11.1 Platform
- Model: Menagerie `unitree_g1/g1_with_hands.xml` @ `8161bba2…`, BSD-3;
  local adaptation `assets/g1_with_hands_ee4705.xml` (diff in
  `assets/g1_upstream_diff.patch`).  The hands variant was stable in every
  settling check (no NaNs, base drift < 1 mm, joint speeds → 0, no limit
  violations, no self-penetration at rest) → no `g1.xml` fallback.
- Timestep 0.002 s (upstream value; 0.005 s was also stable but leaves
  less margin for the kp = 500 finger actuators).  All step counts in
  backbone code are time-based.
- Sliding base = free pelvis welded (solref 0.01 s, solimp 0.99/0.999) to
  a rate-limited mocap body; legs position-held; feet excluded from world
  collisions only.
- Kinematics after `mj_step` lag one integration step; `SimWorld` calls
  `mj_kinematics/mj_comPos/mj_camlight` (never `mj_forward`, never
  stepping) before reading EE/camera poses or rendering, so rendered
  geometry, extrinsics and `sim_time` all describe the same `qpos`.

### 11.2 Control
- `set_base_target` / `set_arm_target` only update targets (non-blocking,
  verified by a test that sim time does not advance); `step` runs one
  `G1Controller.update()` per physics step (mocap rate limits, Cartesian
  setpoint streaming + warm-started IK, synchronized joint-rate limit,
  gravity feed-forward), then `mj_step`.  Live `qpos` is never written by
  the controller.
- Position-only commands request the default tilted top-down approach
  orientation (tilt 20°, finger azimuth 30° from the base heading — the
  widest collision-free band in the reach sweeps) and fall back to
  position-only IK when infeasible; the policy is recorded per command.
- The straight-line Cartesian stream was necessary: joint-space
  interpolation from the hanging rest pose swept the hand under the table
  slab (measured), and even Cartesian streaming from a low rest pose
  skimmed the table edge, hence the raised right-arm "ready" posture.
- IK branch control: shoulder pitch limited to ≤ 0.8 rad in IK (the
  hyper-extended-elbow criterion was wrong: this model's elbow zero is not
  the straight arm), a 0.6 rad continuity guard on streaming solutions, and
  a bounded-rate multi-seed re-solve when the warm start stalls.

### 11.3 Hand / EE / cameras
- Palm reference point (0.10, 0.065, 0) in `right_wrist_yaw_link`; the EE
  site's +z is the palm normal (grasp approach axis).  Thumb posture
  (−1.0, −1.0, −1.0) chosen from a 735-combination search: no
  self/object penetration, thumb tip 1.4 cm behind the palm point.
  `skills.GRASP_DESCEND_OFFSET = 0.02 m` keeps fingers clear of the
  support.
- Wrist cameras on brackets 7.5 cm beside the palm, aimed along the
  approach axis: best of 55 candidate mounts (7 % hand pixels, palm point
  visible, min depth 4 cm).  Head camera 30° down on the torso (≈ 47° with
  the waist lean) so the workspace band is image-centred.
- Effective clip: near 0.005 m / far 10 m computed from the compiled
  extent (2 m).
- Camera routing: head = describe, ground, SEARCH, final verification;
  wrist cameras = available to a future Executor for alignment and
  grasp/place checks; GTPerception and grounding evaluation are head-only.

### 11.4 Atomic multi-camera capture
- `SimWorld.capture` renders every requested camera (RGB then depth) under
  one `RLock` from one un-stepped state; K/T come from that same state;
  arrays are copies.  `RobotEnv.get_obs_multi` assigns
  `ep<episode>_capture<n>_<camera>` with a per-call counter (unique even
  without stepping); `get_obs` routes through the same path.  `frame_id`
  stays unique per observation (ObservationStore key); `capture_id` is
  persisted in frame metadata.

### 11.5 Measured results (this machine, 2026-09-06)
- `scripts/ik_reach_test.py` (grid x 0.28–0.44, y −0.32–−0.08, z 0.90,
  waist pitch 0.3): IK 23/25, collision-free 23/25, tracked 23/25 with
  residuals 1.1 mm / 0.19° mean (2.2 mm / 0.38° max), settle 1.23 s mean
  (1.68 s max) from the ready posture; failures (0.40, −0.32) and
  (0.44, −0.32) are IK reach-boundary failures (3.5 mm / 22.9 mm best).
  Task poses approach/grasp/lift/place: all tracked collision-free
  (1.14 / 0.48 / 0.70 / 0.66 s).
- `scripts/cam_sync_check.py`: 125 atomic batches (375 observations),
  max inter-camera timestamp difference 0, scheduling error 4e-13 s,
  physical-state change across capture 0, unique IDs; ball first table
  contact at 0.24 s (free fall from 0.30 m: 0.247 s expected); head camera
  detected the ball in 56 of 61 burst frames, wrist cameras never during
  the fall (out of their fields of view), right wrist during the reach;
  rendering ≈ 380–440 observations/s.


## 12. Bilateral Robotiq 2F-85 replacement (2026-09-06)

### 12.1 Hardware and restoration

Dex3 hands are retired for now: fixed finger weld grasping offered no useful
physical closing action, the thumb complicated low grasps, and a parallel
pad TCP provides a clearer manipulation frame. The old XML and original
upstream patch are archived under `assets/archive/` with restoration notes.
There is no selectable hands variant. Both G1 no-hands and 2F-85 sources
come from the already pinned Menagerie revision
8161bba264d7fa7c99ca301e91e7fb44737676ad; G1 is BSD-3-Clause and 2F-85 is
BSD-2-Clause (license copies tracked under `assets/licenses/`). The fetch
script now includes both directories. `build_g1_model.py` generates the
composition reproducibly; passive linkage equalities are retained.

### 12.2 Frames, contacts and posture

The wrist-yaw link's forward +x axis is the new approach axis. Wrist → mount
translation is (0.045,0,0), quaternion (0.5,0.5,0.5,0.5). Retaining the
upstream mount → base transform yields effective wrist → TCP translation
(0.1938,0,0), with +z approach, +y closing and +x pad width in the TCP frame.
The old palm point was (0.10,0.065,0) and had a different approach axis.
The right ready joints remain unchanged; measured TCP is roughly
(0.433,−0.564,0.980). Settling and task-pose tests pass without weakened
assertions. The fixed head camera does not see the grippers at ready.

Collision bits distinguish grippers from ordinary scene bodies while
retaining upstream gripper internal/mutual collisions. Only the held
object's gripper-contact eligibility is removed in weld mode; table/object
contacts continue. Six gripper linkage equalities stay active. The
interrupted implementation's masks disabled every gripper/gripper contact;
that was corrected before final validation. No object sizes or masses changed.

### 12.3 IK comparison and contract reconciliation

Grid, height, base pose, waist pose, ready joints and all smoke trial
positions are unchanged. New TCP with old 20°/30° approach produced only
2/25 collision-free IK endpoints and failed all four task poses. Because
that approach no longer describes a workable gripper pose, default tilt /
finger azimuth change to 45°/0°. Final new TCP: 21/25 IK, usable and tracked;
old palm: 23/25. Approach/grasp/lift/place all pass with the new orientation.
The original test accepting a high position-only target is retained.

Two discrepancies were explicitly surfaced to the user for reconciliation:
(1) the repository had no RobotState/get_robot_state even though the prompt
requires it; (2) a mathematically reachable fallback solution for cross-table
transport penetrates the torso. The proposed snapshot adds only robot
proprioception, and the proposed fallback guard rejects actual endpoint
penetration, not upward orientation by itself. Existing move_to then uses
its existing base-reparking branch. See §13 for the subsequent handoff update.
CONTRACT_VERSION remains 1; existing RobotEnv and EvalOracle signatures are
unchanged. AttachmentResult also did not exist: Optional[str] opaque-handle
semantics are preserved. Student A/B/C files are unchanged.

### 12.4 Cameras

Wrist cameras mount on lg_base/rg_base at (−0.058,0,0.02), xyaxes
`0 -1 0  -0.9962 0 0.0872`, FOV 90°. The lens clears the housing and looks
5° toward the approach axis, with open and closed finger tips at the bottom.
Central 20% gripper fraction is zero for both wrists in both states; ready
median depths are 2.830 m right / 0.772 m left. Effective near/far clipping
remains 0.005/10 m. Plane unprojection checks pass for both wrists. Head
mount/optics are unchanged. Sync gives 125 batches/375 observations with no
capture failure, timestamp difference or physical-state mutation.

### 12.5 Bounded physical mode

Default is weld; physical is explicitly experimental. Upstream gripper
position gain/bias/control/force range and all linkage constraints remain.
Close targets are rate-limited, measured from driver joints, and checked
against bilateral pad contact with the same free body inside the gap.
Pad friction 1.0/0.02/0.002, condim 4, solref 0.004/1, solimp
0.95/0.99/0.001; margin is 0.3 mm. The initial 1 mm margin stopped the pads
short of the closed target; reducing it fixed the full-stroke acceptance
check. Scene-wide elliptic cone/impratio tuning from the interrupted work
was removed to respect the pad/object-only scope; object parameters remain
unchanged. Tuning stopped after bounded evaluation below the 7/10 threshold.

Final seeded ±1 cm / ±10° physical attempts: stone 0/10 (slips detected at
6 cm commanded lift), cube 7/10 (2 single-pad contacts, 1 closed-on-nothing),
bottle 1/10 (9 reach timeouts). CSV includes measured object displacement at
slip detection, whose command checkpoints are 3 cm apart; exact continuous
slip-onset heights are not claimed. The canonical cube unit test is
skip-marked with an experimental-mode reason. Full results and representative
failure images are tracked in docs/validation/robotiq/.

### 12.6 Validation and reproducibility

Baseline from Claude's pre-change saved output: 89 tests, smoke 5/5 in
2.36 s, reach 23/25, sync exit 0. Final: 98 passed/1 experimental skip,
smoke 5/5 in 2.50 s, reach 21/25 with all task poses passing, sync exit 0,
physical evaluator exit 1 (expected below-threshold result). The pytest
configuration now adds the repository root to pythonpath so the requested
`pytest -q` command resolves `tests.conftest` just like `python -m pytest`.
Detailed raw outputs, exit codes, provenance and images are in the report.

## 13. Student handoff fixes (2026-09-07, contract v2)

The user requested implementing the three readiness fixes and documenting
each student's work, inputs/outputs and tests. The historical version-1
statements and measurements above describe earlier snapshots. Current
`CONTRACT_VERSION = 2` distinguishes the stricter verification and full-motion
stop from those records; existing records are not rewritten.

- `core.action_targets.resolve_action_position` resolves exact perceived IDs
  from fresh perception, without oracle access. Existing explicit `params.pos`
  waypoints remain overrides. MOVE_TO a region uses its support point + 0.18 m
  (the existing RulePlanner clearance); PLACE uses the support point. Missing,
  ambiguous or unlocalized references fail. The validator also checks waypoint
  overrides and rejects PLACE without either a localized region or an explicit
  position. Student C can reuse this helper; the mock executor already does.
- `RobotEnv.stop_motion` cancels the base, waist, arm and gripper trajectories
  and holds measured poses. It changes control targets only, never qpos/qvel/time.
  Attachment is preserved; a physically held right gripper retains its closing
  command. Orchestrator calls this on every terminal path, including refusal,
  exhausted search/clarification and exceptions. Stop failures are recorded as
  ERROR, not swallowed. RobotEnvProtocol now includes the concrete observation,
  proprioception, gripper, timestep and stop surfaces used by student modules.
- `core.verification.verify_placement` requires fresh 3D estimates of the exact
  object and region IDs, no attachment, finite region XY bounds, support height
  offset in [-0.005, 0.12] m and <= 0.02 m drift over two views 0.2 sim seconds
  apart. There is no same-class replacement, bbox-only success, or extra 3 cm
  XY slack. A successful in-plan VERIFY never bypasses final verification.
  Geometry is shared with the oracle through a pure function in `core.placement`;
  the data sources stay separate and the evaluator retains its 2 s stability
  check. This remains a perception-based claim, not privileged actual success.

`GroundedObject.region_half_extents_xy` is an optional axis-aligned half-size
in world meters. Region pos_world denotes its support-surface center. When
no extent is supplied, known class geometry from assets/objects.yaml is used;
unknown regions fail closed. This is not ground-truth pose access and does not
implement region randomization in SceneConfig.

Regression cases and a full ID-only episode are in
`tests/test_student_handoff.py`; the development/testing guide is
`docs/STUDENT_README.md`. Real A/B/C remain stubs. The guide explicitly
separates mixed-mode integration checks from independent student metrics and
lists region variation and full Task 5 metrics as remaining evaluation work.
