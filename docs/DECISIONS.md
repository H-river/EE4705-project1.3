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
