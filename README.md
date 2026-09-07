# EE4705 Project 1.3 — Backbone

Owner: backbone (ALL)

Shared infrastructure for the language-instructed tabletop manipulation
project: frozen data contracts, MuJoCo simulation environment, plan
validation, orchestration, ground-truth mocks, evaluation/logging, and an
LLM/VLM client. Student A perception and Student C execution remain interface
stubs. Student B has a live Qwen planner adapter: 32/32 predefined planning
cases passed in the recorded evaluation. This is a small B-only test set;
see the [B guide](docs/STUDENT_B_README.md) and
[live results](docs/validation/QWEN_B_LIVE.md) for scope, commands and episodes.

**Students: start with the [student guide](docs/STUDENT_README.md).**
It explains each person's task in simple language, with runnable exercises,
input/output examples, test commands, and steps for connecting A, B, and C.

**Want to watch the workflow first?** The [offline demo guide](docs/DEMO_README.md)
shows how to record and replay a real simulated A → B → C episode, including
one failed grasp followed by a retry. It also lets you plug in your own B
while using the demo A and C. Its rule-based A/B are teaching baselines,
separate from the student implementations and course AI evaluation.

> **Design document note:** `docs/backbone_design_v3.md` was not present at
> implementation time; contract choices and their sources are recorded in
> `docs/DECISIONS.md` §0.  If the v3 document is added later, reconcile
> against that file first.

## Installation and tested versions

```bash
python3.12 -m venv .venv                      # any Python >= 3.10
.venv/bin/python -m pip install -e ".[dev]"
bash scripts/fetch_menagerie.sh               # G1 + Robotiq meshes/MJCF at the pinned revision
```

Tested with: Python 3.12.3, mujoco 3.12.0, numpy 2.5.2, PyYAML 6.0.3,
Pillow 12.3.0, requests 2.34.2, pytest 9.1.1, matplotlib 3.11.1 (Linux,
NVIDIA EGL); `ffmpeg` on PATH for `sync.mp4` (GIF fallback otherwise).
MuJoCo `>=3.1` is required but only 3.12.0 is verified — do not assume
every later version behaves identically (the depth-conversion check in
`docs/DECISIONS.md` §4 must be repeated when upgrading).

## Headless rendering

`core/rendering.py` sets `MUJOCO_GL=egl` (verified working backend on the
reference machine, NVIDIA driver) **before** `mujoco` is imported; all
backbone code imports mujoco via that module.  On machines without EGL:

```bash
MUJOCO_GL=osmesa python -m pytest -q      # software-rendering fallback (slower)
```

If neither backend initializes, rendering tests fail loudly — they are
never skipped.

## Platform: Unitree G1 (sliding humanoid)

The robot is MuJoCo Menagerie **Unitree G1 with two Robotiq 2F-85 grippers**,
composed in `assets/g1_2f85_ee4705.xml`. The pinned sources, licenses, flange,
TCP and camera transforms are in `assets/README.md`.

- The pelvis slides at fixed standing height through a rate-limited mocap
  weld; this remains a non-walking approximation.
- The right arm uses DLS IK and Cartesian streaming on the gripper's
  **pinch TCP**, with joint-rate limits and gravity compensation.
- **Weld grasping is the default**, with a 5 cm attachment radius and opaque
  handles. `SimWorld(grasp_mode="physical")` opts into experimental contact
  grasping; current success counts are stone 0/10, cube 7/10, bottle 1/10.
- `RobotEnv.set_gripper("right", opening)` sets a non-blocking target:
  0 closed, 1 open. Both grippers reset open.
- Cameras remain `head` (alias `onboard`), `left_wrist`, `right_wrist`;
  wrist cameras now mount on the gripper bases and retain finger references
  at the image bottom. Multi-camera capture remains atomic.
- Table top is 0.85 m; the unchanged 25-point reach grid yields 21 usable,
  tracked points, and all four task poses pass.

Dex3 assets are archived under `assets/archive/`; no hands variant is selected
at runtime. See `docs/DECISIONS.md` §12 and
[the validation report](docs/validation/robotiq/REPORT.md) for measurements,
contract reconciliation, images and remaining limitations.

## Running things

```bash
# full test suite (offline; includes real-environment acceptance tests)
.venv/bin/python -m pytest -q

# five smoke trials with all modules mocked (infrastructure check)
.venv/bin/python -m eval.runner --mode e2e --trials eval/trials/smoke --mock-all

# metrics for a finished run
.venv/bin/python -m eval.metrics runs/<run_dir>

# right-arm reachability grid + task poses (CSV, plot, summary under runs/ik_reach/)
.venv/bin/python scripts/ik_reach_test.py

# three-camera synchronization check with a falling ball (contact sheet, video,
# raw depth, diagnostics under runs/cam_sync/)
.venv/bin/python scripts/cam_sync_check.py

# experimental contact-grasp evaluation (30 attempts, CSV and failure images)
.venv/bin/python scripts/grasp_test.py
```

Physical arm control is validated by `tests/test_g1_control.py` and
`scripts/ik_reach_test.py` — NOT by the smoke trials, which use the
TeleportExecutor mock.

Evaluation modes: `grounding` (real A + mock B/C), `planning` (mock A +
real B + mock C), `manipulation` (mock A/B + real C), `e2e` (all real).
`--mock-all` replaces all three modules with mocks.

### Stub-related mode failures (expected today)

Without `--mock-all`, any mode that needs an unimplemented Student module
exits with **code 2** and names the missing module, e.g.:

```
ERROR: mode 'e2e' requires unimplemented module(s):
  - Student A perception (perception/student_a.py)
  ...
Use --mock-all to run an infrastructure check with mocks instead.
```

Mocks are never substituted silently.

## Mock-only versus real-system results

Every trial records its module configuration in `module_config`; a trial
containing a mock is flagged `infrastructure_check: true`. Run-level
metadata also records the mode and `--mock-all` flag. **Mock-only results verify the backbone
plumbing — they are never system performance and never validate Student
A/B/C work.**  Claimed success (the system's own verification) and actual
success (oracle judgement, `eval/criteria.py`) are always reported
separately.

## Directory ownership and student entry points

```
core/        backbone (ALL) — contracts, env, oracle, orchestrator, mocks, llm client
assets/      backbone (ALL) — scene, vocabulary, G1 acquisition docs
eval/        backbone (ALL) — runner, criteria, metrics, logger, smoke trials
tests/       backbone (ALL)
perception/  Student A — implement Perception in perception/student_a.py
planner/     Student B — implement Planner   in planner/student_b.py
executor/    Student C — implement Executor  in executor/student_c.py
```

Student entry points: subclass the ABCs in `core/interfaces.py`, keep the
class names (`StudentAPerception`, `StudentBPlanner`, `StudentCExecutor`),
and flip `IMPLEMENTED` to `True` when real.  Student modules must not
import `core.oracle` (enforced by `tests/test_architecture.py`) and the
executor only sees the ground-truth-free `RobotEnv` surface.

## Contract evolution rules

- `core/types.py` / `core/interfaces.py` changes require agreement of all
  three students; after freezing, only **add optional fields** — never
  remove or repurpose existing ones.
- Bump `CONTRACT_VERSION` on any semantic change; records embed it.
- Cross-module data passes only through `core/types.py` types; modules
  never import each other.

## Output / log formats

Each run creates a unique directory `runs/<timestamp>_<mode>[_mockall]/`:

- `run_meta.json` — mode, mock flags, contract version, infrastructure
  label;
- `<trial_id>/trial_record.json` — strict JSON `TrialRecord`: instruction,
  outcome (`TrialOutcome`), claimed vs actual success, per-action events
  (with frame ids), clarification exchanges (scripted responses consumed
  once), grasp-time oracle records, criteria checks, timings.  Enums are
  serialized as strings; non-finite numbers as `{"__nonfinite__": ...}` —
  `allow_nan=False` guarantees valid JSON;
- `<trial_id>/frames/frame_XXXXXX_{rgb.png,depth.npy,meta.json}` — RGB
  image, metric-depth array, and calibration/timestamp metadata for every
  frame the orchestrator marked (first view, post-action views, final
  verification), persisted no later than eviction from the 32-frame
  ObservationStore;
- `metrics.json` — aggregate metrics (see `eval/metrics.py` for
  denominator policy).

Partial records are still written when a trial crashes.
