# EE4705 Project 1.3 — Backbone

Owner: backbone (ALL)

Shared infrastructure for the language-instructed tabletop manipulation
project: frozen data contracts, MuJoCo simulation environment, plan
validation, orchestration, ground-truth mocks, evaluation/logging, and an
LLM/VLM client.  The three real modules (Student A perception, Student B
planner, Student C executor) are **interface stubs** here — see
"Directory ownership".

> **Design document note:** `docs/backbone_design_v3.md` was not present at
> implementation time; contract choices and their sources are recorded in
> `docs/DECISIONS.md` §0.  If the v3 document is added later, reconcile
> against that file first.

## Installation and tested versions

```bash
python3.12 -m venv .venv                      # any Python >= 3.10
.venv/bin/python -m pip install -e ".[dev]"
```

Tested with: Python 3.12.3, mujoco 3.12.0, numpy 2.5.2, PyYAML 6.0.3,
Pillow 12.3.0, requests 2.34.2, pytest 9.1.1 (Linux, NVIDIA EGL).
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

## Reference platform limitations

The self-contained platform in `assets/scene.xml` is deliberately
simplified (see `assets/README.md`):

- the "arm" is a **Cartesian 3-slide gantry**, not an anthropomorphic arm:
  no IK, no joint-space redundancy, orientation of the end effector is
  fixed;
- arm/torso/head geoms are **non-colliding**; grasping is weld-based
  attachment within 0.08 m of the EE (suction-style), not contact physics
  with fingers;
- the base is a frictionless-steering planar slide (x, y, yaw), so base
  motion is idealized;
- one attachment at a time; manipulable objects are rigid free bodies.

## Unitree G1 status

`assets/README.md` documents a reproducible MuJoCo Menagerie acquisition
path (upstream, revision pinning, license).  **G1 execution is not
implemented and not tested**: no hand control, locomotion, or IK.  Nothing
in this backbone should be described as validating G1 control.

## Running things

```bash
# full test suite (offline; includes real-environment acceptance tests)
.venv/bin/python -m pytest -q

# five smoke trials with all modules mocked (infrastructure check)
.venv/bin/python -m eval.runner --mode e2e --trials eval/trials/smoke --mock-all

# metrics for a finished run
.venv/bin/python -m eval.metrics runs/<run_dir>
```

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

Every run records its module configuration (`run_meta.json` and each
record's `module_config`); any run containing a mock is flagged
`infrastructure_check: true`.  **Mock-only results verify the backbone
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
