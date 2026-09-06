# Assets

Owner: backbone (ALL)

## Scene files

| File | Purpose |
| --- | --- |
| `scene.xml` | Production scene: Unitree G1 (with hands) + shared tabletop world. Used by every test and smoke trial. |
| `scene_common.xml` | Shared world: lighting, finite floor, table, red region, manipulable objects, grasp welds, render/clip settings. |
| `scene_cam_sync.xml` | Diagnostic variant = production scene + a magenta free-falling ball (`diag_ball`). Used only by `scripts/cam_sync_check.py`; the ball is not a manipulable object and can never be attached. |
| `g1_with_hands_ee4705.xml` | Local adaptation of the upstream G1 model (see below). |
| `g1_upstream_diff.patch` | Every change relative to upstream (`scripts/g1_upstream_diff.sh` regenerates it). |
| `menagerie_revision.txt` | Pinned MuJoCo Menagerie commit. |
| `menagerie/` | Git-ignored sparse checkout of `unitree_g1` (meshes + upstream MJCF), fetched by `scripts/fetch_menagerie.sh`. |
| `objects.yaml` | Object vocabulary (unchanged). |
| `reference/scene_simple_legacy.xml` | The pre-G1 Cartesian-gantry platform, kept for reference only; no code loads it. |

## Unitree G1 (MuJoCo Menagerie)

* Upstream: <https://github.com/google-deepmind/mujoco_menagerie>, directory
  `unitree_g1`, revision **`8161bba264d7fa7c99ca301e91e7fb44737676ad`**
  (`assets/menagerie_revision.txt`; fetched 2026-09-06 with a blob-less
  sparse checkout, `scripts/fetch_menagerie.sh`).
* License: the `unitree_g1` directory is **BSD-3-Clause** (Unitree
  Robotics, `assets/menagerie/unitree_g1/LICENSE`); the Menagerie repository
  itself is Apache-2.0.
* Selected variant: **`g1_with_hands.xml`** (29 body DoF + 2×7 Dex3 hand
  DoF, 43 position actuators, `stand` keyframe).  It was physically stable
  in every settling/motion check (finite state, sub-millimetre base drift,
  joint speeds decaying to < 0.02 rad/s, no joint-limit violations, no
  self-penetration at rest), so the `g1.xml` fallback was **not** needed.
  Low rendering throughput was never treated as evidence of instability.
* The upstream file is never edited in place: `g1_with_hands_ee4705.xml` is
  a copy with every change marked `EE4705:` and listed in
  `g1_upstream_diff.patch`.  `meshdir` points into the upstream checkout.

### Local adaptation (what differs from upstream)

1. `meshdir="menagerie/unitree_g1/assets"`; upstream light removed (the
   scene provides lighting).
2. World-child **mocap body `base_target`** + **`base_weld`** (mocap →
   pelvis, `solref="0.01 1"` = 5× the 0.002 s timestep and
   `solimp="0.99 0.999 0.001 0.5 2"`; with the default impedance the pelvis
   lagged its target by 2.4 cm and rose 1.2 cm while translating under the
   torso's lean moment; now height < 1 mm, planar < 2 mm, yaw < 3 mrad).
3. **End-effector sites** `right_ee_site` / `left_ee_site` on the
   `*_wrist_yaw_link` bodies (palm reference points, below).
4. **Cameras** `head`, `left_wrist`, `right_wrist` (below).
5. **Contact exclusion** of the two `*_ankle_roll_link` (foot) bodies with
   `world` only — nothing else is filtered.

### Sliding-base approximation (no walking)

The pelvis keeps its free joint and is welded to `base_target`.  The base
controller (`core/g1.py::G1Controller`) moves the mocap body toward the
commanded planar pose (x, y, yaw) with **translational limit 0.5 m/s and
angular limit 1.0 rad/s**, at the **fixed standing height 0.79 m** with an
upright orientation; the target is never teleported (mocks use explicit
privileged `teleport_base`).  At reset the mocap pose equals the pelvis
pose, so there is no startup jump (verified: < 2 mm base motion in the
first 50 ms).  The 12 leg joints are position-held in the upstream `stand`
posture (all 0 rad).  Because the feet would otherwise drag on the floor
while the weld slides the pelvis, **foot–floor collision is disabled by
excluding only the two ankle-roll bodies from colliding with world geoms**
(the table is its own body, so feet/knees still collide with it; all
other robot/environment and self collisions remain enabled).

### Finger and lower-body constraints

Fixed finger posture, held by the upstream position actuators
(`core/g1.py::RIGHT_HAND_POSTURE`, mirrored for the left hand):

| Joint | Right | Left | Range (right) |
| --- | --- | --- | --- |
| thumb_0 (abduction) | −1.0 | +1.0 | ±1.047 |
| thumb_1 | −1.0 | +1.0 | −1.047 .. 0.724 |
| thumb_2 | −1.0 | +1.0 | −1.745 .. 0 |
| index_0 / index_1 | 0.35 / 0.35 | −0.35 / −0.35 | 0 .. 1.571 / 0 .. 1.745 |
| middle_0 / middle_1 | 0.35 / 0.35 | −0.35 / −0.35 | 0 .. 1.571 / 0 .. 1.745 |

Chosen by a search over 735 thumb postures (docs/DECISIONS.md §11.3): no
self-penetration, no penetration with a 6 cm object centred on the palm
reference point, and the thumb tip stays **1.4 cm behind** the palm point
along the approach axis (it does not hit the table on low grasps).  The
upstream `stand` keyframe thumb (−1.05) touched the palm and is not used.
Lower body: all leg joints at 0 rad (upstream `stand`).

### Actuators, gains and limits (upper body)

Upstream position actuators are kept unchanged (no competing actuators are
added): `kp = 500`, `dampratio = 1` (critically damped), `ctrlrange =
joint range` (`inheritrange`), joint-level force limits from the model
(`actuatorfrcrange`: shoulders/elbow/wrist-roll ±25 N·m, wrist pitch/yaw
±5 N·m, waist yaw ±88, waist roll/pitch ±50, fingers ±1.4–2.45 N·m).
Controller-side additions (`core/g1.py::G1Limits`):

* right-arm **Cartesian setpoint streaming** at 0.25 m/s with a 3 cm
  "pull-along" lead (the setpoint never runs ahead of the live hand) and
  warm-started IK every 2 physics steps;
* **joint-rate limit 2.5 rad/s** (synchronized across the arm) as a safety
  layer; waist 0.8 rad/s;
* **gravity/Coriolis feed-forward** on the seven right-arm DoFs
  (`qfrc_applied = qfrc_bias`), so the actuators only correct tracking
  error;
* the left arm and the waist hold their rest targets (waist pitch 0.30 rad
  forward lean, yaw/roll 0) unless explicitly commanded via
  `SimWorld.set_waist_target` (internal).

Rest postures: left arm = upstream `stand` with shoulder roll widened to
0.4 rad (hand clears the hip); right arm = raised "ready" posture
(shoulder pitch −0.8, roll −1.0, yaw −0.5, elbow 1.0, wrists 0) with the
palm point at ≈ (0.40, −0.46, 0.97) m in the base frame — above the table
plane, at the table's right edge, out of the head camera's view.  Small
(5 cm) motions from a settled reach settle in ≈ 0.4–0.7 s (measured:
grasp/lift/place task poses 0.48 / 0.70 / 0.66 s); long reaches from the
ready posture take ≈ 1–1.7 s (`scripts/ik_reach_test.py`).

### IK configuration and end-effector frame

* `core/ik.py::ChainIK`: damped least squares on the seven right-arm
  joints via `mj_jacSite`, damping λ = 0.05, position weight 1.0,
  orientation weight 0.3 (rotation-vector error of `R_target R_current^T`),
  ≤ 200 iterations, step bound 0.25 rad, joint limits with 0.01 rad
  margin and an active set for clamped joints, rest-posture bias
  (gain 0.02) projected into the exact task null space, stall detection
  (25 iterations).  Solver tolerance 1 mm / 0.5°.  The solver only touches
  a scratch `MjData`.
* IK search range override: shoulder pitch ≤ 0.8 rad (excludes the
  "arm behind the torso" family).
* **Right EE frame** `right_ee_site` in `right_wrist_yaw_link`:
  position (0.10, 0.065, 0) m — the palm reference point, in front of the
  palm face between the finger bases; axes: **+x = finger direction, +z =
  grasp approach axis = palm normal (link +y), +y = −link z**
  (`quat = (0.7071, −0.7071, 0, 0)`).  This one site is used for IK, for
  `RobotEnv.get_ee_pos`, and for the attachment-distance check
  (`ATTACH_RADIUS = 0.05 m`, unchanged object selection and opaque
  handles; the grasp welds anchor on `right_wrist_yaw_link`).
* Default orientation policy for position-only commands: approach axis
  pointing down, tilted 20° toward the fingers, fingers 30° left of the
  base heading (`core/g1.py::approach_rotation`); relaxed to position-only
  IK when infeasible.  Pose-constrained commands exist internally only
  (`SimWorld.set_arm_target(pos, rot)`).

### Cameras

MuJoCo camera frame: x right, y up, looks along −z.  Public optical frame:
+x right, +y down, +z forward (`T_world_camera = T_world_gl · diag(1, −1, −1)`).

| Camera | Mount body | Local position (m) | Local orientation | vfov |
| --- | --- | --- | --- | --- |
| `head` | `torso_link` (the G1 head is a fixed mesh on the torso; no neck joint) | (0.09, 0, 0.40) — just in front of the face (head mesh x ≤ 0.077) | forward (+x torso), pitched 30° down (`xyaxes="0 -1 0  0.5 0 0.866"`); with the 0.30 rad waist lean ≈ 47° below horizontal | 60° |
| `right_wrist` | `right_wrist_yaw_link` | (0.05, 0.01, 0.075) — bracket 7.5 cm to the index-finger side of the palm (palm mesh \|z\| ≤ 0.044) | optical axis aimed at the point 0.235 m ahead of the palm point along the approach axis (link +y) | 90° |
| `left_wrist` | `left_wrist_yaw_link` | (0.05, −0.01, 0.075) | mirror: aimed 0.235 m along link −y | 90° |

Occlusion (segmentation render, arm raised): right wrist view 7 % hand
pixels (4 % in the central half), palm point fully visible; head view 0 %
robot pixels at rest.  Clipping: MJCF `map znear="0.0025" zfar="5"` are
fractions of the explicit `statistic extent="2"`, giving **effective
near = 0.005 m, far = 10 m** (asserted in `tests/test_cam_sync.py`);
depth ≥ 0.98·far → NaN.  Wrist depth is valid down to ≈ 4 cm in front of
the lens.  `"onboard"` remains an accepted alias of `head`.

### Trial geometry (changed for the G1 workspace)

* Table: thin slab, **top at z = 0.85 m** (was 0.40), x ∈ [0.25, 0.65],
  y ∈ [−0.5, 0.5]; no legs so knees/feet can stand at the edge.
* Red region: centre (0.40, 0.30), half-extents 0.08 × 0.08, support
  0.85 m (was (0.65, 0.35) / 0.40 m).
* Floor: finite 5 × 5 m (background is visible beyond it).
* Smoke trial objects moved onto the new table with the same roles
  (stone (0.40, −0.15), cube (0.40, 0.15), bottle (0.55, −0.35); smoke_2
  stone (0.45, −0.05, yaw 0.8), cube (0.35, −0.35), bottle (0.55, −0.45);
  smoke_5 stones at (0.40, −0.15) and (0.50, −0.02), cube (0.35, 0.12)).
  Robot starts, SEARCH (yaw 3.0 rad → nothing visible) and clarification
  (two visible stones) scenarios and all expected outcomes are unchanged.
* Base parking for manipulation: `core/g1.py::ARM_WORKSPACE_OFFSET =
  (0.36, −0.20)` m in the base frame (`skills.approach`, TeleportExecutor).

### Known limitations

* Sliding humanoid: no locomotion, feet do not touch the floor (excluded),
  legs are position-held.
* Right arm only; the left arm is held.  Grasping is weld attachment
  within 5 cm of the palm point with fixed fingers — not contact grasping.
* Reachable band is narrow (≈ 0.28–0.45 m ahead, 0.08–0.32 m to the
  right of the base); objects deeper on the table must be approached from
  another edge.  The declared 5×5 grid has two unreachable corner points
  ((0.40, −0.32), (0.44, −0.32)).
* Straight-line hand paths are not collision-planned; `skills.reach` uses
  a 10 cm via-point above the target, and the reach test reports any
  penetration.
* The left wrist camera looks at the hip at rest (palm faces the body).
* MuJoCo 3.12.0 only is verified.
