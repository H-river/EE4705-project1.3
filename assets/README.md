# Assets

Owner: backbone (ALL)

## Model composition and provenance

Production `scene.xml` includes `g1_2f85_ee4705.xml` and
`scene_common.xml`. `scene_cam_sync.xml` uses the same robot/world and adds
only the diagnostic falling ball. There is no runtime Dex3 variant.

Both upstream sources are from MuJoCo Menagerie revision
**8161bba264d7fa7c99ca301e91e7fb44737676ad**, pinned in
`menagerie_revision.txt`:

| Source | Purpose | License |
| --- | --- | --- |
| `unitree_g1/g1.xml` | G1 without dexterous hands, 29 body joints | BSD-3-Clause, Unitree Robotics; `licenses/unitree_g1.txt` |
| `robotiq_2f85/2f85.xml` | One linked parallel gripper per wrist | BSD-2-Clause, ROS-Industrial; `licenses/robotiq_2f85.txt` |

Fetch both mesh/MJCF directories with `bash scripts/fetch_menagerie.sh`.
The checkout under `assets/menagerie/` is ignored by Git. Generate the local
composition with `python scripts/build_g1_model.py`; regenerate its diff
against no-hands G1 using `bash scripts/g1_upstream_diff.sh`. Upstream
sources are never modified. The generated XML and `g1_upstream_diff.patch`
are tracked; the generator records the two-source composition more precisely
than the G1-only diff. Immutable meshes/materials/defaults are shared under
`2f85_*`; instances use `rg_`/`lg_` names for bodies, joints, geoms, sites,
tendons, actuators and equality constraints. The two connect constraints and
one driver coupling per gripper remain active through reset, attach and detach.

The retired adaptation and its original upstream diff are in `archive/`.
See `archive/README.md` for restoration instructions. `reference/` contains
the still older Cartesian-gantry scene. Neither archive is loaded at runtime.

## Flange and TCP

Both mounting links are the actual `right_wrist_yaw_link` /
`left_wrist_yaw_link`. The wrist mesh ends at link x ≈ 0.042 m. The replacement
mount plate begins at x = 0.045 m, forward of the wrist rather than inside it.

| Transform | Translation, meters | Quaternion, wxyz |
| --- | --- | --- |
| Wrist link → `rg_base_mount` / `lg_base_mount` | (0.045, 0, 0) | (0.5, 0.5, 0.5, 0.5) |
| Mount → `rg_base` / `lg_base` (upstream unchanged) | (0, 0, 0.0038) | normalized (1, 0, 0, −1) |
| Gripper base → `rg_pinch` / `lg_pinch` (upstream unchanged) | (0, 0, 0.145) | identity |

Thus the effective TCP position is **(0.1938, 0, 0) m in the wrist link**.
The base/TCP orientation relative to that link is a +90° rotation about y:
TCP **+z is approach = wrist +x**, **+y is pad closing = wrist +y**, and
**+x is pad width = wrist −z**. The intermediate mount has a −90° upstream
base rotation; it must not be omitted when interpreting the closing axis.
The old palm point was (0.10, 0.065, 0) with approach along wrist +y.

`rg_pinch` is the single right-arm frame for IK, `get_ee_pos()`, state
reporting and attachment-distance checks. The weld parent is `rg_base`.
There is no object snap: the current relative pose is retained, so an object
near the TCP stays visually between the fingers during lifting.

## Platform and controls

The free pelvis is welded to mocap `base_target` at standing height 0.79 m.
This is a sliding-base approximation, not walking. Only the foot bodies are
excluded from floor contact. Base target limits remain 0.5 m/s and 1 rad/s;
`base_weld` uses solref `0.01 1` and solimp `0.99 0.999 0.001 0.5 2`.

Arm position actuators retain upstream kp 500, dampratio 1, joint control
ranges and force limits. Cartesian streaming is 0.25 m/s, maximum lead
0.03 m, with warm-started IK every two 2 ms physics steps. Synchronized
arm joint setpoints are limited to 2.5 rad/s, waist to 0.8 rad/s. Right-arm
bias-force compensation remains enabled. Live qpos is never overwritten
by ordinary control.

Ready right-arm joints (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
remain **(−0.8, −1.0, −0.5, 1.0, 0, 0, 0)**. Left-arm rest is
(0.2, 0.4, 0, 1.28, 0, 0, 0); waist pitch is 0.30 rad. Both grippers reset
open, and the left holds open until explicitly commanded. Measured ready
right TCP ≈ **(0.433, −0.564, 0.980) m**. Settling tests verify no persistent
oscillation, non-finite state, joint-limit violation or body/table penetration.

Each upstream gripper actuator is a `general` actuator with fixed gain and
affine position bias, driving its tendon; it is not converted to a motor.
Control range **0..255** means open..closed, with the upstream gain/bias and
force range **−5..5** retained. Public `set_gripper(side, opening)` reverses
that normalization: **0 closed, 1 open**. It only sets a target. The control
ramp is 637.5 control units/s, equivalent to 2 rad/s over the 0.8 rad driver
range, so the full setpoint ramp takes 0.4 s. Full physical strokes settle
within the declared **1.0 s** (opening tolerance 0.03, speed below 0.05 rad/s).
Measured opening is `clip(1 − mean(driver_angles)/0.8, 0, 1)`, not the target
and not a linear measurement in millimeters. `pad_gap()` separately measures
the opposing inner pad faces along TCP +y; nominal full stroke is 85 mm.

## IK and the declared workspace

`core/ik.py` is unchanged: DLS damping 0.05, position/rotation weights 1/0.3,
200 iterations, 0.25 rad step limit, 1 mm/0.5° solver tolerance, rest bias
0.02, joint-limit margin 0.01 rad. Shoulder pitch search remains −3..0.8 rad.
Explicit orientation commands remain pose-constrained; position-only public
commands first request the default approach and then relax orientation.

The new default approach is **45° tilt / 0° finger azimuth**, replacing
20°/30°. This is necessary because approach now follows the forearm axis.
A full unchanged-grid experiment using the old orientation with the new TCP
found 25/25 mathematical IK solutions but only 2/25 collision-free endpoints,
and all four task poses failed. The new orientation gives 21/25 IK, usable
and tracked points; all four task poses pass. The old palm baseline was
23/25. These are measured results, not a claim of complete workspace coverage.

The grid is unchanged: x 0.28..0.44 and y −0.32..−0.08 (5×5), TCP z 0.90 m,
base (0,0,0), waist (0,0,0.3). No trial positions, object dimensions or masses,
SEARCH scenarios, or clarification scenarios were changed. Table top is
0.85 m, footprint x [0.25,0.65], y [−0.5,0.5]; red region center (0.40,0.30),
half widths (0.08,0.08). The base parking offset remains (0.36,−0.20),
pre-reach height 0.10 m, grasp offset 0.02 m above the object's center.

The proposed endpoint collision guard rejects position-only solutions with
actual penetration, permitting the existing `move_to` adapter to re-park
the base. It does not reject upward orientation alone. This is an explicitly
identified contract exception; see `docs/validation/robotiq/REPORT.md`.

## Cameras

All RGB-D frames are 640×480. MuJoCo camera axes are x right/y up/−z forward;
public optical axes are x right/y down/z forward. Atomic capture semantics,
owned observations and camera aliases are unchanged.

| Camera | Mount | Local position, m | Local orientation | Vertical FOV |
| --- | --- | --- | --- | --- |
| `head` (`onboard` alias) | `torso_link` | (0.09,0,0.40) | xyaxes `0 -1 0  0.5 0 0.866` | 60° |
| `right_wrist` | `rg_base` | (−0.058,0,0.02) | xyaxes `0 -1 0  -0.9962 0 0.0872` | 90° |
| `left_wrist` | `lg_base` | (−0.058,0,0.02) | same | 90° |

The wrist lens sits beyond the base mesh's x half-extent 0.0375 m and looks
along +z, tilted 5° toward the approach axis. Finger tips are deliberately
visible at the bottom in both open and closed states. The central 20% has
**zero gripper pixels**; ready median depths are about 2.830 m right and
0.772 m left, verified with segmentation plus depth. Both wrists pass
non-central pixel unprojection onto known table/floor planes. Head is unchanged.

Scene extent 2 m × map znear 0.0025 / zfar 5 gives effective clipping
**0.005 m / 10 m**. Near/far and projection/unprojection are tested. Sync
validation gives 125 batches / 375 observations, zero capture failures,
zero timestamp difference and zero physical-state change across capture.

Ready grippers are outside the unchanged head camera field of view. The
report includes the actual head-ready image, overhead-ready evidence, and a
separate head image with the right arm reaching onto the table; it does not
mislabel that reach image as ready. Wrist views are available for future
alignment; existing mocks use head for grounding/SEARCH/verification.

## Contacts and grasp modes

Default **`SimWorld(grasp_mode="weld")`** remains the smoke/TeleportExecutor
path. `RobotEnv` wraps that world; it does not silently switch modes.
`ATTACH_RADIUS` remains 0.05 m, maximum one attachment, nearest free active
body chosen by (distance, body id). Return is the existing opaque string
handle or `None` (there was no `AttachmentResult` class in this repository).
Detach releases the object; reset restores all contact masks.

Bit 1 identifies non-gripper physical geoms (affinity 3); bit 2 identifies
grippers (affinity 2). This preserves gripper-vs-table/object, mutual gripper
and gripper-internal contact eligibility. Explicit upstream internal
exclusions remain, plus mount/base/driver/spring-link exclusions against the
same arm's elbow and wrist links. While welded, ONLY the held object's
conaffinity becomes 1: gripper closing is cosmetic, while table and other
object contacts remain active. Detach/reset restores the original affinity.
Visual geometry still has contype/conaffinity zero.

Opt-in **`grasp_mode="physical"` is experimental**. No grasp weld is created.
Close/check lasts at most 1.5 s, ending at the target or after 50 consecutive
steps below 0.05 rad/s driver speed (following the initial 0.05 s). Success
requires both pads on the same candidate, a nonempty opening and the object's
center inside the measured gap (5 mm closing-axis allowance, 5 cm transverse
TCP radius). `detach()` opens the right gripper. Hold queries detect lost
bilateral contact and clear the opaque handle with reason `slipped`.

Pad-only changes from upstream: friction **1.0 0.02 0.002**, condim **4**,
solref **0.004 1**, solimp **0.95 0.99 0.001**, margin **0.0003 m**. Contact
recognition threshold is 0.0006 m. The initial 1 mm margin was reduced because
opposing pads stopped short of the declared closed target. Scene solver
settings remain the original implicitfast/default cone/impratio; the interrupted
implementation's scene-wide elliptic/impratio change was removed. All object
mass, geometry, friction and solver settings remain at their original values.

Final 10-attempt physical evaluation: **stone 0/10, cube 7/10, bottle 1/10**.
Stone slips were detected at the 6 cm commanded lift checkpoint; cube failures
were two single-pad contacts and one closed-on-nothing; bottle had nine reach
timeouts. Tuning stopped per the bounded-effort requirement. Results, measured
object displacement at slip detection, seeded offsets, CSV and representative
failure frames are in `docs/validation/robotiq/`. Robust physical grasping,
continuous slip-onset height and broader layouts remain unverified.
