# Assets

Owner: backbone (ALL)

## Simplified reference platform (`scene.xml`)

Self-contained scene used by all backbone tests and smoke trials.  No
external downloads required.  Contents:

- simplified humanoid-style upper body: planar mobile base (slide x/y +
  yaw hinge), torso, head with onboard RGB-D camera, and a Cartesian
  3-slide arm with a visible end-effector site (`ee_site`);
- table (top at z = 0.40 m), gray stone, blue cube, green bottle, and a
  second dark-red stone (`stone2`) whose placement/visibility is controlled
  entirely by the trial configuration;
- finite red target region: thin visual-only box at center (0.65, 0.35),
  half-extents 0.08 × 0.08 m, support height 0.40 m (explicit bounds — it
  is NOT an infinite plane);
- optional overhead debug camera (`overhead`).

Kinematic/physical limitations are documented in the top-level README
("Reference platform limitations").

## Object vocabulary (`objects.yaml`)

Canonical names, instruction synonyms, default attributes, and geometric
size metadata shared by grounding (A) and planning (B).  `size_xyz` is the
FULL axis-aligned bounding-box extent (meters) in the object's rest
orientation; for non-box shapes it is the AABB of the shape (ellipsoid:
twice the semi-axes; z-aligned cylinder: diameter, diameter, height), not
the shape's native parameters.

## Unitree G1 (MuJoCo Menagerie) — acquisition path

The backbone does NOT implement or test G1 control; the simplified platform
above is the reference test platform.  To obtain the G1 model reproducibly:

- Upstream: <https://github.com/google-deepmind/mujoco_menagerie>,
  directory `unitree_g1` (MJCF converted from Unitree's official
  description).
- License: the `unitree_g1` model directory carries a BSD-3-Clause license
  from Unitree Robotics; the Menagerie repository itself is Apache-2.0.
  Check `assets/menagerie/unitree_g1/LICENSE` after checkout.
- As a submodule (once this project is a git repository):

  ```bash
  git submodule add https://github.com/google-deepmind/mujoco_menagerie assets/menagerie
  git -C assets/menagerie rev-parse HEAD   # the submodule pins the revision
  ```

- Or as a plain clone with a recorded revision:

  ```bash
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie assets/menagerie
  git -C assets/menagerie rev-parse HEAD > assets/menagerie_revision.txt
  ```

Model file: `assets/menagerie/unitree_g1/g1.xml` (plus `scene.xml` variants
in the same directory).

**Actual support status:** loading the G1 MJCF is expected to work with the
tested MuJoCo version, but G1 hand control, locomotion, and IK are NOT
implemented, NOT tested, and NOT claimed by this backbone.  Any future G1
integration must keep model-specific joint/actuator names inside the
environment adapter/configuration (`core/world.py` + a G1 scene XML), never
in module code.
