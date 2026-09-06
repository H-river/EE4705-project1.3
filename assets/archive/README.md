# Archived assets

Owner: backbone (ALL)

## `g1_with_hands_ee4705.xml` (retired 2026-09-06)

The Unitree G1 **Dex3 hands** adaptation used by the first G1 integration
(mocap base, palm-point EE sites, fixed semi-closed finger posture, wrist
cameras on brackets beside the palms).  Retired in favour of Robotiq 2F-85
grippers (`assets/g1_2f85_ee4705.xml`) because weld-only "grasping" with
fixed fingers gave no path to physical grasping, the thumb protruded into
low grasps, and the palm point was a poor tool centre for a parallel-jaw
style pick.  `g1_with_hands_upstream_diff.patch` lists its changes relative
to upstream `unitree_g1/g1_with_hands.xml` (same pinned revision).

Not selectable at runtime.  To restore: copy the XML back to `assets/`,
point `scene.xml` at it, restore `RIGHT_HAND_POSTURE` / `LEFT_HAND_POSTURE`
and the palm-point EE constants in `core/g1.py` from git history
(commit `9c4f670` and earlier), and re-run the tests — the controller
interface did not change.
