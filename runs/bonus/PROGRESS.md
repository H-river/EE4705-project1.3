# Bonus: learned grasp (ACT / Diffusion Policy) — progress log

Branch `learned-grasp` (from e2e 0a2e871). 0 live API calls throughout.

## Stage 0 — environment (done 2026-09-25 ~19:50 +08)
- `pip install lerobot==0.6.1 h5py` into .venv: torch 2.11.0+cu130, CUDA OK (RTX 4080 laptop 12 GB),
  ACTConfig + DiffusionConfig instantiate (`scripts/bonus/check_env.py`).
- Side effects: numpy 2.5.2 -> 2.2.6 (lerobot pin); mujoco 3.12.0 unchanged.
- draccus 0.11.6 installs a stray top-level `tests` package into site-packages that shadows the
  repo's `tests/` (12 collection errors). Deleted `.venv/.../site-packages/tests`; full suite then
  418 passed, 1 skipped. If the venv is rebuilt, delete it again.
- Additive env API (no change to core/types.py, core/interfaces.py, skills.py):
  `RobotEnv.get_arm_q()`, `RobotEnv.set_arm_joint_target(q)` (-> SimWorld -> G1Controller.set_arm_joint_target).
- `.gitignore`: runs/ stays ignored except runs/bonus/*.md|csv|json (STATUS/PROGRESS readable on GitHub).
- Also installed `lerobot[dataset,training,diffusers-dep]==0.6.1` (datasets, diffusers 0.39, torchcodec, av).
  The stray `tests` package came back with it and was deleted again.

## Stage 1 — expert data (done 2026-09-25 19:58)
- `scripts/bonus/collect_grasp_demos.py --n-keep 2000 --workers 10 --seed 0` -> runs/bonus/demos/ep_*.h5
  (episode i uses rng([0, i]); resumable). Post-APPROACH start state = reset + 0.5 s settle + arm tuck
  (C's _ARM_TUCK_OFFSET) + reference APPROACH to approach_base_pose ±3 cm/±5° + open gripper.
- Result (runs/bonus/demos_manifest.json): 2,030 tried, 2,019 kept, expert success 99.5 %
  (8 TIMEOUT, 3 GRASP_MISSED, 0 wrong object); stone 1,037 / cube 982; 1,003 with a distractor;
  length mean 31.9 ± 9.0 steps (min 22, p50 28, max 129) at 10 Hz. ~1.1 MB/episode.
- Head camera: the target sits near the lower-right image edge at the post-APPROACH pose and the
  hand is mostly out of view, so the image carries little signal; the state carries target_pos.
- Incident: the first run hung at 1,708 episodes. The OOM killer took pool workers while a
  6-worker eval ran alongside the 14 collection workers (~540 MB RSS per MuJoCo worker, 14 GB RAM).
  Resumed with 10 workers. RULE: keep total sim workers ≤ ~12 across concurrent jobs.

## Stage 5 tooling (written early, used for the checkpoint curves)
- `executor/learned_grasp.py`: LearnedGraspSkill(policy, ckpt)(env, pos_world) -> SkillResult. 10 Hz,
  ≤ 15 s; ends when the TCP is within skills.EE_POS_TOL of the scripted grasp point (pos + 2 cm), when
  the commanded q settles (Δ < 0.01 rad for 5 ticks after ≥ 1 s), or at the limit; then ONE
  try_attach_near_ee() + step(50), as skills.grasp. Selected by EE4705_GRASP_POLICY
  (scripted|act|diffusion|mlp) + EE4705_GRASP_CKPT (default runs/bonus/best/<policy>). Both executor
  call sites (student_c.py, closed_loop.py) now call grasp_primitive(); default = skills.grasp.
- tests/test_learned_grasp.py (6 tests, stub policies on real MuJoCo, incl. through StudentCExecutor).
- `scripts/bonus/eval_grasp.py`: skill-level closed-loop eval, cells C1–C4 + VAL (seed 500).
  C4 = distractor always present, surface gap 0.3–3 cm between bounding circles.
- Scripted baseline (skill level, 30 eps each): C1 30/30, C2 29/30, C3 27/30, C4 30/30, 0 wrong object.

## Stage 2 — dataset
- `scripts/bonus/to_lerobot.py`: lerobot 0.6.1 only writes codebase v3.0 (not v2); features as specified
  (observation.image 224², observation.state 10-D, action 7-D), images stored as PNG (no video).
  Split 90/10 by episode (seeded permutation, split.json). `scripts/bonus/check_batch.py` loads a
  batch with a 50-step action chunk (+ action_is_pad).
