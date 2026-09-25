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
