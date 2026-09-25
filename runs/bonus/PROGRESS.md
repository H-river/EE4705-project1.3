# Bonus: learned grasp (ACT / Diffusion Policy) — progress log

Branch `learned-grasp` (from e2e 0a2e871). 0 live API calls throughout.

## Stage 0 — environment (done 2026-09-25 ~19:40 +08)
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

## Stage 1 — expert data (done 2026-09-25 ~19:48)
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
- Dataset built (2026-09-25 ~20:05): runs/bonus/lerobot/grasp_2k = first 2,000 kept episodes, 1,800 train /
  200 val episodes, 57,402 train frames, 1.9 GB. check_batch OK.

## Stage 3 — ACT (started ~20:10)
- lerobot-train was data-bound: 3 steps/s at batch 32 (data_s 0.29 s vs update 0.07 s) → ~4.6 h per 50k.
  Aborted at step ~1,300 (no checkpoint yet) and replaced with `scripts/bonus/train_policy.py`: same
  make_policy / presets / processors / ImageNet image stats / delta indices + pad masks, but batches come from
  a uint8 memmap of the SAME episodes (runs/bonus/cache/grasp_2k, 9.6 GB; `--verify`: 30 random frames
  identical to LeRobotDataset). 12.9 steps/s, GPU 99 %. Loss at step 500: 3.731 (lerobot-train: 3.729).
- `scripts/bonus/train.sh <name> <kind> <steps> <dataset> [args]` = trainer + watch_ckpts (20 VAL eps / 5k).
- act_base: chunk 50, n_action_steps 25, resnet18 (ImageNet), batch 32, lr 1e-5 (lerobot default), 50k, seed 1000.
- ACT 5k with the v1 skill trigger (attach when TCP within 1.2 cm = skills.EE_POS_TOL): 8/20. 10 of 12 misses came
  within 1.3–2 cm of the grasp point, then the policy lifted away (the demos descend, pause ~0.1 s, lift).
  → LearnedGraspSkill v2 trigger: ATTACH_TOL 2.5 cm from the grasp point, checked every 20 ms (TCP then ≤ 4.5 cm
  from the object centre, inside ATTACH_RADIUS 5 cm). Post-conditions unchanged; scripted unaffected.
  Same 5k checkpoint re-scored: 16/20. All curve points from here on use v2 (v1 detail kept as 005000_v1trigger.jsonl).
- ACT VAL curve under v2: 5k 16, 10k 16, 15k 17, 20k 16, 25k 16 (/20). ≥ 30 % at 25k → no ACT retries needed.
- The same VAL episodes (2, 4, 14) failed at 15k and 25k with stop="settled" 19–22 cm from the target after
  1.2–1.7 s. With the settle stop disabled, all three attached (the policy resumes after a 1–2 s pause).
  → v3: the settle stop only ends the rollout within SETTLE_NEAR = 5 cm of the grasp point; a pause farther away
  runs to the 15 s limit. Tests updated (7 pass). All ACT checkpoints re-scored under v3 (v2 details in
  runs/bonus/eval/curves/act_base/v2trigger/, v2 curve in runs/bonus/curves/act_base_v2trigger.csv).
- ACT (v3 skill) VAL curve: 5k 18, 10k 18, 15k 20, 20k 19, 25k 19, 30k 19, 35k 18, 40k 19, 45k 20, 50k 20 (/20).
  Selected: 50k (max VAL, tie → latest step). runs/bonus/best/act → train/act_base/checkpoints/050000.
  Figure docs/bonus/figs/act_curve.png; data docs/bonus/data/act_base.csv. Stage 3 DONE (21:35).

## Stage 4 — Diffusion Policy (started ~21:15)
- A first DP launch ran concurrently with ACT: 1.7 it/s, RAM 3 GB free → stopped after ~1k steps (no checkpoint),
  relaunched alone after ACT: 3.6 it/s (GPU-bound: 2 obs × 224² images × batch 64) → ~4.7 h for 60k.
- diffusion_base: horizon 32, n_action_steps 8, n_obs_steps 2, DDIM 10 inference steps (100 train timesteps),
  resnet18 (ImageNet), batch 64, lr 1e-4 (lerobot preset), 60k, seed 1000.

## Stage 6 (started early for ACT while DP trains; JOBS=2 for RAM)
- EE4705_GRASP_LOG sidecar added (the executor replaces the primitive's info after its lift check, so the
  learned-skill stop/ticks were not visible in trial records). A bogus EE4705_GRASP_CKPT makes the trial fail →
  job subprocesses do use the learned policy.
- ACT C1 (full executor): 29/30, 0 false claims. The one failure (c1_11) is LIMIT_EXCEEDED before any GRASP —
  identical for scripted, not a grasp failure.
- 21:38 OOM killed the DP trainer (ACT stage-6 jobs + ACT skill evals + DP watcher eval + DP training at once).
  The 5k checkpoint had been saved (VAL 18/20); resumed from it at ~21:45 (loss continuous). RULE: while DP
  trains, at most ONE extra light process (JOBS=1 / workers=1).
- ACT full executor: C1 29/30, C2 26/30 first-grasp (28/30 after the executor's retry/replan: c2_03, c2_16 missed
  first, post-condition caught it, second attempt held), C3 0/30, C4 30/30; WRONG_OBJECT 0, undetected 0, false claims 0.
- ACT skill level: C1 30/30 (3.18 s), C2 26/30, C3 0/30, C4 30/30 (2.83 s).
- ACT C3 analysis: target z was constant in training (stone/cube centre 0.875); the bottle centre is 3.5 cm higher,
  the policy aims too low (most min distances 5–10 cm); 12/234 calls reached 2.5 cm but did not attach (the
  hand had displaced the tall bottle). Genuine OOD failure; see 8.2.
- ACT ablation C1 (n_action_steps): 10 → full 29/30, skill 27/30 (3.23 s); 25 → 29/30, 30/30 (3.18 s);
  50 → 29/30, 29/30 (2.71 s).
- Videos (existing Recorder via eval.runner --video): docs/bonus/videos/act_{success_c1_00,failure_c2_04,ood_c3_00}.mp4
- 22:25 paused 5k/bottle collection (RAM: DP eval workers ~2.5 GB each); 5k log at ~2,700 tried; resume after DP.
- 23:22 DP watcher hung at 20k: the trainer grew to 8.7 GB GPU; eval workers hit CUDA OOM in the Pool
  initializer, and multiprocessing.Pool respawns failing workers forever (map never returns). Fixes:
  (1) eval_grasp records init errors and raises them from the first episode (fail fast, no hang);
  (2) checkpoint evals run on CPU while training (--device cpu; config loaded with device override; ~4 min / 20 eps);
  (3) DP sampling noise came from torch's unseeded global RNG → evaluations were not reproducible. LeRobotPolicy now
  re-seeds torch per rollout (eval: seed*10000+i per episode; full executor: per-process call counter, each trial is a
  fresh process). ACT is deterministic at inference (VAE latent = 0), so its results are unaffected.
  DP 5k/10k/15k (GPU, unseeded: 18/19/19) are re-scored seeded on CPU; old files in
  runs/bonus/eval/curves/diffusion_base/gpu_unseeded/.
- DP VAL (seeded, CPU eval): 5k 18, 10k 19, 15k 19, 20k 19, 25k 19, 30k 19, 35k 20, 40k 19, 45k 19, 50k 19, 55k 19, 60k 19.
  Selected 35k (only 20/20). runs/bonus/best/diffusion → train/diffusion_base/checkpoints/035000. ≥ 30 % at 30k → no
  retries. Training finished 02:00 (60k steps, ~4.5 h at 3.6 it/s). Stage 4 DONE.
- Figure docs/bonus/figs/curves_act_dp.png; data docs/bonus/data/diffusion_base.csv.

## Stage 6 — DP (02:10–02:50)
- DP full executor: C1 29/30, C2 27/30, C3 0/30, C4 30/30; WRONG_OBJECT 0, undetected 0, false claims 0.
  C2 misses are the same far layouts as ACT (c2_04, c2_20; c2_16). C3: 415 grasp calls (retries), none attached.
- DP skill level: C1 30/30 (3.20 s), C2 25/30, C3 1/30, C4 27/30 (full executor recovers C4 by retrying).
- DP sampler ablation C1 (full / skill): DDIM 5 29/30 · 30/30; DDIM 10 29/30 · 30/30; DDIM 50 29/30 · 30/30;
  DDPM 50 29/30 · 30/30 (verified the scheduler class and step count really change).
  NOTE: a first attempt of these three ran with broken kwargs (a ':'-split in the shell loop cut the JSON) → 0/30,
  invalid; moved to runs/bonus/invalid/ and re-run.
- ACT n_action_steps full-executor numbers (JOBS=1 layout) now read too: 10 → 29/30, 50 → 29/30.
- Videos: docs/bonus/videos/diffusion_{success_c1_00,failure_c2_20,ood_c3_00}.mp4.
- State-only variants: lerobot ACT/DP require an image or observation.environment_state, so state-only = q (7) as
  observation.state + target_pos (3) as observation.environment_state (sliced stats). act_state 25k, diffusion_state 30k
  (retry-protocol lengths) training from 02:55.
- act_state (q → state, target → environment_state, no image) VAL: 5k 15, 10k 12, 15k 13, 20k 14, 25k 16 (/20).
  Selected 25k. C1: full executor 29/30 (retries recover), skill level 26/30 (2.23 s) vs image ACT 30/30 (3.18 s).
  → the head image DOES help ACT despite the target at the frame edge (earlier assumption wrong).
- diffusion_state training 30k at 7.2 it/s (~70 min) from ~03:05.
- Drafts: docs/bonus/RESULTS.md (tables via make_results.py --write) and docs/bonus/BONUS_SECTION.md.
- diffusion_state VAL: 5k 14, 10k 18, 15k–30k 20 (/20). Selected 30k. C1: full 29/30, skill 30/30 (2.68 s).
  → DP does not need the image; ACT does (state-only 26/30 skill vs 30/30).
- Stage 8 data prepared while waiting: runs/bonus/demos now 5,005 kept (5,033 tried, 99.4 %); bottle demos
  runs/bonus/demos_bottle 503 kept of 583 (86.3 %; 38 UNREACHABLE, 23 GRASP_MISSED, 19 TIMEOUT). The bottle run first
  crashed (a 10 cm lift after attaching a bottle had no collision-free position-only IK → ValueError killed the
  pool); fixed: lift falls back to 5 cm, and any per-episode exception is logged as a failed episode.
- Conversions for 8.1 (grasp_5k, grasp_1k, grasp_500 = first N episodes, nested) running in the background.

## Stage 6/7 DONE (04:18)
- docs/bonus/RESULTS.md (tables regenerated by make_results.py --write), figures in docs/bonus/figs/, videos in
  docs/bonus/videos/, docs/bonus/BONUS_SECTION.md (~700 words + table). Full test suite 425 passed, 1 skipped.

## Stage 8
### 8.4 final50, manipulation mode (done 04:45; done early, it is cheap)
- scripted 48/50 (45/47 manip), ACT 38/50 (35/47), DP 38/50 (35/47); false claims 0 for all.
- Learned-only losses: 9 bottle trials + f20 (stone2 at (0.60, 0.20), ~20 cm outside the training range).
  f41 (bottle) and f43 (REFUSED paraphrase) fail for all three. Will be rerun after 8.2 (bottle demos).
