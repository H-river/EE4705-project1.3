# Bonus: learned grasp policies

Results: [RESULTS.md](RESULTS.md) · write-up: [BONUS_SECTION.md](BONUS_SECTION.md) · figures: `figs/` · videos: `videos/`.

Trained checkpoints (ACT + bottle demos, Diffusion Policy) are on the Hugging Face Hub:
https://huggingface.co/jiamo0912/ee4705-learned-grasp. `EE4705_GRASP_CKPT` accepts an `hf://` path; it is
downloaded into `runs/hf/` on first use:

    EE4705_GRASP_POLICY=act EE4705_GRASP_CKPT=hf://jiamo0912/ee4705-learned-grasp/act_bottle_best \
    python -m eval.runner --mode manipulation --trials eval/trials/student_c_v2

Use `EE4705_GRASP_POLICY=diffusion` with `.../diffusion_best` for the Diffusion Policy. Training and evaluation
scripts are in `scripts/bonus/` (`train.sh`, `eval_grasp.py`, `run_stage6.sh`).
