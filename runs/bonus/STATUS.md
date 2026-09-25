# BONUS STATUS (updated 2026-09-26 03:17 +08)
Stage: 6/8 — last ablation (state-only DP, ~50 min); RESULTS/BONUS_SECTION drafted
Full executor, grasp success /30 (C1 C2 C3 C4):
  scripted 29 30 29 30 | ACT 29 26 0 30 | DP 29 27 0 30
  WRONG_OBJECT 0, undetected 0, false claims 0 for all policies
Image ablation: ACT state-only VAL ≤ 16/20, skill C1 26/30 (image 30/30) → image helps
Ablations C1: ACT n_action_steps 10/25/50 and DP samplers → all 29/30 (full)
Also running: collection to 5,000 demos + 500 bottle demos (for stage 8)
Next: DP state-only → finalize + push stage 7 → 8.1 data-size curve (ACT)
Blocked: nothing
