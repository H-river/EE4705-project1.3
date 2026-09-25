# BONUS STATUS (updated 2026-09-26 02:49 +08)
Stage: 6/8 evaluation — main table done; state-only ablation training (~1 h)
Full executor, grasp success /30 (C1 C2 C3 C4):
  scripted 29 30 29 30 | ACT 29 26 0 30 | DP 29 27 0 30
  WRONG_OBJECT 0, undetected 0, false claims 0 for all policies
Both learned policies fail the OOD bottle (they aim at stone/cube height)
Ablations C1: ACT n_action_steps 10/25/50 and DP DDIM 5/10/50, DDPM 50 → all 29/30
Next: state-only (image ablation) → RESULTS.md → BONUS_SECTION.md → stage 8
Blocked: nothing
