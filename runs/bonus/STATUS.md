# BONUS STATUS (updated 2026-09-25 21:35 +08)
Stage: 4/8 DP training 60k (~4.7 h, 3.6 it/s) + stage 6 for ACT in parallel
Done: S0 env, S1 demos (2,019, expert 99.5 %), S2 dataset, S3 ACT (VAL 20/20 at 50k)
Full executor C1 (30 eps): scripted 29/30, ACT 29/30 (both miss c1_11 before GRASP)
Scripted full executor: C2 30/30  C3 29/30  C4 30/30; 0 false claims
ACT C2–C4 + skill-level cells: running
DP: training (curve every 5k)
Skill v3: attach at 2.5 cm, settle stop only near target (both chosen on VAL)
Next: ACT C2–C4 → DP curve → DP stage 6 → ablations → RESULTS.md
Blocked: nothing
