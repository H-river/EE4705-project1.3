# BONUS STATUS (updated 2026-09-25 20:05 +08)
Stage: 2/8 dataset conversion (~40 min), then S3 ACT + S4 DP training
Done: S0 env; S1 2,019 demos (expert 99.5 %); skill + tests + eval harness
Scripted (skill-level, 30 eps): C1 30/30  C2 29/30  C3 27/30  C4 30/30
Best so far: ACT —, DP —
Next: ACT 50k + DP 60k steps, 20-episode val every 5k
Blocked: nothing (note: OOM at 20 sim workers → keep ≤ 12)
