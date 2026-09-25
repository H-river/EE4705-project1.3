# BONUS STATUS (updated 2026-09-25 21:59 +08)
Stage: 4/8 DP training 60k (resumed after an OOM at 5k; ETA ~02:00) — ACT stage 6 done
Full executor, grasp success /30 (C1 C2 C3 C4):
  scripted 29 30 29 30 | ACT 29 26 0 30 | DP pending
  WRONG_OBJECT 0, undetected 0, false claims 0 everywhere so far
ACT: VAL 20/20 at 50k; fails on the OOD bottle (aims at stone/cube height)
DP: VAL 18/20 at 5k
Next: DP curve → DP stage 6 + DDIM/DDPM ablations → state-only → RESULTS.md
Blocked: nothing (rule: ≤1 extra process while DP trains, RAM)
