# BONUS STATUS (updated 2026-09-25 22:49 +08)
Stage: 4/8 DP training 60k — at ~18k (3.6 it/s, ETA ~02:00)
Full executor, grasp success /30 (C1 C2 C3 C4):
  scripted 29 30 29 30 | ACT 29 26 0 30 | DP pending
  WRONG_OBJECT 0, undetected 0, false claims 0 everywhere so far
ACT: VAL 20/20 at 50k; fails on the OOD bottle (aims at stone/cube height)
DP VAL (20 eps): 5k 18, 10k 19, 15k 19
Paused: 5k-demo + bottle collection (RAM), resumes after DP
Next: DP stage 6 + DDIM/DDPM ablations → state-only → RESULTS.md
Blocked: nothing
