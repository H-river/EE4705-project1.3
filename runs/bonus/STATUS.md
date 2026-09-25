# BONUS STATUS (updated 2026-09-26 00:49 +08)
Stage: 4/8 DP training 60k — at ~44k (ETA ~02:10)
Full executor, grasp success /30 (C1 C2 C3 C4):
  scripted 29 30 29 30 | ACT 29 26 0 30 | DP pending
  WRONG_OBJECT 0, undetected 0, false claims 0 everywhere so far
ACT: VAL 20/20 at 50k; fails on the OOD bottle (aims at stone/cube height)
DP VAL (20 eps, seeded): 5k 18, 10k–30k 19, 35k 20, 40k 19 → no retries
Fixed tonight: hung watcher (GPU OOM in pool init) → CPU evals, seeded DP sampling
Next: DP stage 6 + DDIM/DDPM ablations → state-only → RESULTS.md
Blocked: nothing
