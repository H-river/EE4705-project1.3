**Final e2e run (2026-09-24, branch `e2e`, tag `final-40`):** 40/50 correct, 0 false claims. The target was 43, so we missed by 3. Manipulation 37/47 claimed∧achieved; refuse/clarify 3/3. RUN 2 with two C fixes scored 38/50, so both were reverted. Details: `docs/night_run/REPORT.md` §13.
**Contract** stays v4: PlanStatus.REJECTED → replan; TrackingHint as in v3.
**Memory is now B's** (`planner/memory.py`). A reports only the current frame (no `memory_*` attributes, no `recall()`). C keeps no memory across actions. B re-plans from a remembered goal region when the region is out of view, and never from a remembered object.
**New tooling:** `eval/trials/final50` (50 trials with `expected.outcome`). `python -m eval.runner … --jobs 4 --no-cache` runs all 50 in about 17 min; `scripts/final_run.sh N` wraps it.
**Open, A:** a bottle clipped at the image top is rejected ("clipped object centre is below the table top", f34). After a release, the exact bottle/region instance is often not seen, so PLACE_FAILED although the bottle is in the region (f14, f31).
**Open, B:** "red stone" vs A's colour `dark_red` should map through the vocabulary (f19, f20).
**Open, C:**
- Parking beside a bottle at the front-right table edge takes it out of the head view, so GRASP → TARGET_LOST (f25, f41, f45). Candidate fix `70c6bb1` helped f25 but is untested alone.
- Far-edge reach limit (x ≈ 0.56, f38).
- Torso–table contact while carrying (f23). The standoff fix `79c70fe` did not fix it and broke bottle placement.
Every change to A's or C's code is in `docs/night_run/CHANGES.md` (#23–#34); revert any you disagree with.
