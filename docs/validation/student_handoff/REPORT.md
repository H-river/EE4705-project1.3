# Student handoff validation — contract v2

Owner: backbone (ALL). 2026-09-07, Asia/Singapore.

This validates the three shared-infrastructure fixes described in
[the student README](../../STUDENT_README.md), not real Student A/B/C performance.
All checks ran offline using the project `.venv`, MuJoCo 3.12.0 and EGL.

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q -rs` | 121 passed, 1 skipped in 17.66 s; exit 0 |
| New `tests/test_student_handoff.py` cases | 23 cases included in the full suite |
| `.venv/bin/python -m eval.runner --mode e2e --trials eval/trials/smoke --mock-all` | 5/5 pass; all claimed and actual success; exit 0 |
| `git diff --check` for handoff code/docs | No whitespace errors |
| README JSON example | Parsed into Plan/Action and accepted by the real validator |

The one skip is the pre-existing experimental physical cube-grasp test;
the new handoff tests are not skipped. No live model calls or student
implementations were used. The earlier gripper physics benchmarks were not
rerun or relabeled by this change.
The archived upstream diff retains its original patch-context whitespace;
staging that historical file reports four whitespace notices. It was preserved
byte-for-byte instead of rewriting historical evidence.
Likewise the original Robotiq stdout/CSV artifacts retain trailing spaces
and CSV CRLF line endings; whitespace validation is scoped to source and
authored documentation, not normalization of raw experimental records.

The regressions cover a validated ID-only plan running through all mock
actions, transport clearance, malformed waypoints, unlocalized PLACE,
stopping an active arm/base/gripper motion without changing qpos/qvel/time,
resuming motion, preserving a held object, stopping on refusal and exposing
a controller stop exception. Verification cases reject floating or off-region
objects, attachment, bbox-only evidence, wrong instance IDs, stale frames,
invalid region dimensions, excess inter-frame drift, and an executor falsely
reporting a successful in-plan VERIFY. Exact lower/upper support-height bounds
also remain inclusive, matching the previous oracle comparisons.

Controller tests permit bounded physical braking (TCP excursion < 3 cm
over the measured stopping interval), rather than pretending a velocity reset
is a valid stop. The test also verifies a later reach can execute normally.

Raw outputs: [pytest.log](pytest.log), [smoke.log](smoke.log).
The original smoke run is `runs/20260907_000129_e2e_mockall/`.
Copies of its metadata and metrics are retained here as
[run_meta.json](run_meta.json) and [metrics.json](metrics.json).
Every trial uses contract_version 2; the old validation snapshot remains v1.
