# Student B Qwen interface: offline validation

Date: 2026-09-07. Working directory: `/home/jiamo/EE4705/project1.3`.
User requested the interface and offline tests without Qwen credentials.
No real Qwen request was made. The model identifier in live configuration is
`qwen3.7-plus-2026-05-26`; offline runs explicitly use `offline-fixture-NOT-Qwen`.

## Results

| Check | Result | Scope |
| --- | --- | --- |
| B/client/validation/architecture tests | 63 passed in 3.80 s | Fake responses, with B live transport blocked |
| Full project suite | 165 passed, 1 skipped in 27.15 s | Includes existing backbone regression tests |
| Fixed-response B + demo RGB-D A + demo motion C | Passed | MuJoCo fixed scene; correct stone, red region, release, visual claim and independent stability check |
| Offline CLI | Exit 0, READY | Outputs written to `runs/qwen_b_offline/` |
| Real Qwen authentication, endpoint/schema support | Not tested | No credentials or API requests |
| Qwen language accuracy / randomized execution | Not tested | Canned responses are not model predictions |

The skipped test is `tests/test_gripper.py:122`: physical grasp mode remains
experimental. The integration check uses the existing demo C and weld grasp.
No robot hardware was used.

## Reproduce

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m pytest -q -rs
.venv/bin/python -m planner.run --offline-demo --out runs/qwen_b_offline_repeat
```

Use a new/empty output folder. The original full-suite output is retained in
`runs/qwen_b_offline/pytest.txt`. `input.json`, `scene.json`, `response.json`,
`schema.json`, `plan.json`, `diagnostics.json` and the per-call audit show the
offline interface input, model-shaped output and compiled result.

The new checks include invalid JSON, truncation, illegal extra fields,
nonfinite numbers, unknown IDs, wrong roles/colors, wrong skill order,
repeated pickup cycles, bounded repair, wrong held object, frozen goal,
occluded held/released identity, fresh position binding, service errors,
search/clarification, reset, explicit configuration and cache provenance.

Structural validity still cannot prove that the first model answer chose the
object intended by natural language. It also cannot establish IK feasibility
or actual grasp/placement success. See [the B guide](../STUDENT_B_README.md)
for the separate language and execution evaluations still required.
