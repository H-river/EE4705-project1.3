# Offline ABC demo validation — 2026-09-07

This checks the fixed stone-to-red-area teaching demo. It does not report
course AI-model accuracy or randomized-scene success rates.

| Recorded episode | Visual claim | Independent actual result | Video length | Final simulation time |
| --- | --- | --- | --- | --- |
| `runs/abc_workflow/success` | PASS | PASS | 20.3 s | 10.71 s |
| `runs/abc_workflow/retry` | PASS | PASS | 22.1 s | 10.73 s |

Recording starts after a 1 s settling period. Video time includes display
holds; final simulation time includes the evaluator's 2 s stability check.
Both runs grasped the expected `stone`, placed it in `red_region`, released
it, and passed the independent stability check (20 samples over 2 s).
The retry run recorded `GRASP_MISSED` at attempt 1 and success at attempt 2.
The failure was explicitly injected before motion, not detected from contact.

Commands run:

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m demo.run --scenario success --out runs/abc_workflow/success
.venv/bin/python -m demo.run --scenario retry --out runs/abc_workflow/retry
.venv/bin/python -m pytest -q
```

Tests: **128 passed, 1 skipped**. The skip is the existing experimental
physical-grasp test. The default demo uses weld attachment.
The demo contributes 7 tests, including real simulated normal and retry
episodes, image-based localization, planning edge cases, saved evidence,
and the student-stub guard.

Both MP4 files decoded completely with `ffmpeg -v error -i ... -f null -`
(exit 0, no decoder errors). Browser inspection verified the rendered
video, jumping to B's plan, and jumping to the failed grasp with matching
error details. The browser uses a local Blob for seeking because Python's
basic HTTP server does not provide byte-range responses.

`--student B` was checked against the current unimplemented stub: it exits
2 with a clear explanation and does not substitute the demo planner.
A completed student AI module has not been tested; all student files
remain unchanged stubs.

JSON evidence SHA-256:

```text
success/episode.json  766e7293d85f2d490c67012b8b594ddac8d9452cfc2e9c14f84f2158feb6c18a
retry/episode.json    7eaa95234c7296d4a8152db8163503e7dd609757f7e2cafde2a20e77abed0c32
```

During development, one run placed the stone correctly but failed visual
verification because the object was clipped by the head camera. The demo
executor now returns to its initial observation viewpoint after release.
Both recorded runs above include that change and pass fresh visual checks.
This does not establish visibility or manipulation reliability in other
layouts; those need separate trials.
