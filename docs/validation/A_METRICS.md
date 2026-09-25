# Student A metrics
 
Source: `eval.grounding`, `eval.a_scene_description`, `eval.a_bbox_precision`,
`eval.a_spatial`, and `eval.a_vqa_check` runs against the live Qwen-VL API
(`response_source: live` in every case's audit; no fixture/cache results are
included below).
 
## Target Grounding Accuracy — original benchmark (10 unique / 7 missing / 3 ambiguous, 20 cases)
 
| Prompt/code version | cases | correct | accuracy | unique | missing | ambiguous | error cases |
|---|---|---|---|---|---|---|---|
| Baseline (original `SYSTEM`, no ambiguity rules) | 20 | 17 | 85.0% | 10/10 | 6/7 | 1/3 | 0 |
| + ordered ambiguity-selection rules (1–3) | 20 | 19 | 95.0% | 10/10 | 6/7 | 3/3 | 1 |
 
**Validator/prompt ablation**: the single remaining failure at 95% (`a_16`)
was a hallucinated full-frame "gray stone" box for a scene with no stone
present at all — caught by geometry (`box touches image boundary` →
`UNLOCALIZED`) but not prevented at detection time.
 
## Target Grounding Accuracy — balanced benchmark (10 unique / 10 missing / 10 ambiguous, 30 cases)
 
The original 20-case split under-tested missing/ambiguous cases (3–7 samples
each); `capture_dataset(per_kind=10)` was introduced to give each category
equal statistical weight.
 
| Prompt/code version | run | cases | correct | accuracy | unique | missing | ambiguous | error cases |
|---|---|---|---|---|---|---|---|---|
| Reframe not yet applied | 1 | 30 | 23 | 76.7% | 10/10 | 5/10 | 8/10 | 3 |
| Grounding prompt reframed ("identify which one...") | 1 | 30 | 26 | 86.7% | 10/10 | 8/10 | 8/10 | 0 |
| Grounding prompt reframed | 2 (repeat, same dataset) | 30 | 25 | 83.3% | 10/10 | 7/10 | 8/10 | 0 |
| Grounding prompt reframed | 3 (repeat, same dataset) | 30 | 26 | 86.7% | 10/10 | 9/10 | 7/10 | 0 |
| + duplicate-instance + identical-bbox guards | 1 | 30 | 25 | 83.3% | 10/10 | 6/10 | 9/10 | 0 |
| + duplicate-instance + identical-bbox guards | 2 (repeat, same dataset) | 30 | 26 | 86.7% | 10/10 | 7/10 | 9/10 | 0 |
| + duplicate-instance + identical-bbox guards | 3 (repeat, same dataset) | 30 | 26 | 86.7% | 10/10 | 8/10 | 8/10 | 0 |
| + duplicate-instance + identical-bbox guards | 4 (repeat, same dataset) | 30 | 25 | 83.3% | 10/10 | 7/10 | 8/10 | 0 |
| + duplicate-instance + identical-bbox guards | 5 (repeat, same dataset) | 30 | 26 | 86.7% | 10/10 | 8/10 | 8/10 | 0 |
| + duplicate-instance + identical-bbox guards | 6 (repeat, same dataset) | 30 | 26 | 86.7% | 10/10 | 8/10 | 8/10 | 0 |
| + duplicate-instance + identical-bbox guards | 7 (repeat, same dataset) | 30 | 27 | 90.0% | 10/10 | 9/10 | 8/10 | 0 |
| Current shipped code (guards confirmed reverted; `vision_contract.py` byte-identical to the reframe-only version) | 8 (repeat, same dataset) | 30 | 27 | 90.0% | 10/10 | 9/10 | 8/10 | 0 |
| Current shipped code | 9 (repeat, same dataset) | 30 | 26 | 86.7% | 10/10 | 8/10 | 8/10 | 0 |
| Current shipped code | 10 (repeat, same dataset) | 30 | 25 | 83.3% | 10/10 | 8/10 | 7/10 | 0 |
| Current shipped code | 11 (repeat, same dataset) | 30 | 26 | 86.7% | 10/10 | 9/10 | 7/10 | 0 |
 
**Validator/prompt ablation — outcome of each change, isolated:**
 
| Change | Effect |
|---|---|
| Grounding prompt reframe | Missing accuracy 50%→80% on first measurement; net accuracy +10pp; error cases eliminated (3→0) |
| Duplicate-instance + identical-bbox guards (added on top) | No net accuracy gain across 4 repeated runs (83.3%, 86.7%, 86.7%, 83.3% — mean 85.0%); introduced new instability in previously-stable cases; the case the bbox guard specifically targeted (`a_04`) failed in every run regardless |
 
**Case-level reproducibility, same dataset (`dataset_sha256: 40403a54...`),
15 runs total (spans the reframe-only, +guards, and current-shipped-code
versions — all three share the same `vision_contract.py` SYSTEM text once the
guards were reverted, so they are pooled as one series):**
 
| Case | Kind | Result across 15 runs | Interpretation |
|---|---|---|---|
| `a_23` | ambiguous | FAIL ×15 | Still fully deterministic across every run recorded so far (one of two same-class objects not detected) |
| `a_04` | missing | FAIL ×13, PASS ×2 | **No longer treated as deterministic.** Passed for the first time in runs 8 and 11 of this batch — see note below |
| `a_10` | missing | PASS ×12, FAIL ×1 | Mostly stable; one outlier, not a repeatable pattern |
| `a_19` | missing | PASS ×11, FAIL ×1 | Mostly stable; one outlier, not a repeatable pattern |
| `a_17` | ambiguous | PASS ×9, FAIL ×3 | Weakly non-deterministic; two more flips this batch than previously recorded |
| `a_16` | missing | FAIL ×10, PASS ×2 | Non-deterministic, leans FAIL |
| `a_28` | ambiguous | FAIL ×11, PASS ×2 | Non-deterministic, leans FAIL; earlier "genuine improvement" reading stays retracted |
 
**Correction to the previous entry**: `a_04` was reported above as
"deterministic, FAIL ×11" through the last update. Two of the four most
recent runs (both taken on the exact same dataset and, per this session's
review of the current zip, the exact same `vision_contract.py`) came back
PASS. That retracts the "deterministic model limitation" framing for
`a_04` specifically — it now belongs in the same non-deterministic bucket as
`a_16`/`a_28`, just with a higher pass rate (2/15 vs. their ~15-20%). `a_23`
is now the only case with a clean FAIL streak across every run recorded.
Whether the 2 new passes reflect real non-determinism in the model or a
side-effect of `student_a.py`'s new `_plane_projected`/`_depth_in_region`
fallbacks (both new in this session's reviewed zip, both scoped to
non-fresh-hint `describe()` calls — `eval.grounding` calls `ground()`
directly, which should NOT go through those fallbacks) has not been isolated;
worth re-running with a debugger/log check on which code path produced the
two PASS cases before writing this up as resolved.
 
The 90.0% figure (runs 7 and 8) has now been reproduced once, but the
83.3%–90.0% spread across 11 runs on the current code means neither end
should be read as "the" accuracy — report a range, not a point estimate.
 
## Scene Description Accuracy
 
| Query phrasing | cases | correct | accuracy |
|---|---|---|---|
| "What supported objects and colors are visible?" (original eval default) | 29 (1 error case excluded) | 16 | 55.2% |
| "What objects are nearby?" (PDF's exact wording, same 30 scenes) | 30 | 20 | 66.7% |
| "What supported objects and colors are visible?" (repeat, later code version) | 30 | 17 | 56.7% |
| "What supported objects and colors are visible?" (repeat, later code version) | 30 | 15 | 50.0% |
| "What supported objects and colors are visible?" (repeat, current shipped code) | 30 | 17 | 56.7% |
 
Four runs of the same default query on the same 30 scenes now read 55.2%,
56.7%, 50.0%, 56.7% — a roughly 50–57% band, not a single number; scene
description accuracy should be reported as a range against the ≥90% target.
 
**Hallucination breakdown, most recent run (13 failing cases, all
hallucinations, zero missed-object failures):**
 
| Hallucinated class | Count | Share of all 30 cases |
|---|---|---|
| stone | 5 | 16.7% |
| cube | 5 | 16.7% |
| bottle | 3 | 10.0% |
| region | 1 | 3.3% |
 
`eval.a_vqa_check` (object-mention recall/hallucination, run on the same
data) reports the identical 17/30 "fully correct" count and the identical
per-case hallucinated-class list as `eval.a_scene_description` above — the
two scripts measure the same underlying thing in different report formats;
treat their results as one data point, not two independent confirmations.
This run's `eval.a_vqa_check` also reports **0/30 color self-consistency
mismatches** (`find_color_mismatches`), versus the earlier finding of 2
mismatches (`a_12`, `a_19`) on a previous run — the color self-consistency
failure mode is evidently non-deterministic too, not a standing defect on
those two specific scenes.
 
Every failure in this run was a hallucination, not a missed object — recall
itself is not the limiting factor; the model reliably reports what it can
correctly identify, but fabricates additional objects at a meaningfully high
rate across all three classes, not just the stone/region case documented
elsewhere.
 
## Bounding Box Precision
 
| Run | Cases checked | Mean IoU | Cases with IoU ≥ 0.5 |
|---|---|---|---|
| 1 | 10 | 0.872 | 10/10 = 100.0% |
| 2 (repeat) | 10 | 0.874 | 10/10 = 100.0% |
| 3 (repeat) | 10 | 0.860 | 10/10 = 100.0% |
| 4 (repeat) | 10 | 0.868 | 10/10 = 100.0% |
 
Mean IoU across 4 runs: 0.868 (range 0.860–0.874) — stable, well clear of the
≥0.5 pass threshold every time.
 
## Spatial Reasoning Accuracy
 
| Relation type | cases | correct | accuracy |
|---|---|---|---|
| Containment | 6 | 6 | 100.0% |
| Relative distance | 6 | 6 | 100.0% |
| Combined (pilot, n=12) | 12 | 12 | 100.0% |
| Containment (larger sample) | 10 | 10 | 100.0% |
| Relative distance (larger sample) | 10 | 10 | 100.0% |
| **Combined (n=20)** | **20** | **20** | **100.0%** |
| Combined (n=20, independent repeat) | 20 | 20 | 100.0% |
| Combined (n=20, 2nd independent repeat) | 20 | 20 | 100.0% |
 
Test cases were deliberately well-separated, not boundary/near-tie placements;
this measures reliability on unambiguous spatial relations only, now
confirmed consistent across four independent runs.
 
## VQA example-question cross-check
 
| Question | Interface | Accuracy |
|---|---|---|
| "Where is the stone?" (free-text VQA, presence/absence) | `describe()` | 76.7% (23/30) |
| Same underlying fact (stone present/absent), structured API | `ground()` missing-case accuracy | 50.0% (5/10) |
| "Which object is inside the red area?" (identification framing) | `describe()` | 100.0% (10/10) |
 
The same underlying fact answered differently depending on which interface
method and which question framing was used — see "Remaining limits" below.
 
## Full end-to-end result (round 9, 50-trial formal suite)
 
`eval/trials/final50`, tag `final-40`: **40/50 correct, 0 false claims**
(target was 43). Manipulation 37/47 claimed∧achieved; refuse/clarify 3/3.
 
## Remaining limits
 
- **Memory moved to B** (round 9, `planner/memory.py`): A no longer retains
  any cross-frame state of its own; `recall()` and `memory_*` attributes are
  removed and explicitly test-enforced absent.
- **Depth-in-region fallback exists but is not yet proven effective**: added
  specifically for the mislabelled-object-on-region pattern below, but did
  not fire in the case it targeted in the most recent formal run (`f48`,
  remained `UNLOCALIZED`).
- **A bottle clipped at the image top is incorrectly rejected**
  (`"clipped object centre is below the table top"`, case `f34`) — a bug in
  `partial_object_center`'s rest-height margin check for that orientation,
  open as of round 9.
- **Region/object confusion at the image boundary** (`a_04`): **retracted as
  "deterministic."** It reproduced in 11/11 runs through the previous update,
  but 2 of the 4 most recent runs on the identical dataset and current
  shipped code came back PASS. The underlying mechanism (a fabricated
  detection sharing the real region's exact box/depth, so every individual
  field reads as valid) is unchanged and still the best explanation for why
  it fails as often as it does, but it can no longer be called a stable
  100%-reproducible limitation — treat it as a high-frequency but genuinely
  non-deterministic failure (13/15 runs), in the same category as `a_16` and
  `a_28`, not a separate "proven" class. The earlier query-adaptive-color and
  `f34`-style rejection observations from prior runs still stand as evidence
  of the mechanism; they just no longer imply the failure is unavoidable
  every time.
- **Duplicate-instance recall** (`a_23`, now 15/15 failures): the only case in
  the dataset that has never once passed across every run recorded. The model
  omits one of two present same-class objects, despite an explicit
  system-prompt instruction to report every instance, and confirmed
  unaddressed by the full merged fix set for the same structural reason —
  every downstream filter operates on detections the model already produced;
  none can recover a detection that was never emitted. This is now the
  strongest deterministic-failure claim in the whole dataset.
- **Response non-determinism**: across 15 runs on an identical scene, prompt,
  and dataset at `temperature=0`, **no case has proven fully deterministic
  except `a_23`** (the `a_04` correction above). Everything else that ever
  failed showed some variance, ranging from a single outlier (`a_10`, `a_19`
  — 1 flip in 12) to moderate instability (`a_17` — 3 flips in 12) to genuine
  instability (`a_16`, `a_28`, and now `a_04` — 13–20% flip rate). A single
  evaluation run is one sample, not a stable result; this dataset has now
  produced enough repeats that the practical takeaway is: report accuracy as
  a range (83.3%–90.0% on current code) and name `a_23` as the one
  reproducible failure, rather than claiming any other case is "solved" or
  "broken" from a small number of runs.
- **Query-phrasing sensitivity**: semantically equivalent scene-description
  queries differ by 11.5 percentage points (55.2% vs. 66.7%); the same
  presence/absence fact differs by 26.7 points depending on interface method
  (`ground()` vs. `describe()`).
- **Spatial reasoning is untested near decision boundaries**: 100% accuracy
  reflects easy, well-separated cases only.
- Prompt-only mitigation attempts (duplicate-instance instruction,
  identical-bbox guard) measured no net benefit and introduced new
  instability elsewhere; this is treated as a characterized limit of
  prompt-only iteration on an already-lengthy system prompt, not an
  unfinished task.
