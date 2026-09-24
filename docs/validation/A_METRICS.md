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
 
**Validator/prompt ablation — outcome of each change, isolated:**
 
| Change | Effect |
|---|---|
| Grounding prompt reframe | Missing accuracy 50%→80% on first measurement; net accuracy +10pp; error cases eliminated (3→0) |
| Duplicate-instance + identical-bbox guards (added on top) | No net accuracy gain across 4 repeated runs (83.3%, 86.7%, 86.7%, 83.3% — mean 85.0%); introduced new instability in previously-stable cases; the case the bbox guard specifically targeted (`a_04`) failed in every run regardless |
 
**Case-level reproducibility, same dataset (`dataset_sha256: 40403a54...`),
11 runs total (spans both the reframe-only and +guards code versions):**
 
| Case | Kind | Result across 11 runs | Interpretation |
|---|---|---|---|
| `a_04` | missing | FAIL ×11 | Deterministic model limitation (region/object box conflated at image boundary) |
| `a_23` | ambiguous | FAIL ×11 | Deterministic model limitation (one of two same-class objects not detected) |
| `a_10` | missing | PASS ×8, FAIL ×1 | Mostly stable; one outlier, not a repeatable pattern |
| `a_17` | ambiguous | PASS ×7, FAIL ×1 | Mostly stable; one outlier, not a repeatable pattern |
| `a_19` | missing | PASS ×7, FAIL ×1 | Mostly stable; one outlier, not a repeatable pattern |
| `a_16` | missing | FAIL ×6, PASS ×2 | Non-deterministic, leans FAIL |
| `a_28` | ambiguous | FAIL ×7, PASS ×2 | Non-deterministic, leans FAIL; earlier "genuine improvement" reading stays retracted |
 
The 90.0% run (the current single-run high point) is not yet independently
repeated — treat it the same as every other individual run in this table,
not as a new stable baseline, until confirmed again.
 
## Scene Description Accuracy
 
| Query phrasing | cases | correct | accuracy |
|---|---|---|---|
| "What supported objects and colors are visible?" (original eval default) | 29 (1 error case excluded) | 16 | 55.2% |
| "What objects are nearby?" (PDF's exact wording, same 30 scenes) | 30 | 20 | 66.7% |
| "What supported objects and colors are visible?" (repeat, later code version) | 30 | 17 | 56.7% |
| "What supported objects and colors are visible?" (repeat, later code version) | 30 | 15 | 50.0% |
 
**Hallucination breakdown, most recent run (15 failing cases, all
hallucinations, zero missed-object failures except one case that both missed
and hallucinated):**
 
| Hallucinated class | Count | Share of all 30 cases |
|---|---|---|
| stone | 7 | 23.3% |
| cube | 5 | 16.7% |
| bottle | 2 | 6.7% |
| region | 1 | 3.3% (first time this class has been hallucinated, not just stone/cube/bottle) |
 
`eval.a_vqa_check` (object-mention recall/hallucination, run on the same
data) reports the identical 15/30 "fully correct" count and the identical
per-case hallucinated-class list as `eval.a_scene_description` above — the
two scripts measure the same underlying thing in different report formats;
treat their results as one data point, not two independent confirmations.
 
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
 
Test cases were deliberately well-separated, not boundary/near-tie placements;
this measures reliability on unambiguous spatial relations only, now
confirmed consistent across three independent runs.
 
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
- **Region/object confusion at the image boundary**: reproduced in 11/11 runs
  on one specific scene (`a_04`), confirmed against the full merged codebase
  (off-table filtering, class/color palette validation, edge-clipped partial
  localization, and tracking hints all active). Traced to why none of these
  catch it: the fabricated detection shares the real region's exact box and
  depth data, so it reads as a legitimately on-table position with a
  perfectly valid color for its (wrong) class — every individual field is
  independently valid, so no field-level or geometric filter can flag it.
  This is a detection-time class hallucination, not a filterable artifact.
  One run additionally showed the hallucinated color changing between the
  general scene description (`dark_red`) and the targeted grounding call for
  "the gray stone" (`gray`) at the identical box, in the same evaluation run
  — the clearest single instance yet of the fabrication adapting to match
  the query rather than anything actually rendered. The same run also
  triggered the team's independently-tracked "clipped object centre is below
  the table top" rejection (previously documented only for a real bottle,
  case `f34`) on this fabricated stone, indicating that geometry check's
  edge case is not specific to bottles.
- **Duplicate-instance recall** (`a_23`, 11/11 failures): the model omits one of
  two present same-class objects roughly 1 in 5 same-class-duplicate scenes,
  despite an explicit system-prompt instruction to report every instance.
  Confirmed unaddressed by the full merged fix set for the same structural
  reason — every downstream filter operates on detections the model already
  produced; none can recover a detection that was never emitted.
- **Response non-determinism**: across 10 runs on an identical scene, prompt,
  and dataset at `temperature=0`, only 2 of 30 cases (`a_04`, `a_23`) were
  fully deterministic (10/10 fails each). Everything else that ever failed
  showed some variance, ranging from a single outlier (`a_10`, `a_17`, `a_19`
  — 1 flip in 8–9 runs) to genuine instability (`a_16`, `a_28` — roughly
  30–20% flip rate). A single evaluation run is one sample, not a stable
  result; only a case that fails across many independent runs should be
  treated as a real, reproducible limitation rather than noise.
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
 
