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
 
**Validator/prompt ablation — outcome of each change, isolated:**
 
| Change | Effect |
|---|---|
| Grounding prompt reframe | Missing accuracy 50%→80% on first measurement; net accuracy +10pp; error cases eliminated (3→0) |
| Duplicate-instance + identical-bbox guards (added on top) | No net accuracy gain across 4 repeated runs (83.3%, 86.7%, 86.7%, 83.3% — mean 85.0%); introduced new instability in previously-stable cases; the case the bbox guard specifically targeted (`a_04`) failed in every run regardless |
 
**Case-level reproducibility, same dataset (`dataset_sha256: 40403a54...`),
7 runs total:**
 
| Case | Kind | Result across 7 runs | Interpretation |
|---|---|---|---|
| `a_04` | missing | FAIL ×7 | Deterministic model limitation (region/object box conflated at image boundary) |
| `a_23` | ambiguous | FAIL ×7 | Deterministic model limitation (one of two same-class objects not detected) |
| `a_10` | missing | PASS ×5, FAIL ×1 (most recent run) | Previously assumed stable; now confirmed non-deterministic too |
| `a_16` | missing | FAIL, FAIL, PASS, FAIL, FAIL | Non-deterministic |
| `a_19` | missing | PASS, FAIL, PASS, PASS, PASS | Non-deterministic |
| `a_17` | ambiguous | PASS, PASS, FAIL, PASS, PASS | Non-deterministic |
| `a_28` | ambiguous | FAIL, FAIL, PASS, PASS, FAIL, FAIL | **Not** a genuine improvement — reverted to failing on further sampling; the earlier "plausible improvement" reading is retracted |
 
## Scene Description Accuracy
 
| Query phrasing | cases | correct | accuracy |
|---|---|---|---|
| "What supported objects and colors are visible?" (original eval default) | 29 (1 error case excluded) | 16 | 55.2% |
| "What objects are nearby?" (PDF's exact wording, same 30 scenes) | 30 | 20 | 66.7% |
| "What supported objects and colors are visible?" (repeat, later code version) | 30 | 17 | 56.7% |
 
**Hallucination breakdown, most recent run (13 failing cases, all
hallucinations, zero missed-object failures):**
 
| Hallucinated class | Count | Share of all 30 cases |
|---|---|---|
| stone | 6 | 20.0% |
| cube | 5 | 16.7% |
| bottle | 2 | 6.7% |
 
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
 
## Spatial Reasoning Accuracy
 
| Relation type | cases | correct | accuracy |
|---|---|---|---|
| Containment | 6 | 6 | 100.0% |
| Relative distance | 6 | 6 | 100.0% |
| Combined (pilot, n=12) | 12 | 12 | 100.0% |
| Containment (larger sample) | 10 | 10 | 100.0% |
| Relative distance (larger sample) | 10 | 10 | 100.0% |
| **Combined (n=20)** | **20** | **20** | **100.0%** |
 
Test cases were deliberately well-separated, not boundary/near-tie placements;
this measures reliability on unambiguous spatial relations only, confirmed
consistent across two independent runs at different sample sizes.
 
## VQA example-question cross-check
 
| Question | Interface | Accuracy |
|---|---|---|
| "Where is the stone?" (free-text VQA, presence/absence) | `describe()` | 76.7% (23/30) |
| Same underlying fact (stone present/absent), structured API | `ground()` missing-case accuracy | 50.0% (5/10) |
| "Which object is inside the red area?" (identification framing) | `describe()` | 100.0% (10/10) |
 
The same underlying fact answered differently depending on which interface
method and which question framing was used — see "Remaining limits" below.
 
## Remaining limits
 
- **Region/object confusion at the image boundary**: reproduced in 5/5 runs
  on one specific scene (`a_04`); survived every prompt and code-level change
  attempted, including the identical-bbox guard written specifically to
  address it.
- **Duplicate-instance recall**: the model omits one of two present same-class
  objects in roughly 1 of 5 same-class-duplicate scenes, despite an explicit
  system-prompt instruction to report every instance. No selection-time logic
  (prompt or code) can correct a detection that was never produced.
- **Response non-determinism**: 5 of 30 cases (`a_10`, `a_16`, `a_17`, `a_19`,
  `a_28`) changed verdict across repeated runs on an identical scene, prompt,
  and dataset, at `temperature=0`. Only `a_04` and `a_23` have proven stable
  (7/7 fails each) across every run so far. A single evaluation run is one
  sample, not a stable result — this project's own repeated-run data shows
  roughly 1 in 6 cases is unstable even under otherwise identical conditions.
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
