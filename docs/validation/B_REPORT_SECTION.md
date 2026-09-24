# Part B: Language planning with a validated Qwen planner (draft)

## Method

Student B turns a natural-language instruction and Student A's current scene description into an executable plan for Student C. The planner (`planner/student_b.py`) sends Qwen a text-only JSON input. It holds the instruction, A's instances (ID, class, status, 3D position, colour), the candidate groups per class, the bound goal from earlier calls, the execution context (held or just-released instance), the recent action history and the public vocabulary. Qwen answers in a fixed wire schema (`student-b-plan-v1`): a status (READY, NEEDS_SEARCH, NEEDS_CLARIFICATION, INFEASIBLE), a goal (object and region IDs, class names, requested colour), up to 16 skill actions, a reason, an optional clarification question and, since E3, an optional rationale.

The model never writes coordinates. Code compiles the answer into the backbone `Plan` and copies every position from A's scene. Once a goal field is bound it is locked for the rest of the episode, so replans cannot drift to another object.

All experiments use `qwen3.7-plus-2026-05-26` with thinking disabled. On the 32-case planning suite, thinking off scores 31/32 against 32/32 with thinking on, and cuts mean latency per case from 17.6 s to 5.9 s. The final e2e runs use this setting.

## Validator

`planner/contract.py` checks each answer before it can reach the robot:
- the schema;
- the goal against the vocabulary and A's perceived IDs, including the colour qualifier;
- goal locking;
- per-skill action patterns and skill order (APPROACH before GRASP, MOVE_TO before PLACE, final placement VERIFY and STOP);
- held-object consistency;
- that every motion target is currently LOCALIZED;
- the backbone's `validate_plan`.

A rejected answer gets one repair round, which shows the model the validation error. If the repair also fails, B returns `PlanStatus.REJECTED` (contract v4) and the orchestrator replans from a fresh view instead of ending the episode. Two deterministic normalisations are recorded rather than repaired:
- duplicate ID fields are cleared;
- a READY plan that starts with SEARCH becomes NEEDS_SEARCH.

In the final 50-trial run (RUN 1), 106 of B's 110 calls were valid on the first pass. The other 4 were repaired and accepted (Table B1). Without the validator, those 4 plans would have reached the robot as the model wrote them: three changed the locked goal region and one targeted an instance with no current 3D position.

## Memory

Memory is owned by B. A reports only the current frame, and C keeps no state across actions. `planner/memory.py` records each instance's last LOCALIZED position with its frame index, the base pose, and held/released flags. When the bound goal region is out of view, B plans from its remembered position if it is at most 40 frames old and the base has moved at most 0.5 m. The audit records which instance came from memory and its age. B never recalls the object to be grasped, because GRASP needs fresh 3D evidence, and never a held or released instance. In RUN 1, B planned from memory in 23 of 110 calls (median age 17 frames, maximum 38; Figure B3).

## Metrics

**Table B1: validation and conversions (RUN 1, 110 live calls).**

| measure | value |
|---|---|
| first-pass valid | 106/110 (96.4 %) |
| repaired → accepted | 4/4 |
| READY with SEARCH → NEEDS_SEARCH | 1 |
| final status READY / NEEDS_SEARCH / NEEDS_CLARIFICATION / INFEASIBLE | 59 / 46 / 4 / 1 |
| mean latency: initial plan / replan | 5.2 s / 4.2 s |

**Table B2: Qwen tier comparison, cache off, thinking off.**

| model | 32-case | 20-case variation | mean latency per case |
|---|---|---|---|
| qwen3.7-flash | 29/32 | 17/20 | 4.4 s |
| **qwen3.7-plus (used)** | 31/32 | 19/20 | 5.3 s |
| qwen3.7-max | 32/32 | 20/20 | 6.2 s |

Token use is about the same across tiers (≈ 98–101k prompt and 11–12k completion tokens on the 32-case suite), so the difference in cost comes from each tier's price per token. Flash's errors are wrong bindings and needless clarifications. Plus has two misses. One is test_26: the answer is a correct refusal, but it names a class outside the vocabulary, so the contract rejects it. The other is a relational reference in the variation suite.

**End to end.** RUN 1 of the final 50-trial run scored 40/50 with 0 false claims. Only 2 of the 10 failures were attributed to B. In both, the instruction said "red stone" while A labels that stone `dark_red`. B asked which stone was meant in one trial, and could not match its SEARCH in the other. The four scripted clarification chains all succeeded: B asked, the answer bound the right object or region, and the replan was READY (Figure B4).

## Figures

- **Figure B1** (`figs/b_status.png`): final plan status of every B call in RUN 1. NEEDS_SEARCH is common because trials start with objects out of view.
- **Figure B2** (`figs/b_latency.png`): distribution of live response latency by call type (initial plan, replan, held-object replan, post-clarification).
- **Figure B3** (`figs/b_memory_age.png`): age in frames of the region positions B planned from.
- **Figure B4** (`figs/b_clarification_timeline.png`): four clarification chains, from question to scripted answer to READY plan, on a time axis.
- **Figure B5** (`figs/b_model_compare.png`): accuracy and mean latency per case for the three Qwen tiers on both suites.

## Limitations

- **Colour words.** B does not map instruction colours onto A's palette. "Red stone" should resolve to A's `dark_red` stone through the vocabulary synonyms. This caused both B-attributed e2e failures.
- **Refusals outside the vocabulary.** The contract rejects a correct INFEASIBLE answer that names a class outside the vocabulary (test_26).
- **Memory scope.** Memory covers only the goal region. Trials that used memory succeeded less often (2/8 vs 28/33), but B needs memory exactly in the hard episodes, so this is not a controlled comparison.
- **Measurement scope.** Accuracy comes from 52 labelled planning cases and one 50-trial e2e set, and each configuration was run once. Run-to-run variance is visible: in the two full e2e runs, 6 trials changed result. Five of the changes followed STAGE F code changes that were later reverted; one (f19) was model variance.
- **Rationale.** The model often writes several sentences where one was asked for.
