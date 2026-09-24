# Part B: Language planning with a validated Qwen planner (draft v2, final2)

## Method

Student B turns a natural-language instruction and Student A's current scene description into an executable plan for Student C. The planner (`planner/student_b.py`) sends Qwen a text-only JSON input. It holds the instruction, A's instances (ID, class, status, 3D position, colour), the candidate groups per class, the bound goal from earlier calls, the execution context (held or just-released instance), the recent action history the public vocabulary and, since final2, an explicit list of every supported object and region class (`supported_classes`). Qwen answers in a fixed wire schema (`student-b-plan-v1`): a status (READY, NEEDS_SEARCH, NEEDS_CLARIFICATION, INFEASIBLE), a goal (object and region IDs, class names, requested colour), up to 16 skill actions, a reason, an optional clarification question, an optional rationale (E3) and, since final2, an optional relational `reference` {relation, anchor class, anchor colour} (prompt `qwen-b-v5`).

The model never writes coordinates. Code compiles the answer into the backbone `Plan` and copies every position from A's scene. Once a goal field is bound it is locked for the rest of the episode, so replans cannot drift to another object. When the instruction picks the object by a relation (next to, left/right of, closest to, farthest from), the model only names the relation and the anchor class; `planner/relations.py` chooses the instance from A's positions (nearest/farthest neighbour, or the signed lateral axis in the robot frame, 3 cm margin, exactly one anchor) and overwrites the model's choice when the relation is decisive.

All experiments use `qwen3.7-plus-2026-05-26` with thinking disabled. On the 32-case planning suite, thinking off scored 31/32 against 32/32 with thinking on, and cut mean latency per case from 17.6 s to 5.9 s. The final e2e runs use this setting. With the final2 changes, thinking off scores 32/32 and 20/20 on the variation suite.

## Validator

`planner/contract.py` checks each answer before it can reach the robot:
- the schema;
- the goal against the vocabulary and A's perceived IDs, including the colour qualifier;
- goal locking;
- per-skill action patterns and skill order (APPROACH before GRASP, MOVE_TO before PLACE, final placement VERIFY and STOP);
- held-object consistency;
- that every motion target is currently LOCALIZED;
- the backbone's `validate_plan`;
- (final2) that an INFEASIBLE answer does not call a supported object→region move unsupported. RUN 5 f45 refused the green bottle because A's caption called it "a green cylinder … not a supported object type"; such an answer now goes back for one repair with the supported classes named, and operation refusals (stacking, pouring) are unaffected.

A rejected answer gets one repair round, which shows the model the validation error. If the repair also fails, B returns `PlanStatus.REJECTED` (contract v4) and the orchestrator replans from a fresh view instead of ending the episode. Two deterministic normalisations are recorded rather than repaired:
- duplicate ID fields are cleared;
- a READY plan that starts with SEARCH becomes NEEDS_SEARCH;
- an instruction colour that names a shade of a perceived colour ('red' → `dark_red`, round 9).

**History guard (final2).** If the same action on the same instance has already failed twice in the episode, B does not emit it unchanged: it first asks for a SEARCH (when none ran since that failure), then an APPROACH from a standoff rotated by ±90° (`standoff_rotation_deg`), then returns REJECTED with a reason that says what was tried.

In the final 50-trial run (RUN 1), 106 of B's 110 calls were valid on the first pass. The other 4 were repaired and accepted (Table B1). Without the validator, those 4 plans would have reached the robot as the model wrote them: three changed the locked goal region and one targeted an instance with no current 3D position.

## Memory

Memory is owned by B. A reports only the current frame, and C keeps no state across actions. `planner/memory.py` records each instance's last LOCALIZED position with its frame index, the base pose, and held/released flags. When the bound goal region is out of view, B plans from its remembered position if it is at most 40 frames old and the base has moved at most 0.5 m. The audit records which instance came from memory and its age. B never uses a remembered position to GRASP, because GRASP needs fresh 3D evidence, and never recalls a held or released instance. Since final2 one exception exists for the object: when B would search for a goal object that A cannot localize now, it first APPROACHes the object's remembered position and searches from there. In RUN 1, B planned from memory in 23 of 110 calls (median age 17 frames, maximum 38; Figure B3); in the final2 run, 23 of 103 calls used the remembered region and 11 the remembered object for APPROACH.

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

**Table B3: final2 run (50 trials, `final2-44`, 103 live B calls).**

| measure | value |
|---|---|
| first-pass valid | 100/103 (97.1 %); the 3 repairs were goal-locking errors ("Original goal must keep …"), all accepted |
| final status READY / NEEDS_SEARCH / NEEDS_CLARIFICATION / INFEASIBLE | 60 / 40 / 2 / 1 |
| planned from memory: region / object (APPROACH only) | 23 / 11 |
| history guard: SEARCH first / rotated standoff | 2 / 2 (C ignores the rotation since its re-park was reverted, so those two plans were effectively unchanged) |
| colour aliasing normalisations | 3 |
| mean / median latency per call | 5.3 s / 5.5 s |
| trials failed because of B | **0 of 6** (the run scored 44/50, 0 false claims) |

**Failure types.** Over every audit on disk (`docs/validation/B_FAILURE_TYPES.md`, Figure B6): the 47 labelled benchmark errors are WRONG_STATUS 24, INVALID_PARAMS 15 and WRONG_BINDING 8, with no WRONG_GOAL and no HALLUCINATED_INSTANCE. The 117 contract errors that were repaired are mostly WRONG_GOAL (59, nearly all attempts to change the locked goal on a replan) and INVALID_PARAMS (39). The validator exists for exactly these cases.

**Outcome sentence.** Each episode now ends with one sentence for the user, composed without a model call from the goal, the outcome and the final verification (`TrialRecord.user_report`), e.g. "Placed the grey stone on the red area (2 cm from its centre), confirmed visually." / "Could not confirm - the grey stone is out of view."

**End to end.** RUN 1 of the final 50-trial run scored 40/50 with 0 false claims. Only 2 of the 10 failures were attributed to B. In both, the instruction said "red stone" while A labels that stone `dark_red`. B asked which stone was meant in one trial, and could not match its SEARCH in the other. The four scripted clarification chains all succeeded: B asked, the answer bound the right object or region, and the replan was READY (Figure B4).

## Figures

- **Figure B1** (`figs/b_status.png`): final plan status of every B call in RUN 1. NEEDS_SEARCH is common because trials start with objects out of view.
- **Figure B2** (`figs/b_latency.png`): distribution of live response latency by call type (initial plan, replan, held-object replan, post-clarification).
- **Figure B3** (`figs/b_memory_age.png`): age in frames of the region positions B planned from.
- **Figure B4** (`figs/b_clarification_timeline.png`): four clarification chains, from question to scripted answer to READY plan, on a time axis.
- **Figure B5** (`figs/b_model_compare.png`): accuracy and mean latency per case for the three Qwen tiers on both suites.
- **Figure B6** (`figs/b_failure_types.png`): B failure types in labelled benchmark errors, repaired contract errors and B-attributed e2e failures.

## Limitations

- **Colour words** were a limitation in RUN 1 (both B-attributed failures); round 9's colour aliasing fixed them (f19/f20).
- **Refusals outside the vocabulary.** test_26 (a refusal naming a 'drawer') now passes after one repair; the contract still rejects the first answer.
- **Memory scope.** Memory covers the goal region, and the goal object only for APPROACH. Trials that used memory succeeded less often in RUN 1 (2/8 vs 28/33), but B needs memory exactly in the hard episodes, so this is not a controlled comparison.
- **History guard.** Its rotated-standoff step needs C to honour `standoff_rotation_deg`; C's re-park was reverted in final2, so today the guard effectively goes SEARCH → REJECTED.
- **Measurement scope.** Accuracy comes from 52 labelled planning cases and one 50-trial e2e set. B-only run-to-run variance (plan item 6.3) and the prompt diet (6.4, patch in `docs/night_run/pending/`) were not measured: the run's 2,000-call cap was used up (2,057).
- **Rationale.** The model often writes several sentences where one was asked for.
