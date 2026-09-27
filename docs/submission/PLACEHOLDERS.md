# Report placeholders

Values for the `[token]` placeholders in `docs/submission/report_draft.docx`. `scripts/fill_report.py` reads the
table below and replaces each token in the draft. Tokens are provisional until the draft is in the repo; they will
be aligned with the draft's exact bracketed tokens.

**Headline run (the only source for end-to-end numbers):** `runs/final2/full_final`, tag `final2-44`, commit
`8e1dd23` (2026-09-25 03:09 +08). 50 trials from `eval/trials/final50`; A = Student A perception (Qwen-VL),
B = Student B planner (`qwen3.7-plus-2026-05-26`, thinking off), C = Student C executor. All derived values are
recomputed by `python scripts/submission_numbers.py` (output `docs/submission/numbers.json`, 0 API calls).

## Values

| Token | Section | Fill | Detail | Source |
|---|---|---|---|---|
| [run id] | Task 5 | runs/final2/full_final | tag `final2-44` | `runs/final2/full_final/E2E_TABLE.md` |
| [commit] | Task 5 | 8e1dd23 |  | `git rev-parse --short final2-44` |
| [e2e score] | Task 5 | 44/50 | correct outcome (success, refusal or clarification as expected) | `runs/final2/full_final/rows.json` score |
| [false claims] | Task 5 | 0 |  | same |
| [manipulation correct] | Task 5 | 41/47 | claimed ∧ achieved among the 47 manipulation trials | same |
| [reject/clarify correct] | Task 5 | 3/3 |  | same |
| [claimed∧actual] | Task 5 confusion | 43 |  | `numbers.json` confusion |
| [actual not claimed] | Task 5 confusion | 2 | f20, f31: placed correctly, never verified | same |
| [claimed not actual] | Task 5 confusion | 0 |  | same |
| [neither] | Task 5 confusion | 4 | f34, f38, f41, f45 | same |
| [correct refusal] | Task 5 confusion | 1 | f49 | same |
| [clarification then success] | Task 5 confusion | 2 | f05, f50; included in the 43 | same |
| [mean sim time] | Task 5 | 12.4 s | mean simulated time of the 43 successful trials | `numbers.json` sim_task_time_success_s |
| [median sim time] | Task 5 | 10.9 s |  | same |
| [mean wall time] | Task 5 | 61.1 s per trial | median 34.4 s; successful trials 40.7 s | `numbers.json` wall_s_per_trial |
| [calls per trial A] | Task 5 | 14.1 | 703 in total | `rows.json` calls_A |
| [calls per trial B] | Task 5 | 2.1 | 106 in total | `rows.json` calls_B |
| [calls per trial] | Task 5 | 16.2 | 809 in total | `rows.json` |
| [standard] | Task 5 variation | 19/20 | f06–f25: student_c + student_c_v2 layouts | `numbers.json` groups |
| [scene variation] | Task 5 variation | 13/17 | f02, f26–f41 | same |
| [instruction variation] | Task 5 variation | 5/6 | f03, f42–f46 | same |
| [search] | Task 5 variation | 1/1 | f04 | same |
| [two-stone colour] | Task 5 variation | 2/2 | f47, f48 | same |
| [clarification/infeasible] | Task 5 variation | 3/3 | f05, f49, f50 | same |
| [smoke standard] | Task 5 variation | 1/1 | f01 | same |
| [stone] | Task 5 per object | 25/26 | includes stone2 targets | `numbers.json` per_object |
| [cube] | Task 5 per object | 13/13 |  | same |
| [bottle] | Task 5 per object | 5/10 |  | same |
| [perception failures] | Task 5 attribution | 2: f20, f31 | placed correctly but the exact instance was never re-verified | `E2E_TABLE.md` module A; REPORT §14 |
| [execution failures] | Task 5 attribution | 1: f41 | bottle near the right edge: TARGET_LOST / UNREACHABLE after parking | module C |
| [planning failures] | Task 5 attribution | 0 | B caused none of the six | REPORT §14 |
| [scene failures] | Task 5 attribution | 3: f34, f38, f45 | far bottles: the parking pose touches the table, the arm is then out of reach, search exhausted | module A/C |
| [two-stone trials] | advanced scenario | f47, f48 | colour disambiguation; stone2 also present in f05, f18, f19, f20, f23, f25, f50 | `eval/trials/final50/*.yaml` |
| [clarification trials] | advanced scenario | f05, f50 | end to end; plus the 4 chains in B_CLARIFICATION_CHAINS.md | same |
| [obstacle trials] | advanced scenario |  | **none in the repo** (see Discrepancies) | — |
| [grounding accuracy] | Task 2 |  | from `docs/validation/A_METRICS.md` (checked against the draft table when it arrives) | A_METRICS.md |
| [planning 32] | Task 3 | 32/32 | final2 gate, thinking off | `runs/final2/gate/s2_eval32`; `runs/final2/PROGRESS.md` |
| [planning 20] | Task 3 | 20/20 | final2 gate, thinking off | `runs/final2/gate/s2_var20` |
| [clarification chains] | Task 3 | 4/4 | each chain reaches the correct READY plan | `docs/validation/B_CLARIFICATION_CHAINS.md` |
| [student_c] | Task 4 | 10/10 | grasp 10/10, place 10/10; xy error mean 0.81 cm, max 1.62 cm | re-run 2026-09-27 on e2e `e674c62`, `runs/submission/manip_student_c` |
| [student_c_v2] | Task 4 | 10/10 | grasp 10/10, place 10/10; xy error mean 0.76 cm, max 1.14 cm | `runs/submission/manip_student_c_v2` |
| [final50 manipulation] | Task 4 | 48/50 | scripted skills, GT perception + rule planner | `runs/bonus/final50/scripted/rows.json` |
| [ACT C2] | bonus | 27/30 | ACT + bottle demos, C2 full executor first grasp; 28/30 tasks; skill level 25/30 | `runs/bonus/manip/act_2k_bottle_C2`, `runs/bonus/eval/act_2k_bottle_C2.jsonl` (run 2026-09-27) |
| [ACT C4] | bonus | 30/30 | ACT + bottle demos, C4 full executor; skill level 28/30 | `runs/bonus/manip/act_2k_bottle_C4`, `runs/bonus/eval/act_2k_bottle_C4.jsonl` |
| [test count] | code | 427 passed, 1 skipped |  | `pytest -q` on e2e `e674c62`, 2026-09-27 |
| [final run calls] | AI services | 809 live calls | A 703, B 106 | `runs/final2/CALLS.txt`, `rows.json` |
| [final run tokens] | AI services | A 447,659 in / 117,207 out; B 342,516 in / 43,033 out |  | sum of `api_stats` in the 50 trial records |
| [project calls] | AI services | 9,164 ledgered live calls | night run 1,498 + final 5,562 + final2 2,057 + video re-recording 47 | `runs/night/CALLS.txt` + REPORT §7, `runs/final/CALLS.txt`, `runs/final2/CALLS.txt`, `runs/final/STATUS.md` |
| [project tokens] | AI services | ≥ 6.94 M input / 1.38 M output tokens | A 3.82 M / 1.02 M; B 3.13 M / 0.36 M, over the 6,983 live requests with trial records (lower bound) | sum of `api_stats` over all unique `runs/**/trial_record.json` |

## Discrepancies

These are listed, not silently changed. Each needs a human decision in the draft or in the named file.

1. `docs/validation/B_REPORT_SECTION.md` line 74 says "RUN 1 of the final 50-trial run scored 40/50". That is the
   historical `final-40` run; the headline result is 44/50 (`final2-44`).
2. `docs/validation/B_REPORT_SECTION.md` Table B3 says "103 live B calls" for the final2 run; its `rows.json` sums to
   106 B attempts (A 703 + B 106 = 809). The 3-call difference is probably retried attempts; use 106 to match 809.
3. Planning accuracy has two sets: 31/32 and 19/20 (plus tier, before the final2 changes: Table B2 and REPORT
   line 484) and 32/32 and 20/20 (final2 gate, thinking off). Say which the draft means; the submitted code gives
   32/32 and 20/20.
4. No trial in `eval/trials/**` places an obstacle on the region ("obstacle-blocked region"). Either drop the case
   from the advanced-scenario list or name the trial that exercises it.
5. The project token total is a lower bound. Calls without a trial record (B benchmark suites, aborted runs, provider
   reruns: 9,164 − 6,983 = 2,181 calls) are not in the token sum, and development before 2026-09-23 was never
   ledgered.
6. "10/10" for student_c / student_c_v2 dates from 2026-09-07 (`docs/STUDENT_C_README.md`). It was re-verified
   today on the submission code: 10/10 on both, so it stands. REPORT line 13's 9/10 is a historical night-run
   regression that was reverted.
7. Cross-check of the draft's own numbers (Task 2–4 tables, 44/50, 0 false claims, 31/32, 19/20, 10/10, 48/50,
   47/50, 43/50, 0.21/0.45 cm, 5.9 s / 17.6 s, tier comparison) is pending until `report_draft.docx` is in the repo.
   Repo values: 44/50 ✔, 0 ✔, 31/32 and 19/20 ✔ (plus tier, pre-final2), 10/10 ✔, 48/50 ✔, 47/50 ✔, 43/50 ✔,
   0.21 vs 0.45 cm ✔ (`place_act_wide` vs scripted, C1), 5.9 s / 17.6 s ✔ (REPORT line 484), tier comparison
   flash 29/32 · 17/20, plus 31/32 · 19/20, max 32/32 · 20/20 ✔ (`B_MODEL_COMPARE.md`).

## Video link

See `docs/submission/DEMO_NOTES.md` (`[video link]` until the mp4 is uploaded).
