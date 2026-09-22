# Student B metrics

Source: every B audit under `runs/night` (live or cached responses only).

- planner calls (audits): 168
- first-pass-valid rate: 163/168 = 97.0%
- repair rate (calls that needed a repair): 5/168 = 3.0%
- repair success rate (accepted after repair): 3/5 = 60.0%
- accepted overall: 166/168 = 98.8%
- normalisations applied: 0 in total

**Validator ablation**: first responses the contract rejected before repair. Without the validator these plans would have gone to the robot as they were: **5/168**.

| first-response rejection reason | count |
|---|---|
| Goal object color mismatch | 2 |
| Action 2: target needs a current, unambiguous 3D position | 1 |
| Unknown region class 'drawer' | 1 |
| Unknown object class 'sphere' | 1 |

| call type | calls | live responses | cached | prompt tokens | completion tokens | latency s |
|---|---|---|---|---|---|---|
| initial plan | 76 | 72 | 8 | mean 2545.3, median 2512.0 | mean 1143.0, median 1110.5 | mean 17.5, median 16.6 |
| held-object replan | 3 | 3 | 0 | mean 2511.7, median 2475.0 | mean 1048.3, median 1101.0 | mean 14.7, median 15.1 |
| post-clarification | 10 | 10 | 0 | mean 2827.5, median 2818.5 | mean 1526.0, median 1493.0 | mean 22.4, median 21.0 |
| other replan | 79 | 74 | 6 | mean 2783.0, median 2772.0 | mean 1342.3, median 1272.5 | mean 20.8, median 19.6 |


## Variation suite (`eval/trials/student_b_variation`)

| category | cases | goal accuracy | status accuracy |
|---|---|---|---|
| ambiguous | 4 | 4/4 = 100.0% | 4/4 = 100.0% |
| color_negation | 4 | 4/4 = 100.0% | 4/4 = 100.0% |
| infeasible | 4 | 3/4 = 75.0% | 3/4 = 75.0% |
| paraphrase | 4 | 4/4 = 100.0% | 4/4 = 100.0% |
| relation | 4 | 4/4 = 100.0% | 4/4 = 100.0% |
