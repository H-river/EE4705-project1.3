### Full executor: grasp success (first GRASP holds the intended object), 30 episodes/cell

| Policy | C1 in-dist. | C2 ±20 cm | C3 bottle (OOD) | C4 near distractor | WRONG_OBJECT | undetected | false claims |
|---|---|---|---|---|---|---|---|
| Scripted skill | 29/30 | 30/30 | 29/30 | 30/30 | 0 | 0 | 0 |
| ACT | 29/30 | 26/30 | 0/30 | 30/30 | 0 | 0 | 0 |
| Diffusion Policy | 29/30 | 27/30 | 0/30 | 30/30 | 0 | 0 | 0 |

### Full executor: task success (object placed in the region, oracle)

| Policy | C1 in-dist. | C2 ±20 cm | C3 bottle (OOD) | C4 near distractor |
|---|---|---|---|---|
| Scripted skill | 29/30 | 30/30 | 29/30 | 30/30 |
| ACT | 29/30 | 28/30 | 0/30 | 30/30 |
| Diffusion Policy | 29/30 | 27/30 | 0/30 | 30/30 |

### Skill level: grasp success and mean time-to-attach, 30 episodes/cell

| Policy | C1 in-dist. (t̄ s) | C2 ±20 cm (t̄ s) | C3 bottle (OOD) (t̄ s) | C4 near distractor (t̄ s) | WRONG_OBJECT |
|---|---|---|---|---|---|
| Scripted skill | 30/30 (3.02) | 29/30 (2.78) | 27/30 (2.77) | 30/30 (2.98) | 0 |
| ACT | 30/30 (3.18) | 26/30 (2.70) | 0/30 (—) | 30/30 (2.83) | 0 |
| Diffusion Policy | 30/30 (3.20) | 25/30 (3.04) | 1/30 (2.02) | 27/30 (3.05) | 0 |

### Ablations on C1 (30 episodes each)

| Variant | full executor C1 (grasp) | skill level C1 | mean time-to-attach (skill, s) | undetected |
|---|---|---|---|---|
| ACT n_action_steps 10 | 29/30 | 27/30 | 3.23 | 0 |
| ACT n_action_steps 25 (trained) | 29/30 | 30/30 | 3.18 | 0 |
| ACT n_action_steps 50 | 29/30 | 29/30 | 2.71 | 0 |
| DP DDIM 5 steps | 29/30 | 30/30 | 3.22 | 0 |
| DP DDIM 10 steps (trained) | 29/30 | 30/30 | 3.20 | 0 |
| DP DDIM 50 steps | 29/30 | 30/30 | 3.28 | 0 |
| DP DDPM 50 steps | 29/30 | 30/30 | 3.30 | 0 |
