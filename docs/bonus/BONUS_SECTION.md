# Bonus: learned grasping behind the skill interface

## Setup

Can a learned visuomotor policy replace the scripted grasp primitive without touching the rest of the system? We
trained two standard imitation-learning policies with lerobot 0.6.1: **ACT** (action chunking with transformers)
and **Diffusion Policy (DP)**. We placed each one behind the existing grasp interface: `LearnedGraspSkill(policy, ckpt)(env, pos)`
has the same call as `skills.grasp(env, pos)`, and the executor picks it through `EE4705_GRASP_POLICY`. From the
post-APPROACH pose the policy runs at 10 Hz for at most 15 s. It observes the arm joints, the target position in the
base frame and the head camera, and outputs the next arm joint setpoint. It then calls `try_attach_near_ee()` once,
as the script does. Everything after that is unchanged: the executor's lift check, the planner, the verification, and
the evaluator's oracle. All experiments use ground-truth perception and the rule planner, so they cost 0 API calls.

## Data

The expert is the scripted APPROACH→REACH→GRASP pipeline. Each demonstration starts at the post-APPROACH pose (the
parking pose perturbed by ±3 cm and ±5°) and ends 0.5 s into the lift. The target is a stone or a cube within ±10 cm
of its nominal pose, at any yaw, with one distractor in half of the scenes. The bottle is held out as an
out-of-distribution object. The expert succeeded in 99.5 % of 2,030 attempts. We kept 2,000 successful episodes
(about 28 steps each), converted them to a LeRobotDataset with 224×224 head images, a 10-D state and a 7-D
joint-target action, and split them 90/10 by episode.

## Policies

ACT: chunk 50, 25 executed steps, ResNet-18, 50k steps. DP: horizon 32, 8 executed steps, 2 observation steps,
DDIM with 10 inference steps, 60k steps. Both use lerobot's architectures and presets. Only the data loader was
replaced, because `lerobot-train` was data-bound at 3 it/s; our memmap loader of the same episodes runs at 13 it/s.
We scored every 5k-step checkpoint on 20 closed-loop validation episodes and kept the best: ACT at 50k and DP at
35k, both 20/20. Neither needed the fallback ladder (learning-rate change, state-only, shorter chunks).

## Results

The full-executor evaluation runs 30 episodes per cell, with identical scenes for every policy:

| Grasp success | C1 in-dist. | C2 ±20 cm | C3 bottle (OOD) | C4 distractor < 3 cm | false claims |
|---|---|---|---|---|---|
| Scripted | 29/30 | 30/30 | 29/30 | 30/30 | 0 |
| ACT | 29/30 | 26/30 | 0/30 | 30/30 | 0 |
| Diffusion Policy | 29/30 | 27/30 | 0/30 | 30/30 | 0 |

On C1 every policy loses the same trial, which never reaches GRASP. At the skill level (one grasp call from the
post-APPROACH state) all three score 30/30 on C1 with similar time-to-attach: 3.2 s for both learned policies against
3.0 s for the script. Neither the ACT chunk length (10/25/50 executed steps) nor the DP sampler (DDIM 5/10/50,
DDPM 50) changed C1 success. Longer ACT chunks grasp faster (2.7 s), and 5 DDIM steps are enough.

**Where learned matches the script:** in-distribution positions, and a distractor 0.3–3 cm from the target (0 wrong
objects out of 60 episodes). The target position in the state tells the policy which object to grasp.
**Where it loses:** positions beyond the ±10 cm training range (3–4 extra misses, clustered on the same far layouts
for both policies), and above all the unseen bottle. The learned policies reach for stone/cube height, 3.5 cm below
the bottle's centre, and never attach. The script simply reaches whatever point it is given, so it generalises for
free.

## The detected-failure point

The most useful result is not the success rate. Inside the full executor the learned skill was called 920 times;
746 of those calls missed, most of them in the bottle cell. **Not one** miss turned into a claimed success. Every GRASP
the executor reported as successful held the intended object (0 undetected failures in 240 episodes). When a learned
grasp missed, the same post-condition that guards the script (attachment, then a lift check) reported GRASP_MISSED.
The executor retried, which recovered 3 episodes, or the planner gave up honestly, with **0 false task claims**. The
verification never needed to know that a neural network was driving the arm.

## Limitations

The ground-truth target position is an input, so the policies learn the reach, not the perception. The "attach"
is the platform's weld within 5 cm, not a force-closure grasp, and the rollout trigger (2.5 cm from the scripted
grasp point) was tuned on validation episodes. Episodes are short (about 3 s), which makes chunking almost open-loop.
A 20-episode validation set cannot rank checkpoints that all sit at 95–100 %. The head camera barely sees the target
from the parking pose, which the state-only ablation tests. The OOD failure is structural: target height never
varied in training.

## What the interface made possible

Because the learned policy sits behind the grasp interface, the swap needed no change to the planner, the executor
logic, the evaluator or the trial files. We evaluated all three grasp implementations on byte-identical trials, with
the same post-conditions deciding success. The same interface also bounds the risk: a learned policy can fail, but it
cannot make the system lie about failing.
