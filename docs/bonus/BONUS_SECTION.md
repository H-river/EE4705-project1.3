# Bonus: learned grasping behind the skill interface

## Setup

Can a learned visuomotor policy replace the scripted grasp without touching the rest of the system? We trained
**ACT** and **Diffusion Policy (DP)** with lerobot 0.6.1 and put each one behind the existing grasp interface:
`LearnedGraspSkill(policy, ckpt)(env, pos)` has the same call as `skills.grasp(env, pos)`, and the executor selects
it with `EE4705_GRASP_POLICY`. From the post-APPROACH pose the policy runs at 10 Hz for at most 15 s. It observes the
arm joints, the target position in the base frame and the head camera, and outputs the next arm joint setpoint. It
then calls `try_attach_near_ee()` once, exactly as the script does. The executor's lift check, the planner, the
verification and the evaluator are unchanged. Perception is ground truth and the planner is the rule planner, so
every experiment costs 0 API calls.

## Data

The expert is the scripted APPROACH→REACH→GRASP pipeline. Each episode runs from the post-APPROACH pose (parking
perturbed by ±3 cm and ±5°) to 0.5 s into the lift. The target is a stone or a cube within ±10 cm of its nominal
pose, at any yaw, with a distractor in half of the scenes. The bottle is held out as the OOD object. The expert
succeeded in 99.5 % of 2,030 attempts. We kept 2,000 episodes (about 28 steps each) as a LeRobotDataset: 224² head
image, 10-D state, 7-D joint-target action, split 90/10 by episode.

## Policies

ACT: chunk 50, 25 executed steps, ResNet-18, 50k steps. DP: horizon 32, 8 executed steps, 2 observation steps,
DDIM with 10 steps, 60k steps. We kept lerobot's architectures and presets and replaced only the data loader:
`lerobot-train` was data-bound at 3 it/s, and a memmap of the same episodes runs at 13 it/s. Scoring each 5k
checkpoint on 20 closed-loop validation episodes selected ACT at 50k and DP at 35k, both 20/20. Neither needed the
fallback ladder.

## Results

Full executor, 30 episodes per cell, identical scenes for every policy:

| Grasp success | C1 in-dist. | C2 ±20 cm | C3 bottle (OOD) | C4 distractor < 3 cm | false claims |
|---|---|---|---|---|---|
| Scripted | 29/30 | 30/30 | 29/30 | 30/30 | 0 |
| ACT | 29/30 | 26/30 | 0/30 | 30/30 | 0 |
| Diffusion Policy | 29/30 | 27/30 | 0/30 | 30/30 | 0 |

The shared C1 miss never reaches GRASP. At skill level, one grasp call from the post-APPROACH state gives all three
30/30 on C1, with time-to-attach of 3.2 s (learned) against 3.0 s (script). The ACT chunk length and the DP sampler
(DDIM 5/10/50, DDPM 50) do not change C1 success. The image matters for ACT: state-only ACT drops to 26/30 at skill
level. It does not matter for DP: state-only DP stays at 30/30 and is faster.

**Where learned matches the script:** in-distribution positions, and a distractor 0.3–3 cm from the target (0 wrong
objects in 60 episodes). The target position in the state selects the object.
**Where it loses:** positions beyond the ±10 cm training range, where both policies miss on the same far layouts,
and above all the unseen bottle. Target height never varied in training, so the policies reach for stone/cube
height, 3.5 cm below the bottle's centre. The script reaches whatever point it is given and generalises for free.

## The detected-failure point

Inside the full executor the learned skill was called 920 times, and 746 of those calls missed (mostly on the
bottle). **Not one** miss became a claimed success. Every GRASP the executor reported as successful held the
intended object (0 undetected failures in 240 episodes). The post-condition that guards the script (attachment,
then a lift check) reported every learned miss as GRASP_MISSED. The executor then retried, recovering 3 episodes,
or gave up honestly: **0 false task claims**. The verification never needed to know that a network drove the arm.

## Limitations

The policies receive the ground-truth target position, so they learn the reach, not perception. "Attach" is the
platform's 5 cm weld, not force closure. The rollout stop rule (within 2.5 cm of the scripted grasp point, or
settled within 5 cm) was tuned on validation episodes. Episodes last about 3 s, so chunking is nearly open-loop.
Twenty validation episodes cannot rank checkpoints at 95–100 %. The OOD failure is structural.

## What the interface made possible

The swap changed no planner, executor, evaluator or trial file. We evaluated three grasp implementations on
identical trials with the same post-conditions deciding success. The interface also bounds the risk: a learned
policy can fail, but it cannot make the system lie about failing.
