# Bonus: learned grasping behind the skill interface

## Setup

Can a learned policy replace the scripted grasp without touching the rest of the system? We trained **ACT** and
**Diffusion Policy (DP)** with lerobot 0.6.1 and put each behind the existing interface: `LearnedGraspSkill(policy,
ckpt)(env, pos)` takes the same call as `skills.grasp`, selected by `EE4705_GRASP_POLICY`. From the post-APPROACH
pose the policy reads the arm joints, the target position in the base frame and the head camera at 10 Hz, commands
the next joint setpoint for at most 15 s, then calls `try_attach_near_ee()` once, as the script does. The planner,
the executor's post-conditions and the evaluator are unchanged. Perception is ground truth and the planner is rule
based: 0 API calls.

## Data and policies

We collected 2,000 scripted demonstrations (expert 99.5 %) from the post-APPROACH pose to 0.5 s into the lift. The
target is a stone or a cube within ±10 cm, at any yaw, with a distractor in half of the scenes; the bottle is held
out. ACT used chunk 50 with 25 executed steps; DP used horizon 32, 8 executed steps and DDIM with 10 steps. Both
use a ResNet-18, lerobot's architectures and presets, and a memmap copy of the same LeRobotDataset (13 it/s against
`lerobot-train`'s data-bound 3 it/s). Each 5k checkpoint was scored on 20 closed-loop validation episodes: ACT
peaked at 50k and DP at 35k, both 20/20.

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

## Follow-ups: data, coverage and a learned PLACE

**More data** did not help: ACT trained on 500, 1k, 2k and 5k demos is flat on C1 (29–30/30) and gains only one
C2 episode. **Coverage** did. Adding 503 bottle demos lifted the bottle cell from 0 to 26/30 with no in-distribution
loss, and full `final50` in manipulation mode rose from 38 to 47/50 (script: 48). Adding ±20 cm demos then fixed the
position shift (C2 skill 26 → 28/30) but diluted the bottle share and cost bottle trials (net 44/50), so it was
not adopted. The same recipe learned PLACE's carry: 29/30 on C1, as good as the script and slightly closer to the
region centre (0.33 vs 0.45 cm); with far-layout demos it also matches the script on C2 (29/30). With both skills
learned, `final50` reaches 43/50, and a place policy that is better in isolation (C2 30/30, bottle 28/30) dropped to
38/50 end to end: after a learned grasp its carry stopped 1.6–2.3 cm short, and the executor's unchanged 1.2 cm check
rejected it. Seed variance was ±1–2 episodes across three ACT seeds. None of these variants produced a false claim.

## Limitations

The policies receive the ground-truth target position, so they learn the reach, not perception. "Attach" is the
platform's 5 cm weld, not force closure. The rollout stop rule (within 2.5 cm of the scripted grasp point, or
settled within 5 cm) was tuned on validation episodes. Episodes last about 3 s, so chunking is nearly open-loop.
Twenty validation episodes cannot rank checkpoints at 95–100 %. What the policies generalise to is exactly what
the demos cover.

## What the interface made possible

The swap changed no planner, executor, evaluator or trial file. We evaluated three grasp implementations on
identical trials with the same post-conditions deciding success. The interface also bounds the risk: a learned
policy can fail, but it cannot make the system lie about failing.
