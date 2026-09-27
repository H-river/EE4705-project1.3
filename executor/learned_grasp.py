"""Learned grasp skill (bonus): ACT / Diffusion Policy / BC-MLP behind the
existing grasp interface.

``LearnedGraspSkill(policy, ckpt)(env, pos_world)`` has the same call and
return type as ``core.skills.grasp(env, pos_world)``.  From the current
(post-APPROACH) pose it runs the policy closed-loop at 10 Hz for at most
15 s of sim time; each tick it reads proprioception (arm q), the target
position in the base frame and, for image policies, the head camera, and
commands the next arm joint setpoint.  The rollout ends when the TCP comes
within ``ATTACH_TOL`` of the scripted grasp point (checked every 20 ms, so a
policy that passes through the point without stopping is caught), when the
commanded motion has settled, or at the time limit; then
``env.try_attach_near_ee()`` is called ONCE, exactly as the script does.

ATTACH_TOL (v2, 2026-09-25): v1 used ``skills.EE_POS_TOL`` (1.2 cm), the
scripted REACH tolerance.  The script converges onto the point; the learned
policies (like the demos they imitate) descend, pause briefly and lift, and
at ACT 5k 10/12 VAL misses came within 1.3-2 cm and then lifted away.  2.5 cm
from the grasp point (2 cm above the object centre) keeps the TCP within
4.5 cm of the centre, i.e. inside the 5 cm ATTACH_RADIUS used by
try_attach_near_ee, so the trigger never fires where attaching is
geometrically impossible.

Settle stop (v3, same day): v2 also stopped whenever the commanded q stayed
still for 0.5 s after the first second.  At ACT 15k/25k the same three VAL
episodes "settled" 19-22 cm from the target; with the stop disabled the
policy resumed after a 1-2 s pause and all three attached.  The settle stop
now only applies once the TCP is within SETTLE_NEAR of the grasp point (a
pause there is final); a pause farther away runs on until the time limit.  Everything after that (lift check, the
executor's post-conditions) is unchanged.

Selection: ``EE4705_GRASP_POLICY`` in {scripted, act, diffusion, mlp}
(default scripted), ``EE4705_GRASP_CKPT`` (default
``runs/bonus/best/<policy>``) and optional ``EE4705_GRASP_KWARGS`` (JSON,
e.g. ``{"n_action_steps": 10}`` or ``{"num_inference_steps": 50,
"scheduler": "DDPM"}`` for ablations); see ``grasp_primitive()``.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
from typing import Callable, Optional, Protocol

import numpy as np

from core import skills
from core.types import ErrorCode, SkillResult

ROOT = pathlib.Path(__file__).resolve().parents[1]
HZ = 10.0
MAX_S = 15.0
IMG = 224
SETTLE_RAD = 0.01  # max |Δq_cmd| per tick counted as "settled"
SETTLE_TICKS = 5  # 0.5 s
MIN_TICKS = 10  # never stop on "settled" in the first second
ATTACH_TOL = 0.025  # m from the grasp point (see module docstring, v2)
SUBSTEPS = 10  # physics steps between proximity checks (20 ms)
SETTLE_NEAR = 0.05  # m: "settled" only ends the rollout this close to the grasp point (v3)
POLICIES = ("scripted", "act", "diffusion", "mlp")


class ArmPolicy(Protocol):
    needs_image: bool

    def reset(self) -> None: ...

    def __call__(self, state: np.ndarray, image: Optional[np.ndarray]) -> np.ndarray: ...


def target_in_base(pos_world: np.ndarray, base: np.ndarray) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(pos_world[:2], float) - np.asarray(base[:2], float)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], float(pos_world[2])])


def head_image(env) -> np.ndarray:
    import cv2
    return cv2.resize(env.get_obs("head").rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)


class LeRobotPolicy:
    """A lerobot ACT / Diffusion checkpoint (``pretrained_model`` dir)."""

    def __init__(self, kind: str, ckpt: str | os.PathLike, device: Optional[str] = None,
                 n_action_steps: Optional[int] = None, num_inference_steps: Optional[int] = None,
                 scheduler: Optional[str] = None, seed: int = 0) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        if kind == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy as cls
        elif kind == "diffusion":
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as cls
        else:
            raise ValueError(kind)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from lerobot.configs.policies import PreTrainedConfig
        pcfg = PreTrainedConfig.from_pretrained(str(ckpt))
        pcfg.device = self.device  # load straight onto the requested device (cpu evals while training)
        self.policy = cls.from_pretrained(str(ckpt), config=pcfg)
        cfg = self.policy.config
        if n_action_steps is not None:
            cfg.n_action_steps = int(n_action_steps)
        if kind == "diffusion" and (num_inference_steps is not None or scheduler is not None):
            self._set_diffusion_sampler(num_inference_steps, scheduler)
        self.policy.to(self.device).eval()
        self.needs_image = bool(cfg.image_features)
        self.env_state = "observation.environment_state" in cfg.input_features
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": self.device}})
        self.seed, self.calls = int(seed), 0
        self.reset()

    def _set_diffusion_sampler(self, steps: Optional[int], scheduler: Optional[str]) -> None:
        from diffusers import DDIMScheduler, DDPMScheduler
        m = self.policy.diffusion
        cfg = self.policy.config
        kind = scheduler or cfg.noise_scheduler_type
        klass = DDIMScheduler if kind == "DDIM" else DDPMScheduler
        m.noise_scheduler = klass.from_config(m.noise_scheduler.config)
        cfg.noise_scheduler_type = kind
        m.num_inference_steps = int(steps or cfg.num_inference_steps or cfg.num_train_timesteps)
        cfg.num_inference_steps = m.num_inference_steps

    def set_episode_seed(self, seed: int) -> None:
        """Diffusion sampling noise comes from torch's global RNG: re-seed it
        per rollout so evaluations are reproducible (R6)."""
        self.seed, self.calls = int(seed), 0

    def reset(self) -> None:
        self.torch.manual_seed(self.seed * 1000 + self.calls)  # one fixed seed per rollout
        self.calls += 1
        self.policy.reset()

    def __call__(self, state: np.ndarray, image: Optional[np.ndarray]) -> np.ndarray:
        t = self.torch
        st = t.from_numpy(state.astype(np.float32))[None]
        if self.env_state:  # state-only checkpoints: q -> state, target -> environment_state
            obs = {"observation.state": st[:, :7], "observation.environment_state": st[:, 7:]}
        else:
            obs = {"observation.state": st}
        obs["task"] = ["grasp the target object"]
        if self.needs_image:
            obs["observation.image"] = t.from_numpy(image).permute(2, 0, 1)[None].float() / 255.0
        with t.no_grad():
            batch = self.pre(obs)
            action = self.policy.select_action(batch)
            action = self.post(action)
        return action.squeeze(0).detach().cpu().numpy().astype(float)


class MLPPolicy:
    """BC-MLP fallback: (q, target) -> next q (see scripts/bonus/train_mlp.py)."""

    needs_image = False

    def __init__(self, ckpt: str | os.PathLike, device: Optional[str] = None) -> None:
        import torch
        self.torch = torch
        blob = torch.load(pathlib.Path(ckpt) / "mlp.pt", map_location="cpu", weights_only=False)
        from scripts.bonus.train_mlp import make_mlp  # noqa: E402 (repo root on sys.path)
        self.net = make_mlp(**blob["arch"])
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        self.stats = {k: torch.as_tensor(v) for k, v in blob["stats"].items()}

    def reset(self) -> None:
        pass

    def __call__(self, state: np.ndarray, image: Optional[np.ndarray]) -> np.ndarray:
        t, s = self.torch, self.stats
        x = (t.as_tensor(state, dtype=t.float32) - s["x_mean"]) / s["x_std"]
        with t.no_grad():
            y = self.net(x[None])[0]
        return (y * s["y_std"] + s["y_mean"]).numpy().astype(float)


class LearnedGraspSkill:
    """Drop-in for ``skills.grasp(env, pos_world)`` driven by a learned policy."""

    def __init__(self, policy: ArmPolicy | str, ckpt: Optional[str | os.PathLike] = None,
                 max_s: float = MAX_S, **policy_kwargs) -> None:
        if isinstance(policy, str):
            policy = load_policy(policy, ckpt, **policy_kwargs)
        self.policy = policy
        self.max_s = float(max_s)
        self.last_info: dict = {}

    def __call__(self, env, pos_world: np.ndarray, timeout_s: Optional[float] = None) -> SkillResult:
        if env.is_attached():
            return SkillResult(False, ErrorCode.ALREADY_HOLDING, {"primitive": "grasp"})
        p = np.asarray(pos_world, dtype=float)
        grasp_pt = p + np.array([0.0, 0.0, skills.GRASP_DESCEND_OFFSET])
        every = max(1, int(round(1.0 / HZ / env.timestep())))
        max_ticks = int(round(min(self.max_s, timeout_s or self.max_s) * HZ))
        self.policy.reset()
        t0 = env.sim_time()
        prev_cmd, settled, ticks, stop = None, 0, 0, "timeout"
        min_dist = float("inf")
        for ticks in range(1, max_ticks + 1):
            q = np.asarray(env.get_arm_q(), dtype=float)
            state = np.concatenate([q, target_in_base(p, env.get_base_pose())])
            image = head_image(env) if self.policy.needs_image else None
            cmd = np.asarray(self.policy(state, image), dtype=float)
            if cmd.shape != (7,) or not np.all(np.isfinite(cmd)):
                return SkillResult(False, ErrorCode.UNREACHABLE, {"primitive": "grasp", "detail": "invalid policy action"})
            env.set_arm_joint_target(cmd)
            done = 0
            while done < every:
                k = min(SUBSTEPS, every - done)
                env.step(k)
                done += k
                dist = float(np.linalg.norm(env.get_ee_pos() - grasp_pt))
                min_dist = min(min_dist, dist)
                if dist < ATTACH_TOL:
                    stop = "reached"
                    break
            if stop == "reached":
                break
            if prev_cmd is not None and float(np.max(np.abs(cmd - prev_cmd))) < SETTLE_RAD:
                settled += 1
            else:
                settled = 0
            prev_cmd = cmd
            if ticks >= MIN_TICKS and settled >= SETTLE_TICKS and dist < SETTLE_NEAR:
                stop = "settled"
                break
        info = {"primitive": "grasp", "policy": type(self.policy).__name__, "stop": stop, "ticks": ticks,
                "rollout_s": env.sim_time() - t0, "ee_error": float(np.linalg.norm(env.get_ee_pos() - grasp_pt)),
                "min_ee_error": min_dist}
        handle = env.try_attach_near_ee()
        info["attached"] = handle is not None
        self.last_info = info
        log = os.environ.get("EE4705_GRASP_LOG")
        if log:  # optional sidecar (the executor replaces primitive info after its lift check)
            with open(log, "a") as f:
                f.write(json.dumps({**info, "sim_time": env.sim_time()}) + "\n")
        if handle is None:
            return SkillResult(False, ErrorCode.GRASP_MISSED, info)
        env.step(50)  # let the attachment stabilize (as skills.grasp)
        return SkillResult(True, ErrorCode.NONE, {**info, "attachment": handle})


class LearnedPlaceSkill:
    """Stage 8.3: drop-in for the ``skills.move_to(env, release_pos)`` call in
    PLACE (carry the held object from the post-MOVE_TO pose down to the
    release pose).  The release itself (open + ``skills.place``) and the
    placement verification stay scripted and unchanged.  The rollout ends
    when the TCP is within ``skills.EE_POS_TOL`` of the release pose (the
    executor rejects anything farther, unchanged), when the commanded motion
    settles within SETTLE_NEAR, or at the time limit."""

    def __init__(self, policy: ArmPolicy | str, ckpt: Optional[str | os.PathLike] = None,
                 max_s: float = MAX_S, **policy_kwargs) -> None:
        if isinstance(policy, str):
            policy = load_policy(policy, ckpt, **policy_kwargs)
        self.policy = policy
        self.max_s = float(max_s)
        self.last_info: dict = {}

    def __call__(self, env, pos_world: np.ndarray, timeout_s: Optional[float] = None) -> SkillResult:
        p = np.asarray(pos_world, dtype=float)
        every = max(1, int(round(1.0 / HZ / env.timestep())))
        max_ticks = int(round(min(self.max_s, timeout_s or self.max_s) * HZ))
        self.policy.reset()
        t0 = env.sim_time()
        prev_cmd, settled, ticks, stop, min_dist, dist = None, 0, 0, "timeout", float("inf"), float("inf")
        for ticks in range(1, max_ticks + 1):
            q = np.asarray(env.get_arm_q(), dtype=float)
            state = np.concatenate([q, target_in_base(p, env.get_base_pose())])
            image = head_image(env) if self.policy.needs_image else None
            cmd = np.asarray(self.policy(state, image), dtype=float)
            if cmd.shape != (7,) or not np.all(np.isfinite(cmd)):
                return SkillResult(False, ErrorCode.UNREACHABLE, {"primitive": "move_to", "detail": "invalid policy action"})
            env.set_arm_joint_target(cmd)
            done = 0
            while done < every:
                k = min(SUBSTEPS, every - done)
                env.step(k)
                done += k
                dist = float(np.linalg.norm(env.get_ee_pos() - p))
                min_dist = min(min_dist, dist)
                if dist < skills.EE_POS_TOL:
                    stop = "reached"
                    break
            if stop == "reached":
                break
            if prev_cmd is not None and float(np.max(np.abs(cmd - prev_cmd))) < SETTLE_RAD:
                settled += 1
            else:
                settled = 0
            prev_cmd = cmd
            if ticks >= MIN_TICKS and settled >= SETTLE_TICKS and dist < SETTLE_NEAR:
                stop = "settled"
                break
        env.set_arm_joint_target(np.asarray(env.get_arm_q(), dtype=float))  # hold here for the release
        info = {"primitive": "move_to", "policy": type(self.policy).__name__, "stop": stop, "ticks": ticks,
                "rollout_s": env.sim_time() - t0, "ee_error": float(np.linalg.norm(env.get_ee_pos() - p)),
                "min_ee_error": min_dist, "attached": env.is_attached()}
        self.last_info = info
        log = os.environ.get("EE4705_PLACE_LOG")
        if log:
            with open(log, "a") as f:
                f.write(json.dumps({**info, "sim_time": env.sim_time()}) + "\n")
        if info["ee_error"] < skills.EE_POS_TOL:
            return SkillResult(True, ErrorCode.NONE, info)
        return SkillResult(False, ErrorCode.UNREACHABLE if stop != "timeout" else ErrorCode.TIMEOUT, info)


def place_primitive() -> Callable:
    """The PLACE carry primitive selected by EE4705_PLACE_POLICY (default: skills.move_to)."""
    kind = os.environ.get("EE4705_PLACE_POLICY", "scripted").strip().lower() or "scripted"
    if kind not in POLICIES:
        raise ValueError(f"EE4705_PLACE_POLICY={kind!r}; expected one of {POLICIES}")
    if kind == "scripted":
        return skills.move_to
    key = ("place", kind, os.environ.get("EE4705_PLACE_CKPT", ""), os.environ.get("EE4705_PLACE_KWARGS", ""))
    if key not in _CACHE:
        kwargs = json.loads(key[3]) if key[3] else {}
        ckpt = key[2] or str(ROOT / "runs/bonus/best" / f"place_{kind}")
        _CACHE[key] = LearnedPlaceSkill(kind, ckpt, **kwargs)
    return _CACHE[key]


def load_policy(kind: str, ckpt: Optional[str | os.PathLike] = None, **kwargs) -> ArmPolicy:
    ckpt = pathlib.Path(ckpt) if ckpt else ROOT / "runs/bonus/best" / kind
    if kind in ("act", "diffusion"):
        return LeRobotPolicy(kind, ckpt, **kwargs)
    if kind == "mlp":
        return MLPPolicy(ckpt, **kwargs)
    raise ValueError(f"unknown learned policy {kind!r}; expected one of {POLICIES[1:]}")


_CACHE: dict[tuple, Callable] = {}


def grasp_primitive() -> Callable:
    """The grasp primitive selected by EE4705_GRASP_POLICY (default: the script)."""
    kind = os.environ.get("EE4705_GRASP_POLICY", "scripted").strip().lower() or "scripted"
    if kind not in POLICIES:
        raise ValueError(f"EE4705_GRASP_POLICY={kind!r}; expected one of {POLICIES}")
    if kind == "scripted":
        return skills.grasp
    key = (kind, os.environ.get("EE4705_GRASP_CKPT", ""), os.environ.get("EE4705_GRASP_KWARGS", ""))
    if key not in _CACHE:
        kwargs = json.loads(key[2]) if key[2] else {}
        _CACHE[key] = LearnedGraspSkill(kind, key[1] or None, **kwargs)
    return _CACHE[key]
