#!/usr/bin/env python
"""Bonus stages 3/4: train a lerobot ACT / Diffusion policy with a fast data path.

Why not ``lerobot-train``: on this machine its LeRobotDataset loader (per-sample
chunk queries on the parquet dataset) delivered ~3 steps/s at batch 32 with
the GPU ~idle (data_s 0.29 s vs update 0.07 s), i.e. ~4.6 h per 50k steps.
This trainer keeps everything that defines the model identical to
lerobot-train — ``make_policy`` with the dataset metadata, the policy's own
optimizer/scheduler presets and grad clipping, ``make_pre_post_processors``
with the dataset stats (images: ImageNet stats, as ``use_imagenet_stats``),
the policy's ``observation_delta_indices`` / ``action_delta_indices`` with
episode-boundary clamping and ``*_is_pad`` masks — and only replaces the
data source by a uint8 memmap of the SAME LeRobotDataset episodes
(``--verify`` checks cache == LeRobotDataset on random frames).

Checkpoints: <out>/checkpoints/<step>/pretrained_model (policy + processors,
loadable by executor.learned_grasp) + training_state.pt (resume).

    python scripts/bonus/train_policy.py --out runs/bonus/train/act_base --kind act --steps 50000 \
        --dataset runs/bonus/lerobot/grasp_2k [--state-only] [--lr 5e-5] [--chunk 25] ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import h5py
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMG = 224


# ----------------------------------------------------------------------------- cache

def build_cache(ds_root: pathlib.Path) -> pathlib.Path:
    """uint8 image memmap + state/action/episode arrays in LeRobot episode order."""
    cache = ROOT / "runs/bonus/cache" / ds_root.name
    if (cache / "meta.json").exists():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    split = json.loads((ds_root / "split.json").read_text())
    lens = []
    for src in split["sources"]:
        with h5py.File(ROOT / src) as f:
            lens.append(len(f["q"]))
    n = int(sum(lens))
    img = np.lib.format.open_memmap(cache / "images.npy", mode="w+", dtype=np.uint8, shape=(n, IMG, IMG, 3))
    state = np.zeros((n, 10), np.float32)
    action = np.zeros((n, 7), np.float32)
    episode = np.zeros(n, np.int32)
    k = 0
    for e, src in enumerate(split["sources"]):
        with h5py.File(ROOT / src) as f:
            m = len(f["q"])
            img[k:k + m] = f["image"][:]
            state[k:k + m] = np.concatenate([f["q"][:], f["target_b"][:]], 1)
            action[k:k + m] = f["action"][:]
            episode[k:k + m] = e
        k += m
    img.flush()
    np.save(cache / "state.npy", state)
    np.save(cache / "action.npy", action)
    np.save(cache / "episode.npy", episode)
    (cache / "meta.json").write_text(json.dumps({"frames": n, "episodes": len(lens), "dataset": str(ds_root)}))
    return cache


class CacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache: pathlib.Path, episodes: list[int], obs_delta, act_delta, use_image: bool,
                 split_env_state: bool = False):
        self.cache = cache
        self.state = np.load(cache / "state.npy")
        self.action = np.load(cache / "action.npy")
        ep = np.load(cache / "episode.npy")
        starts = np.searchsorted(ep, np.arange(ep.max() + 1), side="left")
        ends = np.searchsorted(ep, np.arange(ep.max() + 1), side="right")  # exclusive
        self.start, self.end = starts[ep], ends[ep]
        keep = np.isin(ep, np.asarray(episodes))
        self.frames = np.nonzero(keep)[0]
        self.obs_delta = None if obs_delta is None else np.asarray(obs_delta)
        self.act_delta = np.asarray(act_delta)
        self.use_image = use_image
        self.split_env_state = split_env_state  # state-only: q -> observation.state, target -> environment_state
        self._img = None

    def __len__(self):
        return len(self.frames)

    def _idx(self, t, delta):
        raw = t + delta
        s, e = self.start[t], self.end[t]
        return np.clip(raw, s, e - 1), (raw < s) | (raw >= e)

    def __getitem__(self, i):
        if self.use_image and self._img is None:
            self._img = np.load(self.cache / "images.npy", mmap_mode="r")
        t = int(self.frames[i])
        out = {}
        ai, apad = self._idx(t, self.act_delta)
        out["action"] = torch.from_numpy(self.action[ai])
        out["action_is_pad"] = torch.from_numpy(apad)
        if self.obs_delta is None:
            out["observation.state"] = torch.from_numpy(self.state[t])
            if self.use_image:
                out["observation.image"] = torch.from_numpy(np.array(self._img[t])).permute(2, 0, 1)
        else:
            oi, opad = self._idx(t, self.obs_delta)
            out["observation.state"] = torch.from_numpy(self.state[oi])
            out["observation.state_is_pad"] = torch.from_numpy(opad)
            if self.use_image:
                out["observation.image"] = torch.from_numpy(np.array(self._img[oi])).permute(0, 3, 1, 2)
                out["observation.image_is_pad"] = torch.from_numpy(opad)
        if self.split_env_state:
            st = out.pop("observation.state")
            out["observation.state"] = st[..., :7].contiguous()
            out["observation.environment_state"] = st[..., 7:].contiguous()
            if "observation.state_is_pad" in out:
                out["observation.environment_state_is_pad"] = out["observation.state_is_pad"]
        out["index"] = t
        return out


def verify(cache: pathlib.Path, ds_root: pathlib.Path, n: int = 20) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f"local/{ds_root.name}", root=ds_root)
    img = np.load(cache / "images.npy", mmap_mode="r")
    state, action = np.load(cache / "state.npy"), np.load(cache / "action.npy")
    rng = np.random.default_rng(0)
    for t in rng.integers(0, len(state), n):
        item = ds[int(t)]
        assert np.allclose(item["observation.state"].numpy(), state[t], atol=1e-6), t
        assert np.allclose(item["action"].numpy(), action[t], atol=1e-6), t
        ref = (item["observation.image"].numpy() * 255).round().astype(np.uint8).transpose(1, 2, 0)
        assert np.abs(ref.astype(int) - img[t].astype(int)).max() <= 1, t
    print(f"verify: {n} random frames identical in cache and LeRobotDataset", flush=True)


# ----------------------------------------------------------------------------- policy

def make_config(args, meta):
    from lerobot.configs.types import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features
    if args.kind == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        cfg = ACTConfig(chunk_size=args.chunk, n_action_steps=args.n_action_steps, vision_backbone="resnet18",
                        device="cuda", push_to_hub=False)
        if args.lr:
            cfg.optimizer_lr = args.lr
    else:
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        cfg = DiffusionConfig(horizon=args.horizon, n_action_steps=args.n_action_steps, n_obs_steps=args.n_obs_steps,
                              noise_scheduler_type="DDIM", num_inference_steps=args.inference_steps,
                              device="cuda", push_to_hub=False)
        if args.lr:
            cfg.optimizer_lr = args.lr
    feats = dataset_to_policy_features(meta.features)
    cfg.output_features = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    cfg.input_features = {k: f for k, f in feats.items() if k not in cfg.output_features
                          and not (args.state_only and f.type is FeatureType.VISUAL)}
    if args.state_only:
        # lerobot ACT/DP need an image or an environment state: the target position IS environment state.
        from lerobot.configs.types import PolicyFeature
        cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                              "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,))}
    return cfg


def split_state_stats(stats: dict) -> dict:
    """10-D observation.state stats -> 7-D state + 3-D environment_state (same numbers, sliced)."""
    st = stats.pop("observation.state")
    stats["observation.state"] = {k: (v[:7] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == 10 else v)
                                  for k, v in st.items()}
    stats["observation.environment_state"] = {k: (v[7:] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == 10 else v)
                                              for k, v in st.items()}
    return stats


def save(out: pathlib.Path, step: int, policy, pre, post, opt, sched, args) -> None:
    d = out / "checkpoints" / f"{step:06d}"
    pm = d / "pretrained_model"
    pm.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(pm)
    pre.save_pretrained(pm)
    post.save_pretrained(pm)
    (pm / "train_args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    torch.save({"step": step, "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, d / "training_state.pt")
    last = out / "checkpoints" / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(d.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--kind", choices=("act", "diffusion"), required=True)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--state-only", action="store_true")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--n-obs-steps", type=int, default=2)
    ap.add_argument("--inference-steps", type=int, default=10)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.batch is None:
        args.batch = 32 if args.kind == "act" else 64
    if args.n_action_steps is None:
        args.n_action_steps = 25 if args.kind == "act" else 8
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import IMAGENET_STATS

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache = build_cache(args.dataset)
    if args.verify:
        verify(cache, args.dataset)
    meta = LeRobotDatasetMetadata(f"local/{args.dataset.name}", root=args.dataset)
    stats = {k: {s: torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for s, v in d.items()}
             for k, d in meta.stats.items()}
    for s, v in IMAGENET_STATS.items():
        stats["observation.image"][s] = torch.tensor(v, dtype=torch.float32)
    cfg = make_config(args, meta)
    if args.state_only:
        stats = split_state_stats(stats)
    split = json.loads((args.dataset / "split.json").read_text())
    use_image = "observation.image" in cfg.input_features
    ds = CacheDataset(cache, split["train"], cfg.observation_delta_indices, cfg.action_delta_indices, use_image,
                      split_env_state=args.state_only)
    g = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, sampler=torch.utils.data.RandomSampler(ds, replacement=True,
                                                                          num_samples=10**9, generator=g),
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
        worker_init_fn=lambda w: np.random.seed(args.seed + w))
    policy = make_policy(cfg, ds_meta=meta)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    opt_cfg = cfg.get_optimizer_preset()
    opt = opt_cfg.build(policy.get_optim_params())
    sched_cfg = cfg.get_scheduler_preset()
    sched = sched_cfg.build(opt, args.steps) if sched_cfg is not None else None
    step = 0
    last = args.out / "checkpoints" / "last"
    if last.exists():  # resume
        from safetensors.torch import load_file
        policy.load_state_dict(load_file(last / "pretrained_model" / "model.safetensors"), strict=False)
        st = torch.load(last / "training_state.pt", weights_only=False)
        opt.load_state_dict(st["opt"])
        if sched and st["sched"]:
            sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state(st["cuda_rng"])
        step = st["step"]
        g.manual_seed(args.seed + step)
        print(f"resumed at step {step}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    policy.train()
    it = iter(loader)
    t0, tl, losses = time.time(), time.time(), []
    log = (args.out / "train_log.jsonl").open("a")
    while step < args.steps:
        batch = next(it)
        batch = {k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        if use_image:
            batch["observation.image"] = batch["observation.image"].float().div_(255.0)
        batch["task"] = ["grasp the target object"] * args.batch
        batch = pre(batch)
        loss, _ = policy.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if opt_cfg.grad_clip_norm and opt_cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), opt_cfg.grad_clip_norm)
        opt.step()
        if sched:
            sched.step()
        step += 1
        losses.append(loss.item())
        if step % 500 == 0:
            rec = {"step": step, "loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
                   "steps_per_s": 500 / (time.time() - tl), "elapsed_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            losses, tl = [], time.time()
        if step % args.save_freq == 0 or step == args.steps:
            save(args.out, step, policy, pre, post, opt, sched, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
