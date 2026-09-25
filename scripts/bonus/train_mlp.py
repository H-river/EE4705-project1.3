#!/usr/bin/env python
"""BC-MLP (bonus fallback): 3-layer MLP (q, target_pos) -> next q.

Reads the HDF5 demos directly (no images), trains on the same 90 % split
episodes as the lerobot policies, saves <out>/mlp.pt (arch, weights,
normalisation stats) for executor.learned_grasp.MLPPolicy.

    python scripts/bonus/train_mlp.py --dataset runs/bonus/lerobot/grasp_2k --out runs/bonus/train/mlp_2k
"""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def make_mlp(d_in: int = 10, d_out: int = 7, hidden: int = 256):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, d_out))


def load(split_root: pathlib.Path, which: str):
    split = json.loads((split_root / "split.json").read_text())
    xs, ys = [], []
    for k in split[which]:
        with h5py.File(ROOT / split["sources"][k]) as f:
            xs.append(np.concatenate([f["q"][:], f["target_b"][:]], 1))
            ys.append(f["action"][:])
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    x, y = load(args.dataset, "train")
    xv, yv = load(args.dataset, "val")
    stats = {"x_mean": x.mean(0), "x_std": x.std(0) + 1e-6, "y_mean": y.mean(0), "y_std": y.std(0) + 1e-6}
    X = torch.as_tensor((x - stats["x_mean"]) / stats["x_std"]).cuda()
    Y = torch.as_tensor((y - stats["y_mean"]) / stats["y_std"]).cuda()
    Xv = torch.as_tensor((xv - stats["x_mean"]) / stats["x_std"]).cuda()
    Yv = torch.as_tensor((yv - stats["y_mean"]) / stats["y_std"]).cuda()
    net = make_mlp().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(X), (256,), device="cuda", generator=g)
        loss = torch.nn.functional.mse_loss(net(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 5000 == 0:
            with torch.no_grad():
                vl = torch.nn.functional.mse_loss(net(Xv), Yv).item()
            print(f"step {step} train {loss.item():.4f} val {vl:.4f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": {"d_in": 10, "d_out": 7, "hidden": 256}, "state_dict": net.cpu().state_dict(),
                "stats": {k: v.tolist() for k, v in stats.items()}, "seed": args.seed}, args.out / "mlp.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
