"""Stage 2 check: one ACT-style batch (50-step action chunk) loads from the LeRobotDataset."""
import json, sys, pathlib, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root = pathlib.Path(sys.argv[1])
split = json.loads((root / "split.json").read_text())
ds = LeRobotDataset(f"local/{root.name}", root=root, episodes=split["train"],
                    delta_timestamps={"action": [i / 10 for i in range(50)]})
b = next(iter(torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)))
for k, v in b.items():
    if hasattr(v, "shape"):
        print(k, tuple(v.shape), v.dtype)
print("frames", ds.num_frames, "episodes", ds.num_episodes)
