"""Stage 0 check: lerobot import, CUDA, ACT + Diffusion configs instantiate."""
import torch, lerobot, mujoco, numpy
print("lerobot", lerobot.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "mujoco", mujoco.__version__, "numpy", numpy.__version__)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
print(type(ACTConfig()).__name__, "ok;", type(DiffusionConfig()).__name__, "ok")
