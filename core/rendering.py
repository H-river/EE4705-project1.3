# Owner: backbone (ALL)
"""Headless-rendering GL backend selection.

MuJoCo picks its GL backend from the MUJOCO_GL environment variable when the
`mujoco` module is first imported, so this module must be imported (or the
variable set) BEFORE `import mujoco` anywhere in the process.  All backbone
modules import mujoco via `from core.rendering import mujoco`.

Tested working backend on the reference machine: EGL (NVIDIA driver).
Override with e.g. `MUJOCO_GL=osmesa` in the environment if EGL is
unavailable; an unset variable defaults to "egl" here.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402  (must follow the env var)

__all__ = ["mujoco"]
