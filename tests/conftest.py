# Owner: backbone (ALL)
"""Shared fixtures.  A single SimWorld is reused per test session (reset()
restores a reproducible state; creating GL contexts repeatedly is slow)."""

from __future__ import annotations

import pytest

from core.env import RobotEnv
from core.oracle import EvalOracle
from core.types import SceneConfig, SceneObjectSpec
from core.world import SimWorld


@pytest.fixture(scope="session")
def world():
    w = SimWorld()
    yield w
    w.close()


@pytest.fixture(scope="session")
def oracle(world):
    return EvalOracle(world)


@pytest.fixture()
def env(world):
    return RobotEnv(world)


def standard_scene(seed: int = 0) -> SceneConfig:
    return SceneConfig(
        seed=seed,
        objects=[
            SceneObjectSpec("stone", (0.40, -0.15, 0.88)),
            SceneObjectSpec("cube", (0.40, 0.15, 0.88)),
            SceneObjectSpec("bottle", (0.55, -0.35, 0.915)),
        ],
    )


@pytest.fixture()
def standard_world(world):
    """World reset to the standard scene and settled."""
    world.reset(standard_scene())
    world.step(int(round(1.0 / world.timestep)))  # 1 s settle
    return world
