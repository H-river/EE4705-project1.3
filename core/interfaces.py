# Owner: backbone (ALL)
"""Abstract interfaces implemented by Student A/B/C modules and by mocks.

These depend only on core.types (no concrete implementations), so any module
may import them without pulling in MuJoCo, the oracle, or another student's
code.  The executor receives the public RobotEnv protocol, which exposes no
ground-truth state.
"""

from __future__ import annotations

import abc
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from core.types import (
    Action,
    ExecutionResult,
    GroundedObject,
    Observation,
    Plan,
    SceneDescription,
    ExecutionContext,
)


@runtime_checkable
class RobotEnvProtocol(Protocol):
    """Public environment surface available to the executor.

    Deliberately excludes: object names, ground-truth IDs, object poses,
    oracle access, and raw MjModel/MjData handles.
    """

    def get_obs(self, camera: str = "onboard") -> Observation: ...
    def get_ee_pos(self) -> np.ndarray: ...
    def get_base_pose(self) -> np.ndarray: ...  # (x, y, yaw)
    def set_base_target(self, x: float, y: float, yaw: float) -> None: ...
    def set_arm_target(self, pos_world: np.ndarray) -> None: ...
    def step(self, n: int = 1) -> None: ...
    def try_attach_near_ee(self) -> Optional[str]: ...  # opaque handle or None
    def detach(self) -> bool: ...
    def is_attached(self) -> bool: ...
    def sim_time(self) -> float: ...
    def close(self) -> None: ...


class Perception(abc.ABC):
    """Student A's contract."""

    @abc.abstractmethod
    def describe(self, obs: Observation, query: Optional[str] = None) -> SceneDescription:
        """Full-scene description of one observation."""

    @abc.abstractmethod
    def ground(self, obs: Observation, target: str) -> Optional[GroundedObject]:
        """Locate one referent (normalized name or free-form phrase).
        Returns None when the target is not visible; a GroundedObject with
        status AMBIGUOUS when several candidates match equally."""

    def reset(self) -> None:
        """Clear per-episode tracking state (instance ID lifecycle)."""


class Planner(abc.ABC):
    """Student B's contract."""

    @abc.abstractmethod
    def plan(self, instruction: str, scene: SceneDescription) -> Plan:
        """Produce a Plan for the instruction given the current scene."""

    @abc.abstractmethod
    def replan(
        self,
        instruction: str,
        scene: SceneDescription,
        history: list[ExecutionResult],
        context: ExecutionContext,
        clarification: Optional[str] = None,
    ) -> Plan:
        """Produce a fresh plan mid-episode.  ``history`` holds prior
        execution results; ``clarification`` is the user's response if one
        was just obtained.  The returned plan REPLACES any remaining
        actions of the previous plan."""

    def reset(self) -> None:
        """Clear per-episode state."""


class Executor(abc.ABC):
    """Student C's contract."""

    @abc.abstractmethod
    def execute(
        self,
        action: Action,
        env: RobotEnvProtocol,
        perception: Perception,
    ) -> ExecutionResult:
        """Execute one action on the public environment.  ``perception`` is
        provided so SEARCH/VERIFY can use vision; the executor must not
        access ground truth."""

    def reset(self) -> None:
        """Clear per-episode state."""


class ClarificationProvider(abc.ABC):
    """Source of user clarification responses.  In evaluation this is a
    scripted queue; each scripted response is consumed exactly once."""

    @abc.abstractmethod
    def ask(self, question: str) -> Optional[str]:
        """Return the user's response, or None when no response is
        available (exhausted script)."""
