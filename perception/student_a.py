# Owner: Student A
"""Student A's real perception module (VLM grounding).  NOT part of the
backbone deliverable; this stub only pins the contract."""

from __future__ import annotations

from typing import Optional

from core.interfaces import Perception
from core.types import GroundedObject, Observation, SceneDescription


class StudentAPerception(Perception):
    """Owner: Student A.  Real VLM-based perception. Unimplemented."""

    IMPLEMENTED = False

    def describe(self, obs: Observation, query: Optional[str] = None) -> SceneDescription:
        raise NotImplementedError("Student A: Perception.describe is not implemented yet")

    def ground(self, obs: Observation, target: str) -> Optional[GroundedObject]:
        raise NotImplementedError("Student A: Perception.ground is not implemented yet")
