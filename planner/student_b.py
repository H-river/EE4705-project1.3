# Owner: Student B
"""Student B's real planner module (LLM planning).  NOT part of the
backbone deliverable; this stub only pins the contract."""

from __future__ import annotations

from typing import Optional

from core.interfaces import Planner
from core.types import ExecutionContext, ExecutionResult, Plan, SceneDescription


class StudentBPlanner(Planner):
    """Owner: Student B.  Real LLM-based planner. Unimplemented."""

    IMPLEMENTED = False

    def plan(self, instruction: str, scene: SceneDescription) -> Plan:
        raise NotImplementedError("Student B: Planner.plan is not implemented yet")

    def replan(
        self,
        instruction: str,
        scene: SceneDescription,
        history: list[ExecutionResult],
        context: ExecutionContext,
        clarification: Optional[str] = None,
    ) -> Plan:
        raise NotImplementedError("Student B: Planner.replan is not implemented yet")
