# Owner: Student C
"""Student C's real executor module (closed-loop motion skills).  NOT part
of the backbone deliverable; this stub only pins the contract."""

from __future__ import annotations

from core.interfaces import Executor, Perception, RobotEnvProtocol
from core.types import Action, ExecutionResult


class StudentCExecutor(Executor):
    """Owner: Student C.  Real closed-loop executor. Unimplemented."""

    IMPLEMENTED = False

    def execute(
        self,
        action: Action,
        env: RobotEnvProtocol,
        perception: Perception,
    ) -> ExecutionResult:
        raise NotImplementedError("Student C: Executor.execute is not implemented yet")
