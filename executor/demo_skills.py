"""Demo C wrapper; fault injection is separate from the reusable motion skills."""
from core.types import ErrorCode, ExecutionResult, Skill
from executor.closed_loop import ClosedLoopExecutor


class DemoExecutor(ClosedLoopExecutor):
    LABEL = "C: closed-loop simulation control + weld attachment (demo)"

    def __init__(self, on_action=None, fail_first_grasp=False):
        self.fail_first_grasp = fail_first_grasp
        super().__init__(on_action=on_action)

    def reset(self):
        super().reset()
        self._injected = False

    def _execute(self, action, env, perception):
        if action.skill is Skill.GRASP and self.fail_first_grasp and not self._injected:
            self._injected = True
            return ExecutionResult(action, False, ErrorCode.GRASP_MISSED,
                                   info={"injected": True, "detail": "Explicit demo fault: first grasp attempt rejected"})
        return super()._execute(action, env, perception)
