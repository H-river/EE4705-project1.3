# executor/student_c.py
"""Working C starting point. Extend the shared closed-loop simulation skills here."""
import numpy as np

from core.action_targets import resolve_action_position
from core.types import ExecutionResult, Skill, ErrorCode, SkillResult
from core import skills
from executor.closed_loop import ClosedLoopExecutor

class StudentCExecutor(ClosedLoopExecutor):
    """Eight bounded skills, with public sensor checks and one local grasp retry.

    Tested initially with weld attachment. This flag enables the interface;
    it does not certify contact-only grasping or the full course evaluation.
    """

    IMPLEMENTED = True
    LABEL = "C: Student C closed-loop baseline (simulation / weld)"

    def _execute(self, action, env, perception):
        # Dispatch skills you've overridden yourself; everything else falls
        # through to the shared closed_loop.py reference implementation.
        if action.skill is Skill.APPROACH:
            return self._approach(action, env, perception)
        
        elif action.skill is Skill.REACH:
            return self._reach(action, env, perception)
        return super()._execute(action, env, perception)

#Approach function: park the base so the target lands inside the right arm's workspace.
    def _approach(self, action, env, perception):
        """Park the base so the target lands inside the right arm's workspace."""
        # 1. Fresh observation + scene: never reuse a stale/cached position.
        obs = env.get_obs()
        scene = perception.describe(obs)

        # 2. Resolve the action's target into a world position. This honors
        #    an explicit params["pos"] override first, and otherwise looks
        #    up action.target in the CURRENT scene (raises TargetResolutionError
        #    for a missing/ambiguous/unlocalized reference).
        pos = resolve_action_position(action, scene)

        # 3. Run the reference motion primitive (bounded, timeout-guarded).
        primitive_result = skills.approach(env, pos)

        # 4. Let the base settle before returning, so a following action's
        #    fresh perception capture isn't taken mid-drift.
        env.stop_motion()
        env.step(100)

        # 5. Convert SkillResult -> ExecutionResult.
        post = env.get_obs()
        return ExecutionResult(
            action=action,
            success=primitive_result.success,
            error_code=primitive_result.error_code,
            post_frame_id=post.frame_id,
            info=primitive_result.info,
        )

#Reach function: move the TCP to the target's current position, then independently confirm the measured position actually got there.
    def _reach(self, action, env, perception):
        """REACH: move the TCP to the target's current position, then
        independently confirm the measured position actually got there."""
        scene = self._scene(env, perception)          # fresh obs -> validated, non-stale scene
        pos = resolve_action_position(action, scene)   # exact current target position (or params.pos override)
        primitive = skills.reach(env, pos)

        ee_error = float(np.linalg.norm(env.get_ee_pos() - pos))
        if primitive.success and ee_error >= skills.EE_POS_TOL:
            primitive = SkillResult(False, ErrorCode.UNREACHABLE,
                                    {**primitive.info, "detail": "TCP did not reach the requested point"})

        return ExecutionResult(action, primitive.success, primitive.error_code,
                               info={**primitive.info, "ee_error_m": ee_error})