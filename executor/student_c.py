# executor/student_c.py
"""Working C starting point. Extend the shared closed-loop simulation skills here."""
import numpy as np

from core import skills
from core.action_targets import TargetResolutionError, resolve_action_position
from core.types import ErrorCode, ExecutionResult, GroundStatus, Skill, SkillResult
from executor.closed_loop import ClosedLoopExecutor

class StudentCExecutor(ClosedLoopExecutor):
    """Eight bounded skills, with public sensor checks and one local grasp retry.

    Tested initially with weld attachment. This flag enables the interface;
    it does not certify contact-only grasping or the full course evaluation.
    """

    IMPLEMENTED = True
    LABEL = "C: Student C closed-loop baseline (simulation / weld)"

    _LIFT_CHECK_M = 0.12  # how high to lift to prove a REAL GRASP, not just "gripper closed near it"

    def _execute(self, action, env, perception):
        # Dispatch skills you've overridden yourself; everything else falls
        # through to the shared closed_loop.py reference implementation.
        if action.skill is Skill.APPROACH:
            return self._approach(action, env, perception)
        
        elif action.skill is Skill.REACH:
            return self._reach(action, env, perception)

        elif action.skill is Skill.GRASP:
            return self._grasp(action, env, perception)
        
        return super()._execute(action, env, perception)

#APPROACH function: park the base so the target lands inside the right arm's workspace.
    def _approach(self, action, env, perception):
        """Park the base so the target lands inside the right arm's workspace."""
        # 1. Fresh observation + scene: never reuse a stale/cached position.
        obs = env.get_obs()
        scene = perception.describe(obs)

        # 2. Resolve the action's target into a world position. This honors
        #    an explicit params["pos"] override first, and otherwise looks
        #    up action.target in the CURRENT scene (raises TargetResolutionError
        #    for a missing/ambiguous/unlocalized reference).
        try:
            pos = resolve_action_position(action, scene)
        except TargetResolutionError:
            return ExecutionResult(
                action=action,
                success=False,
                error_code=ErrorCode.TARGET_UNRESOLVABLE,
                info={"detail": "Failed to resolve action target"}
            )

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

#REACH function: move the TCP to the target's current position, then independently confirm the measured position actually got there.
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

#GRASP function: attempt to grasp the target, retrying if the first attempt fails.
    def _grasp(self, action, env, perception):
        if env.is_attached():
            return ExecutionResult(action, False, ErrorCode.ALREADY_HOLDING)

        attempts = []
        primitive = None
        for attempt in range(self.grasp_attempts):  # inherited default: 2
            # Rule: never grasp against an old/cached position — re-observe every attempt.
            obs = env.get_obs()
            scene = perception.describe(obs)
            if scene.frame_id != obs.frame_id or abs(scene.sim_time - obs.sim_time) > 1e-9:
                raise TargetResolutionError("Perception returned a stale scene", ErrorCode.PERCEPTION_ERROR)

            target = scene.find(action.target or "")
            if (target is None or target.status is not GroundStatus.LOCALIZED
                    or target.pos_world is None):
                raise TargetResolutionError(f"Target {action.target!r} needs fresh, unambiguous 3D evidence")
            if target.kind != "object":
                return ExecutionResult(action, False, ErrorCode.INVALID_ACTION,
                                       info={"detail": "GRASP target is not an object"})

            pos = resolve_action_position(action, scene)  # world meters, object center

            # Open first: closing without opening first can "pinch nothing" and still look closed.
            if not self._open(env):
                primitive = SkillResult(False, ErrorCode.TIMEOUT,
                                        {"detail": "Gripper did not open before grasp"})
                break

            primitive = skills.grasp(env, pos)  # reaches + attempts weld attachment

            attempts.append({
                "attempt": attempt + 1,
                "frame_id": scene.frame_id,
                "target_pos": pos.tolist(),
                "error_code": primitive.error_code.value,
                "attach_reason": env.get_robot_state().last_attach_reason,
            })

            if env.is_attached():
                # A closed gripper near the object is not proof of a grasp:
                # lift a bit and re-check that the attachment survives it.
                self._held_id = action.target
                lift_target = env.get_ee_pos() + [0.0, 0.0, self._LIFT_CHECK_M]
                primitive = skills.move_to(env, lift_target)
                env.step(50)
                if not env.is_attached():
                    primitive = SkillResult(False, ErrorCode.GRASP_MISSED,
                                            {"detail": "Attachment lost during lift check"})
                break

            if primitive.success:
                # skills.grasp() claimed success but nothing is attached — don't trust it.
                primitive = SkillResult(False, ErrorCode.GRASP_MISSED, {"detail": "No attachment after grasp"})

            if primitive.error_code is not ErrorCode.GRASP_MISSED or attempt + 1 == self.grasp_attempts:
                break  # not a retryable failure, or out of attempts
            env.stop_motion()  # cancel the failed reach before retrying

        return ExecutionResult(
            action=action,
            success=primitive.success,
            error_code=primitive.error_code,
            recovery_attempted=len(attempts) > 1,
            info={**primitive.info, "grasp_attempts": attempts},
        )    

        