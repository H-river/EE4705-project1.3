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

        elif action.skill is Skill.MOVE_TO:
            return self._move_to(action, env, perception)

        elif action.skill is Skill.PLACE:
            return self._place(action, env, perception)
        
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
                self._held_offset = pos - env.get_ee_pos()   # <-- new: needed by PLACE
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

#MOVE_TO function: move the TCP to the target's current position, retrying once if the base's parking pose caused a stall.
    def _move_to(self, action, env, perception):
        if not env.is_attached():
            return ExecutionResult(action, False, ErrorCode.NOT_HOLDING)

        # self._scene() (inherited) does describe() + the staleness check and
        # raises TargetResolutionError on failure; execute()'s outer wrapper
        # catches that, so no try/except is needed here.
        scene = self._scene(env, perception)
        pos = resolve_action_position(action, scene)  # also raises on missing/unlocalized

        primitive_result = skills.move_to(env, pos)
        recovery_attempted = False
        motion_recovery_info = None
        base_repositioned = None

        # A stalled/unreachable reach can be a consequence of the base's
        # CURRENT parking pose, not a genuinely unreachable target.
        # Re-park once and retry a single reach before giving up.
        if not primitive_result.success and primitive_result.error_code in (
            ErrorCode.TIMEOUT, ErrorCode.UNREACHABLE,
        ):
            motion_recovery_info = {"error_code": primitive_result.error_code.value, **primitive_result.info}
            recovery_attempted = True
            env.stop_motion()
            if not env.is_attached():
                # Recovery is pointless if the object was lost during the failed attempt.
                return ExecutionResult(
                    action, False, ErrorCode.NOT_HOLDING,
                    recovery_attempted=recovery_attempted,
                    info={"motion_recovery": motion_recovery_info,
                          "detail": "Attachment lost before recovery could be attempted"},
                )
            parking = skills.approach(env, pos)
            env.stop_motion()
            env.step(100)  # let the new base pose settle before reaching again
            base_repositioned = parking.success
            primitive_result = skills.reach(env, pos) if parking.success else parking

        # Independent post-motion checks — never just trust the primitive's own flag.
        ee_error = float(np.linalg.norm(env.get_ee_pos() - pos))
        still_holding = env.is_attached()

        if primitive_result.success and not still_holding:
            success, error_code = False, ErrorCode.NOT_HOLDING
            detail = "Attachment lost during transport"
        elif primitive_result.success and ee_error >= skills.EE_POS_TOL:
            success, error_code = False, ErrorCode.UNREACHABLE
            detail = "TCP did not reach the requested point"
        else:
            success = primitive_result.success
            error_code = primitive_result.error_code
            detail = primitive_result.info.get("detail", "")

        info = {**primitive_result.info, "ee_error_m": ee_error, "detail": detail}
        if motion_recovery_info is not None:
            info["motion_recovery"] = motion_recovery_info
            info["base_repositioned"] = base_repositioned

        return ExecutionResult(action, success, error_code,
                            recovery_attempted=recovery_attempted, info=info)

#PLACE function: move above the destination surface (never into it), release, let the object settle, then confirm placement from a fresh observation.
    _RELEASE_HEIGHT_M = 0.06
    _RETREAT_HEIGHT_M = 0.14
    _NOT_VISIBLE_DETAIL = "exact object or region instance is not visible"

    def _place(self, action, env, perception):
        """PLACE: move above the destination surface, release, let the
        object settle, then confirm placement from a fresh observation.
        If the object simply isn't in view after retreating, try up to two
        nearby headings before accepting failure."""

        if not env.is_attached():
            return ExecutionResult(action, False, ErrorCode.NOT_HOLDING)

        scene = self._scene(env, perception)
        region_pos = resolve_action_position(action, scene)

        obj_id = action.params.get("object") or self._held_id
        if not obj_id or obj_id != self._held_id:
            return ExecutionResult(action, False, ErrorCode.INVALID_ACTION,
                                   info={"detail": "PLACE object differs from C's tracked grasp target"})

        release_pos = region_pos + np.array([0.0, 0.0, self._RELEASE_HEIGHT_M]) - self._held_offset
        primitive = skills.move_to(env, release_pos)

        if primitive.success and not env.is_attached():
            return ExecutionResult(action, False, ErrorCode.NOT_HOLDING,
                                   info={**primitive.info, "detail": "Attachment lost while moving to the release pose"})

        ee_error = float(np.linalg.norm(env.get_ee_pos() - release_pos))
        if primitive.success and ee_error >= skills.EE_POS_TOL:
            primitive = SkillResult(False, ErrorCode.UNREACHABLE,
                                    {**primitive.info, "detail": "TCP did not reach the release pose"})
        if not primitive.success:
            return ExecutionResult(action, False, primitive.error_code,
                                   info={**primitive.info, "ee_error_m": ee_error})

        if not self._open(env):
            return ExecutionResult(action, False, ErrorCode.TIMEOUT,
                                   info={"detail": "Gripper did not open before release"})

        release_result = skills.place(env)
        if not release_result.success:
            # Never overwrite a failed detach with a successful retreat.
            return ExecutionResult(action, False, release_result.error_code, info=release_result.info)

        retreat_pos = env.get_ee_pos() + np.array([0.0, 0.0, self._RETREAT_HEIGHT_M])
        skills.reach(env, retreat_pos)
        env.stop_motion()
        env.step(100)

        # First visual check from the retreat viewpoint.
        check = closed_loop.verify_placement(env, perception, obj_id, action.target)

        # Only retry the VIEW for "not visible" -- that's a camera-framing
        # problem. "Outside region" / drift are real placement errors that
        # a different heading cannot fix, so we do not retry those.
        view_attempts = 0
        if not check.passed and check.detail == self._NOT_VISIBLE_DETAIL:
            base = np.asarray(self._view_pose)
            bearing = np.arctan2(region_pos[1] - base[1], region_pos[0] - base[0]) - base[2]
            direction = np.sign(np.sin(bearing)) or 1.0
            for offset in (direction * 0.2, -direction * 0.2):
                view_attempts += 1
                view = self._restore_view(env, yaw_offset=offset)
                if not view.success:
                    break  # couldn't even get to the new viewpoint; stop trying
                check = closed_loop.verify_placement(env, perception, obj_id, action.target)
                if check.passed or check.detail != self._NOT_VISIBLE_DETAIL:
                    break  # resolved, or failing for a different (non-view) reason now

        return ExecutionResult(
            action=action,
            success=check.passed,
            error_code=ErrorCode.NONE if check.passed else ErrorCode.PLACE_FAILED,
            recovery_attempted=view_attempts > 0,
            info={"detail": check.detail, "verification_frame_id": check.frame_id,
                  "view_recovery_attempts": view_attempts},
        )

#