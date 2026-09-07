# Owner: backbone demo (ALL), replaceable by Student C
"""Reference executor: real bounded simulated motion, never teleportation."""
import numpy as np

from core import skills
from core.action_targets import TargetResolutionError, resolve_action_position
from core.interfaces import Executor
from core.types import ErrorCode, ExecutionResult, GroundStatus, Skill, SkillResult
from core.verification import verify_placement


class DemoExecutor(Executor):
    LABEL = "C: real simulation control + weld attachment"

    def __init__(self, on_action=None, fail_first_grasp=False):
        self.on_action = on_action
        self.fail_first_grasp = fail_first_grasp
        self.reset()

    def reset(self):
        self._injected = False
        self._held_offset = np.zeros(3)
        self._view_pose = None

    def execute(self, action, env, perception):
        if self._view_pose is None:
            self._view_pose = tuple(env.get_base_pose())
        if self.on_action:
            self.on_action("start", action, None)
        try:
            result = self._execute(action, env, perception)
        except TargetResolutionError as exc:
            result = ExecutionResult(action, False, exc.error_code, info={"detail": str(exc)})
        except ValueError as exc:
            result = ExecutionResult(action, False, ErrorCode.UNREACHABLE, info={"detail": str(exc)})
        if not result.success:
            env.stop_motion()
        result.post_frame_id = env.get_obs().frame_id
        if self.on_action:
            self.on_action("end", action, result)
        return result

    def _execute(self, action, env, perception):
        skill = action.skill
        if skill is Skill.STOP:
            env.stop_motion()
            env.step(100)
            return ExecutionResult(action, True)
        if skill is Skill.VERIFY:
            params = action.params
            if params.get("condition") == "object_in_region":
                check = verify_placement(env, perception, params["object"], params["region"])
                return ExecutionResult(action, check.passed, ErrorCode.NONE if check.passed else ErrorCode.VERIFY_FAILED,
                                       post_frame_id=check.frame_id, info={"detail": check.detail})
            if params.get("condition") == "holding":
                ok = env.is_attached()
            elif params.get("condition") == "object_visible":
                found = perception.ground(env.get_obs(), action.target or "")
                ok = found is not None and found.status is GroundStatus.LOCALIZED
            else:
                return ExecutionResult(action, False, ErrorCode.INVALID_ACTION)
            return ExecutionResult(action, ok, ErrorCode.NONE if ok else ErrorCode.VERIFY_FAILED)
        if skill is Skill.SEARCH:
            primitive = skills.search(env, perception, action.target or "", max_views=9, timeout_s=30.)
        else:
            scene = perception.describe(env.get_obs())
            # Use fresh localization for grasp; explicit transport waypoints remain
            # usable when the static region is briefly outside the head-camera FOV.
            pos = resolve_action_position(action, scene)
            target = scene.find(action.target or "")
            if skill is Skill.GRASP and target and target.status is GroundStatus.LOCALIZED:
                pos = np.asarray(target.pos_world)
            if skill is Skill.APPROACH:
                primitive = skills.approach(env, pos)
                env.stop_motion()
                env.step(100)
            elif skill is Skill.REACH:
                primitive = skills.reach(env, pos)
            elif skill is Skill.GRASP:
                if self.fail_first_grasp and not self._injected:
                    self._injected = True
                    return ExecutionResult(action, False, ErrorCode.GRASP_MISSED,
                                           info={"injected": True, "detail": "Explicit demo fault: first grasp attempt rejected"})
                primitive = skills.grasp(env, pos)
                if primitive.success:
                    self._held_offset = pos-env.get_ee_pos()
                    env.set_gripper("right", .35)
                    primitive = skills.move_to(env, env.get_ee_pos()+[0, 0, .12])
                    if not env.is_attached():
                        primitive = SkillResult(False, ErrorCode.GRASP_MISSED)
            elif skill is Skill.MOVE_TO:
                if not env.is_attached():
                    primitive = SkillResult(False, ErrorCode.NOT_HOLDING)
                else:
                    primitive = skills.move_to(env, pos)
                    if primitive.success and not env.is_attached():
                        primitive = SkillResult(False, ErrorCode.NOT_HOLDING)
            elif skill is Skill.PLACE:
                if not env.is_attached():
                    primitive = SkillResult(False, ErrorCode.NOT_HOLDING)
                else:
                    # Control the object center to 6 cm above the support; compensate
                    # for the measured-at-grasp visual center/TCP offset.
                    release = pos + [0, 0, .06] - self._held_offset
                    primitive = skills.move_to(env, release)
                    if primitive.success:
                        env.stop_motion()
                        primitive = skills.place(env)
                        env.set_gripper("right", 1.)
                        # Lift away so final visual verification can see the object.
                        retreat = skills.reach(env, env.get_ee_pos()+[0, 0, .14])
                        env.stop_motion()
                        env.step(150)
                        if not retreat.success:
                            primitive = retreat
                        else:
                            # Return to the initial camera viewpoint using only
                            # proprioception. At the placement pose the object
                            # can be clipped by the head camera's narrow FOV.
                            primitive = self._restore_view(env)
            else:
                primitive = SkillResult(False, ErrorCode.INVALID_ACTION)
        return ExecutionResult(action, primitive.success, primitive.error_code, info=primitive.info)

    def _restore_view(self, env):
        env.set_base_target(*self._view_pose)
        deadline = env.sim_time() + 8.
        while env.sim_time() < deadline:
            current = np.asarray(env.get_base_pose())
            delta = current - self._view_pose
            yaw_error = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            if np.linalg.norm(delta[:2]) < .02 and abs(yaw_error) < .035:
                env.stop_motion()
                env.step(100)
                return SkillResult(True, info={"primitive": "place", "view_restored": True})
            env.step(20)
        env.stop_motion()
        return SkillResult(False, ErrorCode.TIMEOUT, info={"detail": "Could not restore the observation viewpoint"})
