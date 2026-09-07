"""Reusable C skills using current perception and public robot feedback only."""
import numpy as np

from core import skills
from core.action_targets import TargetResolutionError, resolve_action_position
from core.interfaces import Executor
from core.types import ErrorCode, ExecutionResult, GroundStatus, Skill, SkillResult
from core.verification import verify_placement


class ClosedLoopExecutor(Executor):
    """Bounded simulation baseline. Attachment identity is a tracked belief, not truth."""

    def __init__(self, on_action=None, grasp_attempts=2):
        if grasp_attempts not in (1, 2):
            raise ValueError("grasp_attempts must be 1 or 2")
        self.on_action, self.grasp_attempts = on_action, grasp_attempts
        self.reset()

    def reset(self):
        self._held_id = None
        self._held_offset = np.zeros(3)
        self._view_pose = None

    def execute(self, action, env, perception):
        start = env.sim_time()
        if self._view_pose is None:
            self._view_pose = tuple(env.get_base_pose())
        if not env.is_attached():
            self._held_id = None
        if self.on_action:
            self.on_action("start", action, None)
        try:
            result = self._execute(action, env, perception)
        except TargetResolutionError as exc:
            result = ExecutionResult(action, False, exc.error_code, info={"detail": str(exc)})
        except ValueError as exc:
            result = ExecutionResult(action, False, ErrorCode.INVALID_ACTION, info={"detail": str(exc)})
        except Exception as exc:
            result = ExecutionResult(action, False, ErrorCode.INTERNAL_ERROR,
                                     info={"detail": f"{type(exc).__name__}: {exc}"})
        env.stop_motion()
        attached = env.is_attached()
        if not attached:
            self._held_id = None
        result.info.update(elapsed_sim_s=env.sim_time()-start, attached_after=attached,
                           held_instance_id=self._held_id, ee_pos=env.get_ee_pos().tolist())
        result.post_frame_id = env.get_obs().frame_id
        if self.on_action:
            self.on_action("end", action, result)
        return result

    @staticmethod
    def _scene(env, perception):
        obs = env.get_obs()
        try:
            scene = perception.describe(obs)
        except Exception as exc:
            raise TargetResolutionError(f"Perception failed: {type(exc).__name__}: {exc}",
                                        ErrorCode.PERCEPTION_ERROR) from exc
        if scene.frame_id != obs.frame_id or abs(scene.sim_time-obs.sim_time) > 1e-9:
            raise TargetResolutionError("Perception returned a stale scene", ErrorCode.PERCEPTION_ERROR)
        return scene

    @staticmethod
    def _located(scene, target):
        ref = scene.find(target or "")
        if (ref is None or ref.status is not GroundStatus.LOCALIZED or ref.pos_world is None
                or (ref.frame_id >= 0 and ref.frame_id != scene.frame_id)):
            raise TargetResolutionError(f"Target {target!r} needs fresh, unambiguous 3D evidence")
        return ref

    @staticmethod
    def _open(env):
        env.stop_motion()
        env.set_gripper("right", 1.)
        deadline = env.sim_time() + 2.
        while env.sim_time() < deadline:
            if env.get_robot_state().gripper_opening["right"] >= .9:
                return True
            env.step(10)
        return env.get_robot_state().gripper_opening["right"] >= .9

    @staticmethod
    def _arrived(primitive, env, pos):
        error = float(np.linalg.norm(env.get_ee_pos()-pos))
        primitive.info["ee_error_m"] = error
        if primitive.success and (not np.isfinite(error) or error >= skills.EE_POS_TOL):
            return SkillResult(False, ErrorCode.UNREACHABLE,
                               {**primitive.info, "detail": "TCP did not reach the requested point"})
        return primitive

    def _transport(self, env, pos):
        primitive = self._arrived(skills.move_to(env, pos), env, pos)
        if not primitive.success and primitive.error_code in (ErrorCode.TIMEOUT, ErrorCode.UNREACHABLE):
            # IK can accept a point while the loaded arm stalls in execution.
            # Change the base parking pose once instead of repeating the same reach.
            first = {"error_code": primitive.error_code.value, **primitive.info}
            env.stop_motion()
            if not env.is_attached():
                return SkillResult(False, ErrorCode.NOT_HOLDING, {"motion_recovery": first})
            parking = skills.approach(env, pos)
            env.stop_motion()
            env.step(100)
            primitive = self._arrived(skills.reach(env, pos), env, pos) if parking.success else parking
            primitive.info["motion_recovery"] = first
            primitive.info["base_repositioned"] = parking.success
        return primitive

    def _execute(self, action, env, perception):
        skill = action.skill
        if skill is Skill.STOP:
            return self._stop(action, env)
        if skill is Skill.VERIFY:
            condition = action.params.get("condition")
            if condition == "object_in_region":
                check = verify_placement(env, perception, action.params["object"], action.params["region"])
                return ExecutionResult(action, check.passed, ErrorCode.NONE if check.passed else ErrorCode.VERIFY_FAILED,
                                       info={"detail": check.detail, "verification_frame_id": check.frame_id})
            if condition == "holding":
                ok = env.is_attached() and bool(action.target) and action.target == self._held_id
            elif condition == "object_visible":
                scene = self._scene(env, perception)
                ref = scene.find(action.target or "")
                ok = (ref is not None and ref.status in (GroundStatus.LOCALIZED, GroundStatus.UNLOCALIZED)
                      and (ref.frame_id < 0 or ref.frame_id == scene.frame_id))
            else:
                return ExecutionResult(action, False, ErrorCode.INVALID_ACTION)
            return ExecutionResult(action, ok, ErrorCode.NONE if ok else ErrorCode.VERIFY_FAILED,
                                   info={"condition": condition, "target": action.target})
        if skill is Skill.SEARCH:
            primitive = skills.search(env, perception, action.target or "", max_views=13, timeout_s=30.)
            if primitive.success:
                self._view_pose = tuple(env.get_base_pose())
        elif skill is Skill.GRASP:
            return self._grasp(action, env, perception)
        elif skill in (Skill.APPROACH, Skill.REACH, Skill.MOVE_TO, Skill.PLACE):
            scene = self._scene(env, perception)
            pos = resolve_action_position(action, scene)
            if skill is Skill.APPROACH:
                primitive = skills.approach(env, pos)
                env.stop_motion()
                env.step(100)
            elif skill is Skill.REACH:
                primitive = self._arrived(skills.reach(env, pos), env, pos)
            elif skill is Skill.MOVE_TO:
                if not env.is_attached():
                    primitive = SkillResult(False, ErrorCode.NOT_HOLDING)
                else:
                    primitive = self._transport(env, pos)
                    if not env.is_attached():
                        primitive = SkillResult(False, ErrorCode.NOT_HOLDING,
                                                {**primitive.info, "detail": "Attachment lost during transport"})
            else:
                return self._place(action, env, perception, pos)
        else:
            primitive = SkillResult(False, ErrorCode.INVALID_ACTION)
        return ExecutionResult(action, primitive.success, primitive.error_code, info=primitive.info,
                               recovery_attempted="motion_recovery" in primitive.info)

    def _grasp(self, action, env, perception):
        if env.is_attached():
            return ExecutionResult(action, False, ErrorCode.ALREADY_HOLDING)
        attempts = []
        for attempt in range(self.grasp_attempts):
            scene = self._scene(env, perception)
            # A cached params.pos never authorizes grasping an absent instance.
            target = self._located(scene, action.target)
            if target.kind != "object":
                return ExecutionResult(action, False, ErrorCode.INVALID_ACTION)
            pos = np.asarray(target.pos_world, dtype=float)
            if not self._open(env):
                primitive = SkillResult(False, ErrorCode.TIMEOUT, {"detail": "Gripper did not open before grasp"})
                break
            primitive = skills.grasp(env, pos)
            attempts.append({"attempt": attempt+1, "frame_id": scene.frame_id,
                             "target_pos": pos.tolist(), "error_code": primitive.error_code.value,
                             "attach_reason": env.get_robot_state().last_attach_reason})
            if env.is_attached():
                self._held_id = action.target
                self._held_offset = pos-env.get_ee_pos()
                if primitive.success:
                    env.set_gripper("right", .35)
                    lift = env.get_ee_pos()+[0, 0, .12]
                    primitive = self._arrived(skills.move_to(env, lift), env, lift)
                    env.step(50)
                    if not env.is_attached():
                        primitive = SkillResult(False, ErrorCode.GRASP_MISSED,
                                                {"detail": "Attachment lost during lift check"})
                break
            if primitive.success:
                primitive = SkillResult(False, ErrorCode.GRASP_MISSED, {"detail": "No attachment after grasp"})
            if primitive.error_code is not ErrorCode.GRASP_MISSED or attempt+1 == self.grasp_attempts:
                break
            env.stop_motion()
        return ExecutionResult(action, primitive.success, primitive.error_code,
                               recovery_attempted=len(attempts)>1,
                               info={**primitive.info, "grasp_attempts": attempts})

    def _place(self, action, env, perception, pos):
        if not env.is_attached():
            return ExecutionResult(action, False, ErrorCode.NOT_HOLDING)
        obj = action.params.get("object") or self._held_id
        if not obj or obj != self._held_id:
            return ExecutionResult(action, False, ErrorCode.INVALID_ACTION,
                                   info={"detail": "PLACE object differs from C's tracked grasp target"})
        release = pos + [0, 0, .06] - self._held_offset
        primitive = self._transport(env, release)
        transport_info = primitive.info.copy()
        if primitive.success:
            if not self._open(env):
                return ExecutionResult(action, False, ErrorCode.PLACE_FAILED,
                                       info={"detail": "Gripper did not open before release"})
            primitive = skills.place(env)
            if not primitive.success:
                # Never overwrite a failed detach with a successful retreat.
                return ExecutionResult(action, False, primitive.error_code, info=primitive.info)
            retreat = env.get_ee_pos()+[0, 0, .14]
            primitive = self._arrived(skills.reach(env, retreat), env, retreat)
            env.stop_motion()
            env.step(150)
            if primitive.success:
                primitive = self._restore_view(env)
            if primitive.success:
                check = verify_placement(env, perception, obj, action.target)
                view_attempts = 0
                # A correct placement can be outside the old narrow camera view.
                # Re-observe from two nearby headings; never relax the position check.
                if not check.passed and check.detail == "exact object or region instance is not visible":
                    base = np.asarray(self._view_pose)
                    direction = np.sign(np.sin(np.arctan2(pos[1]-base[1], pos[0]-base[0])-base[2])) or 1.
                    for offset in (direction*.2, -direction*.2):
                        view_attempts += 1
                        view = self._restore_view(env, yaw_offset=offset)
                        if not view.success:
                            break
                        check = verify_placement(env, perception, obj, action.target)
                        if check.passed or check.detail != "exact object or region instance is not visible":
                            break
                primitive = SkillResult(check.passed, ErrorCode.NONE if check.passed else ErrorCode.PLACE_FAILED,
                                        {"primitive": "place", "released": True, "detail": check.detail,
                                         "verification_frame_id": check.frame_id, "view_recovery_attempts": view_attempts})
        primitive.info = {**transport_info, **primitive.info}
        return ExecutionResult(action, primitive.success, primitive.error_code, info=primitive.info,
                               recovery_attempted=bool(primitive.info.get("view_recovery_attempts"))
                               or "motion_recovery" in primitive.info)

    def _restore_view(self, env, yaw_offset=0.):
        target = np.asarray(self._view_pose).copy()
        target[2] = np.arctan2(np.sin(target[2]+yaw_offset), np.cos(target[2]+yaw_offset))
        env.set_base_target(*target)
        deadline = env.sim_time() + 8.
        while env.sim_time() < deadline:
            delta = np.asarray(env.get_base_pose()) - target
            yaw_error = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            if np.linalg.norm(delta[:2]) < .02 and abs(yaw_error) < .035:
                env.stop_motion()
                env.step(100)
                return SkillResult(True, info={"primitive": "place", "view_restored": True})
            env.step(20)
        return SkillResult(False, ErrorCode.TIMEOUT, info={"detail": "Could not restore the observation viewpoint"})

    @staticmethod
    def _stop(action, env):
        env.stop_motion()
        attached = env.is_attached()
        deadline, stable_samples = env.sim_time()+2., 0
        before = env.get_robot_state()
        while env.sim_time() < deadline:
            env.step(max(1, round(.1/env.timestep())))
            after = env.get_robot_state()
            delta = np.asarray(after.base_pose)-before.base_pose
            ee_drift = float(np.linalg.norm(np.asarray(after.ee_pos)-before.ee_pos))
            yaw_drift = abs(float(np.arctan2(np.sin(delta[2]), np.cos(delta[2]))))
            settled = np.linalg.norm(delta[:2]) < .003 and yaw_drift < .01 and ee_drift < .005
            stable_samples = stable_samples+1 if settled else 0
            info = {"ee_drift_m": ee_drift, "base_drift_m": float(np.linalg.norm(delta[:2])),
                    "yaw_drift_rad": yaw_drift, "stable_samples": stable_samples}
            if attached and not env.is_attached():
                return ExecutionResult(action, False, ErrorCode.NOT_HOLDING, info=info)
            if stable_samples >= 2:
                return ExecutionResult(action, True, info=info)
            before = after
        return ExecutionResult(action, False, ErrorCode.TIMEOUT, info=info)
