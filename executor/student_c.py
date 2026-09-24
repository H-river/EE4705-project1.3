# executor/student_c.py
"""Working C starting point. Extend the shared closed-loop simulation skills here."""
import math

import numpy as np

from core import skills
from core.action_targets import TargetResolutionError, resolve_action_position
from core.types import (CONTENT_FILTERED_NOTE, ErrorCode, ExecutionResult, GroundStatus,
                        Skill, SkillResult)
from executor import closed_loop
from executor.closed_loop import ClosedLoopExecutor
from core.verification import verify_placement
from core.scene_geometry import table_aabb, table_center_xy
from core.types import IMAGE_HEIGHT, IMAGE_WIDTH


class _BodyTableContact(RuntimeError):
    def __init__(self, contacts):
        super().__init__("robot body contacted the table")
        self.contacts = contacts


class _ContactGuard:
    """Proxy that interrupts shared motion primitives after each step."""

    def __init__(self, env, contact_reader):
        self._env = env
        self._contact_reader = contact_reader

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        if name in {"_env", "_contact_reader"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)

    def step(self, n=1):
        self._env.step(n)
        contacts = self._contact_reader()
        if contacts:
            self._env.stop_motion()
            raise _BodyTableContact(contacts)

def _planar(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:2, 3] = [x, y]
    return t


def table_top_points(n_x=9, n_y=21):
    """Grid over the table's top face (world frame)."""
    lo, hi = table_aabb()
    xs, ys = np.linspace(lo[0], hi[0], n_x), np.linspace(lo[1], hi[1], n_y)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, hi[2])))


def table_in_view(t_world_camera, intrinsics, max_range_m=4.0):
    """True when some part of the table top projects inside the image."""
    points = table_top_points()
    cam = (points - t_world_camera[:3, 3]) @ t_world_camera[:3, :3]
    z = cam[:, 2]
    ahead = (z > 0.05) & (z < max_range_m)
    if not np.any(ahead):
        return False
    u = intrinsics[0, 0] * cam[ahead, 0] / z[ahead] + intrinsics[0, 2]
    v = intrinsics[1, 1] * cam[ahead, 1] / z[ahead] + intrinsics[1, 2]
    return bool(np.any((u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)))


def search_views(base_pose, t_world_camera, intrinsics, n_views=13, half_range_rad=math.radians(60)):
    """Tabletop-only SEARCH viewpoints.

    The head camera is rigid on the torso, so ``t_world_camera`` observed at
    ``base_pose`` fixes the camera's height and pitch relative to the base;
    only the base yaw can change. Candidate yaws are ``n_views`` evenly spaced
    over ±``half_range_rad`` around the bearing from the base to the table
    centre, ordered from the end nearer the current yaw. A candidate whose
    frustum contains no part of the table top is skipped.
    Returns (views, info) with views = [(yaw, t_world_camera_at_yaw)].
    """
    x, y, yaw0 = (float(v) for v in base_pose)
    base_cam = np.linalg.inv(_planar(x, y, yaw0)) @ np.asarray(t_world_camera, dtype=float)
    cx, cy = table_center_xy()
    center = math.atan2(cy - y, cx - x)
    offsets = np.linspace(-half_range_rad, half_range_rad, n_views)
    wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
    if abs(wrap(center + offsets[-1] - yaw0)) < abs(wrap(center + offsets[0] - yaw0)):
        offsets = offsets[::-1]
    views, skipped = [], []
    for off in offsets:
        yaw = wrap(center + float(off))
        cam = _planar(x, y, yaw) @ base_cam
        (views if table_in_view(cam, intrinsics) else skipped).append((yaw, cam))
    # Pitch is fixed by the mount; report whether it can see the table at all.
    forward = base_cam[:3, 2]
    pitch = math.atan2(-forward[2], math.hypot(forward[0], forward[1]))  # + = looking down
    lo, hi = table_aabb()
    height = float(base_cam[2, 3]) - hi[2]
    far = max(math.hypot(px - x, py - y) for px in (lo[0], hi[0]) for py in (lo[1], hi[1]))
    near = math.hypot(min(max(x, lo[0]), hi[0]) - x, min(max(y, lo[1]), hi[1]) - y)
    half_vfov = math.atan2(IMAGE_HEIGHT / 2, float(intrinsics[1, 1]))
    # Pitch window in which some of the table top stays in the vertical field of view.
    pitch_range = (math.atan2(height, far) - half_vfov, math.atan2(height, max(near, 1e-3)) + half_vfov)
    info = {"table_bearing_rad": center, "candidate_yaws": [wrap(center + float(o)) for o in offsets],
            "skipped_yaws": [v[0] for v in skipped], "camera_pitch_rad": pitch,
            "table_pitch_range_rad": list(pitch_range),
            "pitch_sees_table": pitch_range[0] <= pitch <= pitch_range[1]}
    return views, info


class StudentCExecutor(ClosedLoopExecutor):
    """Eight bounded skills, with public sensor checks and one local grasp retry.

    Tested initially with weld attachment. This flag enables the interface;
    it does not certify contact-only grasping or the full course evaluation.
    """

    IMPLEMENTED = True
    LABEL = "C: Student C closed-loop baseline (simulation / weld)"

    _LIFT_CHECK_M = 0.12  # how high to lift to prove a REAL GRASP, not just "gripper closed near it"
    _CONTACT_BACKOFF_M = 0.100  # reverse clearance, based on the object size scale in objects.yaml

    # Keep the right hand tucked beside the torso while the base moves.  This
    # is deliberately high above the table and close to the shoulder rather
    # than a tabletop waypoint, so approach cannot sweep the hand through an
    # object.
    _ARM_TUCK_OFFSET = np.array([0.16, -0.10, 1.12])

    _RELEASE_HEIGHT_M = 0.06 #PLACE parameters
    _RETREAT_HEIGHT_M = 0.14
    _NOT_VISIBLE_DETAIL = "exact object or region instance is not visible"
    _UNLOCALIZED_DETAIL = "reliable 3D grounding is required"

    _STOP_SETTLE_S = 2.0          # total time budget to observe settling     #STOP parameters
    _STOP_SAMPLE_S = 0.1          # spacing between settling checks
    _STOP_BASE_POS_TOL = 0.003    # m, base xy drift per sample
    _STOP_BASE_YAW_TOL = 0.01     # rad, base yaw drift per sample
    _STOP_EE_DRIFT_TOL = 0.005    # m, end-effector drift per sample
    _STOP_STABLE_SAMPLES_REQUIRED = 2

    def _execute(self, action, env, perception):
        if self._initial_base_pose is None:
            self._initial_base_pose = tuple(env.get_base_pose())
        # Final2 stage 1 (perception diet): what the previous action left for
        # this one. Only the immediately preceding action counts.
        self._diet_prev, self._diet_last = self._diet_last, None

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

        elif action.skill is Skill.SEARCH:
            return self._search(action, env, perception)

        elif action.skill is Skill.VERIFY:
            return self._verify(action, env, perception)

        elif action.skill is Skill.STOP:
            return self._stop(action, env)
        
        return super()._execute(action, env, perception)

    @staticmethod
    def _planned_pos(action):
        """True when resolve_action_position would return params["pos"]."""
        return (action.params or {}).get("pos") is not None

    def reset(self):
        super().reset()
        self._arm_tucked = False
        self._initial_base_pose = None
        self._diet_prev = self._diet_last = None

    @staticmethod
    def _table_contacts(env):
        """Read only the existing simulator diagnostics; mocks report none."""
        public_reader = getattr(env, "get_table_contacts", None)
        if callable(public_reader):
            return list(public_reader())
        world = getattr(env, "_world", None)
        robot_contacts = getattr(world, "robot_contacts", None)
        if not callable(robot_contacts):
            return []
        monitored = ("torso_link", "waist_yaw_link", "waist_roll_link",
                     "left_shoulder", "left_elbow", "left_wrist", "lg_")
        contacts = []
        for body_a, body_b, distance in robot_contacts():
            if "table" not in (body_a, body_b):
                continue
            body = body_b if body_a == "table" else body_a
            if body.startswith(monitored):
                contacts.append((body, "table", distance))
        return contacts

    def _contact_guard(self, env):
        return _ContactGuard(env, lambda: self._table_contacts(env))

    def _back_off_from_contact(self, env):
        """Move 0.050 m away from the tableward displacement before recovery."""
        initial = np.asarray(self._initial_base_pose, dtype=float)
        current = np.asarray(env.get_base_pose(), dtype=float)
        displacement = current[:2] - initial[:2]
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-9:
            return

        direction_away = -displacement / distance
        target_xy = current[:2] + direction_away * self._CONTACT_BACKOFF_M
        env.set_base_target(float(target_xy[0]), float(target_xy[1]), float(current[2]))
        deadline = env.sim_time() + min(2.0, skills.DEFAULT_TIMEOUT_S)
        while env.sim_time() < deadline:
            current = np.asarray(env.get_base_pose(), dtype=float)
            if np.linalg.norm(current[:2] - target_xy) < skills.BASE_POS_TOL:
                env.stop_motion()
                return
            env.step(10)
        env.stop_motion()

    def _restore_after_contact(self, env):
        """Back off before the next planned action."""
        env.stop_motion()
        if env.is_attached():
            return False
        self._back_off_from_contact(env)
        return True

    def _tuck_arm(self, action, env):
        if self._arm_tucked:
            return None, 0.0

        # Retract the arm before translating the base.  The arm may still be
        # extended after a previous REACH/GRASP/PLACE action.
        base = env.get_base_pose()
        c, s = np.cos(base[2]), np.sin(base[2])
        local_x, local_y, tuck_z = self._ARM_TUCK_OFFSET
        tuck_pos = np.array([
            base[0] + c * local_x - s * local_y,
            base[1] + s * local_x + c * local_y,
            tuck_z,
        ])
        try:
            env.set_arm_target(tuck_pos)
        except ValueError as exc:
            return ExecutionResult(
                action=action,
                success=False,
                error_code=ErrorCode.UNREACHABLE,
                info={"detail": f"Could not tuck arm before search/approach: {exc}"},
            ), None

        tuck_deadline = env.sim_time() + 5.0
        while env.sim_time() < tuck_deadline:
            if np.linalg.norm(env.get_ee_pos() - tuck_pos) < skills.EE_POS_TOL:
                break
            env.step(10)
        tuck_error = float(np.linalg.norm(env.get_ee_pos() - tuck_pos))
        if tuck_error >= skills.EE_POS_TOL:
            env.stop_motion()
            return ExecutionResult(
                action=action,
                success=False,
                error_code=ErrorCode.TIMEOUT,
                info={"detail": "Arm did not tuck before approach", "tuck_error_m": tuck_error},
            ), tuck_error
        self._arm_tucked = True
        return None, tuck_error

    def _return_to_initial_pose(self, env):
        """Return the base to its pose when this episode first started."""
        initial_pose = np.asarray(self._initial_base_pose, dtype=float)
        env.set_base_target(*initial_pose)

        deadline = env.sim_time() + skills.DEFAULT_TIMEOUT_S
        while env.sim_time() < deadline:
            current = np.asarray(env.get_base_pose(), dtype=float)
            yaw_error = abs(float(np.arctan2(np.sin(current[2] - initial_pose[2]),
                                             np.cos(current[2] - initial_pose[2]))))
            if (np.linalg.norm(current[:2] - initial_pose[:2]) < skills.BASE_POS_TOL
                    and yaw_error < skills.BASE_YAW_TOL):
                env.stop_motion()
                return SkillResult(True, ErrorCode.NONE, {
                    "returned_to_initial_pose": True,
                    "initial_base_pose": initial_pose.tolist(),
                })
            env.step(10)

        env.stop_motion()
        return SkillResult(False, ErrorCode.TIMEOUT, {
            "detail": "Base did not return to its initial pose",
            "initial_base_pose": initial_pose.tolist(),
        })

#APPROACH function: park the base so the target lands inside the right arm's workspace.
    def _approach(self, action, env, perception):
        """Park the base so the target lands inside the right arm's workspace."""
        tuck_result, tuck_error = self._tuck_arm(action, env)
        if tuck_result is not None:
            return tuck_result

        # 1. Fresh observation + scene: never reuse a stale/cached position.
        #    Final2 diet: when the action carries params["pos"] (B's compiler
        #    always sets it from the plan-time scene), resolve_action_position
        #    returns that override and never reads the scene, so no capture.
        #    The position only parks the base; GRASP re-localizes.
        scene = None if self._planned_pos(action) else perception.describe(env.get_obs())

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

        # 3. Run the reference motion primitive with body/table contact checks.
        try:
            primitive_result = skills.approach(self._contact_guard(env), pos)
        except _BodyTableContact as exc:
            recovered = self._restore_after_contact(env)
            if recovered:
                primitive_result = SkillResult(
                    True, ErrorCode.NONE,
                    {"contact_recovery": True,
                     "contacts": exc.contacts,
                     "detail": "Table contact detected; backed off before the next planned action"},
                )
            else:
                primitive_result = SkillResult(
                    False, ErrorCode.TIMEOUT,
                    {"detail": "Table contact detected; recovery was not possible",
                     "contact_recovery": False, "contacts": exc.contacts},
                )

        # 4. Let the base settle before returning, so a following action's
        #    fresh perception capture isn't taken mid-drift.
        env.stop_motion()
        env.step(100)

        # 4b. Final2 3.1: is the target in view from the parking pose? RUN 5
        #     f25: the bottle was cut off at the right image edge after
        #     parking and GRASP failed TARGET_LOST four times. If A cannot
        #     localize it, turn toward it. GRASP's own capture right after
        #     this one is the same instant, so A reuses this answer (no call).
        turn = None
        if primitive_result.success and action.target:
            try:
                turn = self._look_for_target(action, env, perception)
            except TargetResolutionError as exc:
                turn = {"steps": 0, "detail": str(exc)}  # parking itself succeeded
            if turn.get("steps"):
                self._diet_last = {"skill": Skill.APPROACH, "target": action.target, "turned": turn}

        # 5. Convert SkillResult -> ExecutionResult.
        post = env.get_obs()
        return ExecutionResult(
            action=action,
            success=primitive_result.success,
            error_code=primitive_result.error_code,
            post_frame_id=post.frame_id,
            info={**primitive_result.info, "arm_tucked": True, "tuck_error_m": tuck_error,
                  **({"turned_to_find": turn} if turn and turn.get("steps") else {})},
        )

    _TURN_STEP_RAD = math.radians(20)
    _TURN_MAX_STEPS = 3

    @staticmethod
    def _localized(scene, target):
        ref = scene.find(target or "") if scene is not None else None
        return (ref is not None and ref.status is GroundStatus.LOCALIZED and ref.pos_world is not None
                and ref.attributes.get("pos_basis") != "plane_projected")

    def _look_for_target(self, action, env, perception):
        """Final2 3.1: after parking, turn the base in place toward the planned
        target position in <= 20 deg steps (at most 3) until A localizes it.
        Never grasps; returns what happened."""
        scene = self._scene(env, perception)
        info = {"steps": 0, "found": self._localized(scene, action.target)}
        if info["found"]:
            return info
        base = np.asarray(env.get_base_pose(), dtype=float)
        planned = (action.params or {}).get("pos")
        bearing = 0.0
        if planned is not None:
            bearing = math.atan2(planned[1] - base[1], planned[0] - base[0]) - base[2]
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        # Parking puts the target ahead-right, so without a clear bearing turn right.
        direction = 1.0 if bearing > math.radians(5) else -1.0
        info.update(direction="left" if direction > 0 else "right", bearing_deg=math.degrees(bearing))
        guard = self._contact_guard(env)
        for step in range(1, self._TURN_MAX_STEPS + 1):
            yaw = float(base[2] + direction * self._TURN_STEP_RAD * step)
            env.set_base_target(float(base[0]), float(base[1]), yaw)

            def turned():
                b = env.get_base_pose()
                return abs(math.atan2(math.sin(b[2] - yaw), math.cos(b[2] - yaw))) < skills.BASE_YAW_TOL

            try:
                ok = skills._step_until(guard, turned, timeout_s=4.0, chunk=1)
            except _BodyTableContact as exc:
                info.update(contact=True, contacts=exc.contacts)
                self._restore_after_contact(env)
                break
            env.stop_motion()
            env.step(50)
            info["steps"] = step
            if not ok:
                info["detail"] = "turn timed out"
                break
            if self._localized(self._scene(env, perception), action.target):
                info["found"] = True
                break
        return info

    _CARRY_BACKOFF_M = 0.10       # final2 3.4
    _CARRY_BACKOFF_WITHIN_M = 0.45  # only when the base is this close to the table

    @staticmethod
    def _table_clearance(xy):
        """Distance from the table top's footprint (negative inside)."""
        (x0, y0, _), (x1, y1, _) = table_aabb()
        x, y = float(xy[0]), float(xy[1])
        if x0 < x < x1 and y0 < y < y1:
            return -min(x - x0, x1 - x, y - y0, y1 - y)
        return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))

    def _back_away_from_table(self, env):
        """Final2 3.4: move the base 10 cm straight away from the nearest
        point of the table footprint (heading kept), contact-guarded.
        Returns an info dict, or None when the base is already far enough."""
        base = np.asarray(env.get_base_pose(), dtype=float)
        clearance = self._table_clearance(base[:2])
        if clearance >= self._CARRY_BACKOFF_WITHIN_M or clearance <= 0:
            return None
        (x0, y0, _), (x1, y1, _) = table_aabb()
        nearest = np.array([min(max(base[0], x0), x1), min(max(base[1], y0), y1)])
        away = (base[:2] - nearest) / clearance
        goal = base[:2] + away * self._CARRY_BACKOFF_M
        info = {"from_clearance_m": round(clearance, 3), "goal": [round(float(v), 3) for v in goal]}
        env.set_base_target(float(goal[0]), float(goal[1]), float(base[2]))
        try:
            ok = skills._step_until(self._contact_guard(env), lambda: np.linalg.norm(
                np.asarray(env.get_base_pose())[:2] - goal) < skills.BASE_POS_TOL, timeout_s=3.0)
        except _BodyTableContact as exc:
            env.stop_motion()
            return {**info, "success": False, "contacts": exc.contacts}
        env.stop_motion()
        return {**info, "success": bool(ok)}

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
                    or target.pos_world is None
                    or target.attributes.get("pos_basis") == "plane_projected"):  # final2 3.3: verification only
                raise TargetResolutionError(f"Target {action.target!r} needs fresh, unambiguous 3D evidence")
            if target.kind != "object":
                return ExecutionResult(action, False, ErrorCode.INVALID_ACTION,
                                       info={"detail": "GRASP target is not an object"})

            prev = self._diet_prev or {}
            if attempt == 0 and prev.get("skill") is Skill.APPROACH and prev.get("target") == action.target:
                # Final2 3.1: APPROACH had to turn to see the target; the base
                # turned since the plan, so grasp at the FRESH position.
                pos = np.asarray(target.pos_world, dtype=float)
            else:
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
                elif primitive.success:
                    retreat = self._return_to_initial_pose(env)
                    if not retreat.success:
                        # A failed retreat does not undo a verified grasp:
                        # keep success and report the retreat separately.
                        primitive = SkillResult(
                            True,
                            ErrorCode.NONE,
                            {**primitive.info, "return_to_initial_pose": {
                                "success": False,
                                "error_code": retreat.error_code.value,
                                **retreat.info}},
                        )
                    else:
                        primitive = SkillResult(
                            True,
                            ErrorCode.NONE,
                            {**primitive.info, **retreat.info},
                        )
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
        # Final2 diet: no capture when params["pos"] decides the target anyway.
        scene = None if self._planned_pos(action) else self._scene(env, perception)
        pos = resolve_action_position(action, scene)  # also raises on missing/unlocalized

        # Final2 3.4: before the base turns toward the region with the load,
        # back it 10 cm straight away from the table (RUN 5 f23: torso-table
        # contact on every carry from a start pose 30 cm from the table edge).
        # The parking pose itself is unchanged.
        backed = self._back_away_from_table(env)

        try:
            primitive_result = skills.move_to(self._contact_guard(env), pos)
        except _BodyTableContact as exc:
            env.stop_motion()
            primitive_result = SkillResult(
                False, ErrorCode.TIMEOUT,
                {"detail": "Table contact detected during transport; held object preserved",
                 "contacts": exc.contacts},
            )
        recovery_attempted = False
        motion_recovery_info = None
        base_repositioned = None

        # A stalled/unreachable reach can be a consequence of the base's
        # CURRENT parking pose, not a genuinely unreachable target.
        # Re-park once and retry a single reach before giving up.
        if (not primitive_result.success
            and "contacts" not in primitive_result.info
            and primitive_result.error_code in (
            ErrorCode.TIMEOUT, ErrorCode.UNREACHABLE,
            )):
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
        if backed is not None:
            info["carry_backoff"] = backed
        if motion_recovery_info is not None:
            info["motion_recovery"] = motion_recovery_info
            info["base_repositioned"] = base_repositioned

        return ExecutionResult(action, success, error_code,
                            recovery_attempted=recovery_attempted, info=info)

#PLACE function: move above the destination surface (never into it), release, let the object settle, then confirm placement from a fresh observation.
    def _place(self, action, env, perception):
        """PLACE: move above the destination surface, release, let the
        object settle, then confirm placement from a fresh observation.
        If the object simply isn't in view after retreating, try up to two
        nearby headings before accepting failure."""

        if not env.is_attached():
            return ExecutionResult(action, False, ErrorCode.NOT_HOLDING)

        # Final2 diet: no capture when params["pos"] decides the target anyway.
        scene = None if self._planned_pos(action) else self._scene(env, perception)
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

        # Only retry the VIEW for camera-framing problems: "not visible", or
        # visible but unlocalized (e.g. the region box touches the image edge
        # from the post-place pose). "Outside region" / drift are real
        # placement errors that a different heading cannot fix.
        view_problems = (self._NOT_VISIBLE_DETAIL, self._UNLOCALIZED_DETAIL, CONTENT_FILTERED_NOTE)
        view_attempts = 0
        if not check.passed and check.detail in view_problems:
            base = np.asarray(self._view_pose)
            bearing = np.arctan2(region_pos[1] - base[1], region_pos[0] - base[0]) - base[2]
            direction = np.sign(np.sin(bearing)) or 1.0
            offsets = (direction * 0.2, -direction * 0.2)
            if check.detail == self._UNLOCALIZED_DETAIL:
                # First return to the exact viewpoint where the scene was last
                # localized (episode start or last successful SEARCH).
                offsets = (0.0,) + offsets
            for offset in offsets:
                view_attempts += 1
                view = self._restore_view(env, yaw_offset=offset)
                if not view.success:
                    break  # couldn't even get to the new viewpoint; stop trying
                check = closed_loop.verify_placement(env, perception, obj_id, action.target)
                if check.passed or check.detail not in view_problems:
                    break  # resolved, or failing for a different (non-view) reason now

        if check.passed:
            # Final2 diet: an immediately following VERIFY(object_in_region)
            # of the same pair reuses this two-frame check if nothing moved.
            self._diet_last = {"skill": Skill.PLACE, "object": obj_id, "region": action.target,
                               "check": check, "base": np.asarray(env.get_base_pose(), dtype=float),
                               "ee": np.asarray(env.get_ee_pos(), dtype=float)}
        return ExecutionResult(
            action=action,
            success=bool(check.passed),
            error_code=ErrorCode.NONE if check.passed else ErrorCode.PLACE_FAILED,
            recovery_attempted=view_attempts > 0,
            info={"detail": check.detail, "verification_frame_id": check.frame_id,
                  "view_recovery_attempts": view_attempts},
        )

#SEARCH function: rotate the base through nearby viewpoints until perception
#reports a fresh, unambiguous 3D position for the target CLASS (not an
#instance ID — SEARCH only appears in a NEEDS_SEARCH plan, before anything
#has been grounded, so B hands us a canonical name like "stone" or "red_region").
    def _search(self, action, env, perception):
        """SEARCH: bounded scan for a target class.

        skills.search does the actual work: it stops base motion between
        views, re-observes with a fresh image at each stop, calls
        perception.ground(obs, target) on that image, and only accepts a
        result that is LOCALIZED (a valid 3D position), matches the CURRENT
        observation's frame_id (rejects a stale/mismatched grounding), and
        is found within the view/time budget. We don't re-implement any of
        that here — we just call it and translate the result.
        """
        tuck_result, tuck_error = self._tuck_arm(action, env)
        if tuck_result is not None:
            return tuck_result

        target_class = action.target or ""
        if not target_class:
            return ExecutionResult(
                action, False, ErrorCode.INVALID_ACTION,
                info={"detail": "SEARCH requires a target class name"},
            )

        # Bounds match the C guide: up to 13 views, 30 simulated seconds total.
        try:
            primitive = self._table_sweep(self._contact_guard(env), perception, target_class,
                                          max_views=13, timeout_s=30.)
        except _BodyTableContact as exc:
            recovered = self._restore_after_contact(env)
            if recovered and not env.is_attached():
                try:
                    primitive = self._table_sweep(self._contact_guard(env), perception, target_class,
                                                  max_views=13, timeout_s=30.)
                except _BodyTableContact:
                    primitive = SkillResult(False, ErrorCode.TIMEOUT,
                                           {"detail": "Repeated table contact during SEARCH"})
                primitive.info.update({"contact_recovery": True, "contacts": exc.contacts})
            else:
                primitive = SkillResult(False, ErrorCode.TIMEOUT,
                                         {"detail": "Table contact detected; recovery was not possible",
                                          "contacts": exc.contacts})

        if primitive.success:
            # The robot's viewpoint just changed (possibly a lot). Remember
            # this as the new "home" view so a later PLACE view-recovery
            # (_restore_view) returns here, not to wherever the episode
            # originally started from.
            self._view_pose = tuple(env.get_base_pose())

        # views > 1 means the target was NOT visible from the very first
        # (current) viewpoint and the robot actually had to scan for it —
        # worth flagging to B/the orchestrator as more than a trivial check.
        views_taken = primitive.info.get("views", 0)

        return ExecutionResult(
            action=action,
            success=primitive.success,
            error_code=primitive.error_code,
            recovery_attempted=views_taken > 1,
            info={**primitive.info, "arm_tucked": True, "tuck_error_m": tuck_error},
        )

    def _table_sweep(self, env, perception, target, max_views=13, timeout_s=30.):
        """Round 6 (1b): skills.search with tabletop-only viewpoints.

        Same acceptance as skills.search (fresh frame, LOCALIZED, finite
        position; perception errors are SEARCH_FATAL), but the yaws come from
        search_views(): ±60° around the table bearing, views that cannot
        contain the table skipped. The old sweep turned 0.5 rad per view
        through a full circle, so about half its views showed only the floor.
        """
        deadline = env.sim_time() + timeout_s
        env.stop_motion()
        first = env.get_obs()
        views, plan = search_views(env.get_base_pose(), first.t_world_camera, first.intrinsics,
                                   n_views=max_views)
        base = env.get_base_pose()
        info = {"primitive": "search", "views": 0, "sweep": plan}
        last_status = "NOT_FOUND"
        try:
            for yaw, _ in views:
                if env.sim_time() >= deadline:
                    return SkillResult(False, ErrorCode.TIMEOUT, info)
                env.set_base_target(float(base[0]), float(base[1]), yaw)

                def turned():
                    b = env.get_base_pose()
                    return abs(math.atan2(math.sin(b[2] - yaw), math.cos(b[2] - yaw))) < skills.BASE_YAW_TOL

                if not skills._step_until(env, turned, timeout_s=min(4., max(0., deadline - env.sim_time())), chunk=1):
                    return SkillResult(False, ErrorCode.TIMEOUT, {**info, "detail": "Viewpoint motion timed out"})
                env.stop_motion()
                obs = env.get_obs()
                try:
                    found = perception.ground(obs, target)
                except Exception as exc:
                    return SkillResult(False, ErrorCode.SEARCH_FATAL,
                                       {**info, "detail": f"{type(exc).__name__}: {exc}"})
                info["views"] += 1
                if found is None:
                    continue
                last_status = found.status.value
                if found.frame_id >= 0 and found.frame_id != obs.frame_id:
                    return SkillResult(False, ErrorCode.SEARCH_FATAL, {**info, "detail": "Stale grounding"})
                pos = np.asarray(found.pos_world, dtype=float)
                if found.status is GroundStatus.LOCALIZED and pos.shape == (3,) and np.all(np.isfinite(pos)):
                    return SkillResult(True, ErrorCode.NONE, {**info, "grounded": found, "frame_id": obs.frame_id})
            return SkillResult(False, ErrorCode.SEARCH_NOT_FOUND, {**info, "last_ground_status": last_status})
        finally:
            env.stop_motion()

# VERIFY function: check a claimed condition against FRESH evidence.
# Never trust an earlier action's own success flag -- re-observe.
    def _verify(self, action, env, perception):
        condition = (action.params or {}).get("condition")

        if condition == "object_in_region":
            obj_id = action.params.get("object")
            region_id = action.params.get("region")
            if not obj_id or not region_id:
                return ExecutionResult(action, False, ErrorCode.INVALID_ACTION,
                                       info={"detail": "object_in_region VERIFY needs object and region IDs"})
            # verify_placement does its own two-frame fresh check (release,
            # region containment, <=2cm drift over 0.2s) -- we don't
            # duplicate that logic, just convert its result.
            prev = self._diet_prev or {}
            if (prev.get("skill") is Skill.PLACE and prev.get("object") == obj_id
                    and prev.get("region") == region_id and not env.is_attached()
                    and np.linalg.norm(np.asarray(env.get_base_pose()) - prev["base"]) < 1e-3
                    and np.linalg.norm(np.asarray(env.get_ee_pos()) - prev["ee"]) < 5e-3):
                # Final2 diet: PLACE's own two-frame check of this exact pair
                # just passed and nothing has moved since. The orchestrator's
                # final verification still captures fresh frames.
                check = prev["check"]
                return ExecutionResult(
                    action=action, success=True, error_code=ErrorCode.NONE,
                    post_frame_id=check.frame_id,
                    info={"detail": check.detail, "condition": condition, "object": obj_id,
                          "region": region_id, "reused_place_check": True},
                )
            check = verify_placement(env, perception, obj_id, region_id)
            return ExecutionResult(
                action=action,
                success=bool(check.passed),
                error_code=ErrorCode.NONE if check.passed else ErrorCode.VERIFY_FAILED,
                post_frame_id=check.frame_id,
                info={"detail": check.detail, "condition": condition,
                      "object": obj_id, "region": region_id},
            )

        if condition == "holding":
            # env.is_attached() alone proves SOMETHING is held, not that it's
            # the object C believes it is holding -- check the tracked ID too.
            ok = env.is_attached() and bool(action.target) and action.target == self._held_id
            return ExecutionResult(
                action=action,
                success=ok,
                error_code=ErrorCode.NONE if ok else ErrorCode.VERIFY_FAILED,
                info={"condition": condition, "target": action.target,
                      "attached": env.is_attached(), "held_instance_id": self._held_id},
            )

        if condition == "object_visible":
            # 2D evidence is enough here -- UNLOCALIZED still counts as visible,
            # per core/validation.py's ref_errors(need_located=False).
            scene = self._scene(env, perception)  # fresh obs, staleness-checked
            ref = scene.find(action.target or "")
            ok = (ref is not None
                  and ref.status in (GroundStatus.LOCALIZED, GroundStatus.UNLOCALIZED)
                  and (ref.frame_id < 0 or ref.frame_id == scene.frame_id))
            return ExecutionResult(
                action=action,
                success=ok,
                error_code=ErrorCode.NONE if ok else ErrorCode.VERIFY_FAILED,
                info={"condition": condition, "target": action.target,
                      "status": ref.status.value if ref else "NOT_FOUND"},
            )

        return ExecutionResult(action, False, ErrorCode.INVALID_ACTION,
                               info={"detail": f"Unknown VERIFY condition {condition!r}"})

# STOP function: cancel motion and confirm the robot actually settled.
# env.stop_motion() only cancels targets — it does not prove anything has
# stopped moving. A body under load (e.g. from residual velocity or a
# held object's inertia) can keep drifting for a bit afterward. We treat
# STOP as successful only once we've directly observed two consecutive
# quiet samples, not just because the command was issued.
    def _stop(self, action, env):
        """STOP: cancel all commanded motion, then confirm settling from
        fresh proprioception before declaring success. Attachment is never
        dropped by this action; if it's lost anyway (e.g. the object was
        already slipping), that's reported as a failure, not silently
        ignored."""

        env.stop_motion()
        was_attached = env.is_attached()

        deadline = env.sim_time() + self._STOP_SETTLE_S
        stable_samples = 0
        before = env.get_robot_state()
        info = {}

        while env.sim_time() < deadline:
            # Step a fixed slice of sim time between checks rather than a
            # fixed step count, so the sampling interval is stable regardless
            # of the simulator's timestep.
            env.step(max(1, round(self._STOP_SAMPLE_S / env.timestep())))
            after = env.get_robot_state()

            base_delta = np.asarray(after.base_pose) - np.asarray(before.base_pose)
            yaw_drift = abs(float(np.arctan2(np.sin(base_delta[2]), np.cos(base_delta[2]))))
            base_drift = float(np.linalg.norm(base_delta[:2]))
            ee_drift = float(np.linalg.norm(np.asarray(after.ee_pos) - np.asarray(before.ee_pos)))

            settled = (base_drift < self._STOP_BASE_POS_TOL
                    and yaw_drift < self._STOP_BASE_YAW_TOL
                    and ee_drift < self._STOP_EE_DRIFT_TOL)
            stable_samples = stable_samples + 1 if settled else 0

            info = {"base_drift_m": base_drift, "yaw_drift_rad": yaw_drift,
                    "ee_drift_m": ee_drift, "stable_samples": stable_samples}

            # Attachment must not silently disappear during a stop. Report it
            # immediately rather than waiting out the rest of the settle budget.
            if was_attached and not env.is_attached():
                return ExecutionResult(action, False, ErrorCode.NOT_HOLDING, info=info)

            if stable_samples >= self._STOP_STABLE_SAMPLES_REQUIRED:
                return ExecutionResult(action, True, info=info)

            before = after

        # Ran out of settle budget without two consecutive quiet samples.
        return ExecutionResult(action, False, ErrorCode.TIMEOUT, info=info)

 