"""Final2 stage 1: C skips perception captures whose result no decision uses."""
import numpy as np

from core import skills
from core.types import Action, ErrorCode, GroundStatus, Skill, SkillResult, VerificationResult
from executor.student_c import StudentCExecutor
from tests.test_student_c import ScenePerception, SensorEnv, move_success


class ArmEnv(SensorEnv):
    """SensorEnv plus the arm/attach calls APPROACH/GRASP/PLACE use; logs order."""

    def __init__(self):
        super().__init__()
        self.log = []

    def get_obs(self):
        obs = super().get_obs()
        self.log.append(("obs", obs.frame_id))
        return obs

    def set_arm_target(self, pos):
        self.log.append(("arm", None))
        self.ee = np.asarray(pos, dtype=float).copy()


class LoggingPerception(ScenePerception):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def describe(self, obs):
        self.env.log.append(("describe", obs.frame_id))
        scene = super().describe(obs)
        if self.env.attached is False and getattr(self.env, "placed", False):
            scene.objects[0].pos_world = (.4, .3, .88)  # the stone now lies on the region
        return scene


def _pipeline(monkeypatch):
    env = ArmEnv()
    c = StudentCExecutor()
    perception = LoggingPerception(env)
    monkeypatch.setattr(skills, "approach", lambda env, pos: SkillResult(True))
    monkeypatch.setattr(skills, "move_to", move_success)
    monkeypatch.setattr(skills, "reach", move_success)

    def grasp(env, pos):
        env.attached = True
        return SkillResult(True)

    def release(env):
        env.attached = False
        env.placed = True
        return SkillResult(True)

    monkeypatch.setattr(skills, "grasp", grasp)
    monkeypatch.setattr(skills, "place", release)
    return env, c, perception


def _describes(perception, start):
    return len(perception.frames) - start


STONE, REGION, CARRY = [.4, -.15, .88], [.4, .3, .853], [.4, .3, 1.033]


def _plan(with_pos):
    pos = (lambda p: {"pos": p}) if with_pos else (lambda p: {})
    return [Action(Skill.APPROACH, "p0", pos(STONE)), Action(Skill.GRASP, "p0", pos(STONE)),
            Action(Skill.MOVE_TO, "p2", pos(CARRY)), Action(Skill.PLACE, "p2", {"object": "p0", **pos(REGION)}),
            Action(Skill.VERIFY, None, {"condition": "object_in_region", "object": "p0", "region": "p2"})]


def _run(monkeypatch, plan):
    env, c, perception = _pipeline(monkeypatch)
    per_action = {}
    for action in plan:
        start = len(perception.frames)
        result = c.execute(action, env, perception)
        assert result.success, (action.skill, result.info)
        per_action[action.skill] = _describes(perception, start)
    return per_action


def test_call_sequence_of_a_compiled_plan(monkeypatch):
    # B's compiler sets params.pos on APPROACH/GRASP/MOVE_TO/PLACE. APPROACH,
    # MOVE_TO and PLACE never read a scene then; GRASP still re-localizes;
    # PLACE keeps its two-frame verification; VERIFY reuses PLACE's check.
    assert _run(monkeypatch, _plan(True)) == {Skill.APPROACH: 0, Skill.GRASP: 1, Skill.MOVE_TO: 0,
                                             Skill.PLACE: 2, Skill.VERIFY: 0}


def test_without_planned_positions_every_action_localizes(monkeypatch):
    assert _run(monkeypatch, _plan(False)) == {Skill.APPROACH: 1, Skill.GRASP: 1, Skill.MOVE_TO: 1,
                                              Skill.PLACE: 3, Skill.VERIFY: 0}


def test_place_releases_at_the_region_not_at_the_carry_height(monkeypatch):
    env, c, perception = _pipeline(monkeypatch)
    releases = []
    monkeypatch.setattr(skills, "move_to", lambda env, pos: (releases.append(np.asarray(pos)), move_success(env, pos))[1])
    for action in _plan(True)[:3]:
        assert c.execute(action, env, perception).success
    offset = c._held_offset.copy()
    assert c.execute(_plan(True)[3], env, perception).success
    # PLACE's move: region support point + 6 cm release height (not + carry clearance)
    assert np.allclose(releases[-1], np.asarray(REGION) + [0, 0, c._RELEASE_HEIGHT_M] - offset)


def test_verify_recaptures_when_anything_intervened_or_moved(monkeypatch):
    env, c, perception = _pipeline(monkeypatch)
    env.attached, c._held_id = True, "p0"
    verify = Action(Skill.VERIFY, None, {"condition": "object_in_region", "object": "p0", "region": "p2"})
    assert c.execute(Action(Skill.PLACE, "p2", {"object": "p0"}), env, perception).success
    env.base = env.base + [0.05, 0, 0]  # the base moved after PLACE
    start = len(perception.frames)
    result = c.execute(verify, env, perception)
    assert result.success and "reused_place_check" not in result.info
    assert _describes(perception, start) == 2
    # A STOP in between also breaks the reuse (only the previous action counts).
    env.attached, c._held_id = True, "p0"
    assert c.execute(Action(Skill.PLACE, "p2", {"object": "p0"}), env, perception).success
    c.execute(Action(Skill.STOP), env, perception)
    start = len(perception.frames)
    assert c.execute(verify, env, perception).success
    assert _describes(perception, start) == 2


def test_failed_place_is_never_reused(monkeypatch):
    import executor.closed_loop as module
    env, c, perception = _pipeline(monkeypatch)
    env.attached, c._held_id = True, "p0"
    monkeypatch.setattr(module, "verify_placement",
                        lambda *a: VerificationResult(False, "object_in_region", "outside region", 1))
    assert not c.execute(Action(Skill.PLACE, "p2", {"object": "p0"}), env, perception).success
    monkeypatch.undo()
    _, _, _ = _pipeline(monkeypatch)
    result = c.execute(Action(Skill.VERIFY, None, {"condition": "object_in_region",
                                                    "object": "p0", "region": "p2"}), env, perception)
    assert "reused_place_check" not in result.info


def test_real_sim_capture_without_stepping_is_a_near_duplicate(world, env):
    """The APPROACH reuse rests on this: two captures of an unchanged state
    render (almost) the same image, so A's prediction reuse answers. EGL
    noise changes a handful of pixels by one level at most."""
    from perception.student_a import StudentAPerception
    from tests.conftest import standard_scene
    world.reset(standard_scene())
    frames = [env.get_obs() for _ in range(4)]
    for a, b in zip(frames, frames[1:]):
        diff = np.abs(a.rgb.astype(int) - b.rgb.astype(int))
        assert diff.max() <= StudentAPerception.NEAR_DUPLICATE_MAX_LEVEL
        assert np.count_nonzero(diff.max(axis=-1)) <= StudentAPerception.NEAR_DUPLICATE_MAX_PIXELS
