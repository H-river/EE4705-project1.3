"""Bonus: the learned grasp skill behind the grasp interface (stub policies, real MuJoCo)."""
import numpy as np
import pytest

from core import skills
from core.types import Action, ErrorCode, Skill
from executor import learned_grasp
from executor.learned_grasp import LearnedGraspSkill, grasp_primitive, target_in_base


class StubPolicy:
    """Moves the arm toward a fixed joint goal at a bounded rate (no image)."""
    needs_image = False

    def __init__(self, q_goal=None, value=None, max_step=0.2):
        self.q_goal, self.value, self.max_step = q_goal, value, max_step
        self.calls, self.resets, self.states = 0, 0, []

    def reset(self):
        self.resets += 1

    def __call__(self, state, image):
        self.calls += 1
        self.states.append(state.copy())
        assert image is None and state.shape == (10,)
        if self.value is not None:
            return np.full(7, self.value)
        q = state[:7]
        if self.q_goal is None:
            return q
        return q + np.clip(self.q_goal - q, -self.max_step, self.max_step)


def _grasp_goal_q(world, pos):
    res, _ = world.robot.solve_arm_target(pos + np.array([0.0, 0.0, skills.GRASP_DESCEND_OFFSET]))
    assert res.success
    return res.q


def test_selector_defaults_to_script(monkeypatch):
    monkeypatch.delenv("EE4705_GRASP_POLICY", raising=False)
    assert grasp_primitive() is skills.grasp
    monkeypatch.setenv("EE4705_GRASP_POLICY", "nonsense")
    with pytest.raises(ValueError):
        grasp_primitive()


def test_stub_policy_reaches_and_attaches_like_the_script(standard_world, env, oracle):
    stone = oracle.object_pos("stone")
    assert skills.approach(env, stone).success
    stub = StubPolicy(q_goal=_grasp_goal_q(standard_world, stone))
    r = LearnedGraspSkill(stub)(env, stone)
    assert r.success and r.error_code is ErrorCode.NONE, r
    assert standard_world.attached_body_name() == "stone"
    assert r.info["stop"] in ("reached", "settled") and stub.resets == 1
    # the state is (q, target in the base frame)
    assert np.allclose(stub.states[-1][7:], target_in_base(stone, env.get_base_pose()), atol=0.03)


def test_idle_policy_settles_and_misses(standard_world, env, oracle):
    stone = oracle.object_pos("stone")
    assert skills.approach(env, stone).success
    r = LearnedGraspSkill(StubPolicy())(env, stone)
    assert not r.success and r.error_code is ErrorCode.GRASP_MISSED
    assert r.info["stop"] == "settled" and r.info["rollout_s"] < 2.0
    assert not env.is_attached()


def test_invalid_action_and_already_holding(standard_world, env, oracle):
    stone = oracle.object_pos("stone")
    assert skills.approach(env, stone).success
    r = LearnedGraspSkill(StubPolicy(value=np.nan))(env, stone)
    assert not r.success and r.info["detail"] == "invalid policy action"
    assert skills.grasp(env, stone).success
    r = LearnedGraspSkill(StubPolicy())(env, stone)
    assert r.error_code is ErrorCode.ALREADY_HOLDING


def test_time_limit_bounds_the_rollout(standard_world, env, oracle):
    stone = oracle.object_pos("stone")
    assert skills.approach(env, stone).success
    # oscillating commands never settle and never reach
    class Wobble(StubPolicy):
        def __call__(self, state, image):
            self.calls += 1
            return state[:7] + (0.05 if self.calls % 2 else -0.05)
    t0 = env.sim_time()
    r = LearnedGraspSkill(Wobble(), max_s=2.0)(env, stone)
    assert r.info["stop"] == "timeout" and env.sim_time() - t0 < 2.5 and not r.success


def test_executor_grasp_goes_through_the_selected_policy(standard_world, env, oracle, monkeypatch):
    """StudentCExecutor's GRASP (post-conditions unchanged) runs the learned skill."""
    from executor.student_c import StudentCExecutor
    from core.mocks import GTPerception

    stone = oracle.object_pos("stone")
    assert skills.approach(env, stone).success
    stub = StubPolicy(q_goal=_grasp_goal_q(standard_world, stone))
    monkeypatch.setenv("EE4705_GRASP_POLICY", "act")
    monkeypatch.delenv("EE4705_GRASP_CKPT", raising=False)
    monkeypatch.delenv("EE4705_GRASP_KWARGS", raising=False)
    monkeypatch.setitem(learned_grasp._CACHE, ("act", "", ""), LearnedGraspSkill(stub))
    perception = GTPerception(oracle, mode="omniscient")
    scene = perception.describe(env.get_obs())
    target = next(o for o in scene.objects if o.name == "stone")
    result = StudentCExecutor().execute(Action(Skill.GRASP, target.instance_id), env, perception)
    assert stub.calls > 0
    assert result.success, result
    assert standard_world.attached_body_name() == "stone"
