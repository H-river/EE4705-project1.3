"""C postconditions and recovery: public sensor doubles plus real MuJoCo integration."""
from types import SimpleNamespace

import numpy as np
import pytest

from core import skills
from core.types import Action, ErrorCode, GroundedObject, GroundStatus, SceneDescription, Skill, SkillResult
from executor.student_c import StudentCExecutor


class SensorEnv:
    def __init__(self):
        self.time, self.frame = 0., 0
        self.ee = np.array([.4, -.15, .9])
        self.base = np.zeros(3)
        self.base_target = self.base.copy()
        self.attached = False
        self.opening = 1.
        self.stops = self.steps = self.turns = 0
        self.stalled = self.drifting = False

    def timestep(self): return .002
    def sim_time(self): return self.time
    def get_base_pose(self): return self.base.copy()
    def get_ee_pos(self): return self.ee.copy()
    def is_attached(self): return self.attached
    def get_obs(self):
        self.frame += 1
        return SimpleNamespace(frame_id=self.frame, sim_time=self.time)
    def set_base_target(self, *values):
        self.base_target = np.array(values)
        self.turns += 1
    def stop_motion(self):
        self.stops += 1
        self.base_target = self.base.copy()
    def step(self, n):
        self.steps += 1
        self.time += n*self.timestep()
        if not self.stalled: self.base = self.base_target.copy()
        if self.drifting: self.ee[0] += .02
    def set_gripper(self, side, opening): self.opening = opening
    def get_robot_state(self):
        return SimpleNamespace(base_pose=tuple(self.base), ee_pos=tuple(self.ee),
                               gripper_opening={'right':self.opening}, last_attach_reason='test_sensor')


class ScenePerception:
    def __init__(self, status=GroundStatus.LOCALIZED, missing=False, stale=False):
        self.status, self.missing, self.stale = status, missing, stale
        self.frames = []

    def describe(self, obs):
        self.frames.append(obs.frame_id)
        frame = obs.frame_id - int(self.stale)
        obj = GroundedObject('p0','stone',self.status,pos_world=(.4,-.15,.88),frame_id=frame)
        region = GroundedObject('p2','red_region',GroundStatus.LOCALIZED,kind='region',
                                pos_world=(.4,.3,.853),frame_id=frame)
        return SceneDescription([] if self.missing else [obj], [region], frame_id=frame, sim_time=obs.sim_time)

    def ground(self, obs, target):
        scene = self.describe(obs)
        return scene.objects[0] if scene.objects else None


def move_success(env, pos):
    env.ee = np.asarray(pos).copy()
    return SkillResult(True)


@pytest.mark.parametrize('status', [GroundStatus.AMBIGUOUS, GroundStatus.UNLOCALIZED, GroundStatus.NOT_FOUND])
def test_search_requires_localized_evidence_and_no_unobserved_final_turn(status):
    env = SensorEnv()
    result = skills.search(env, ScenePerception(status), 'stone', max_views=2)
    assert not result.success and result.error_code is ErrorCode.SEARCH_NOT_FOUND
    assert result.info['views'] == 2 and env.turns == 1
    assert np.array_equal(env.base, env.base_target)


def test_search_turn_obeys_global_deadline_and_reports_stale_perception():
    env = SensorEnv(); env.stalled = True
    result = skills.search(env, ScenePerception(missing=True), 'stone', timeout_s=.1)
    assert result.error_code is ErrorCode.TIMEOUT and env.time <= .102
    result = skills.search(SensorEnv(), ScenePerception(stale=True), 'stone')
    assert result.error_code is ErrorCode.SEARCH_FATAL


@pytest.mark.parametrize('missing,stale,code', [(True,False,ErrorCode.TARGET_LOST),
                                             (False,True,ErrorCode.PERCEPTION_ERROR)])
def test_grasp_rejects_missing_or_stale_target_even_with_old_coordinates(missing,stale,code,monkeypatch):
    def forbidden(*args): raise AssertionError('No motion should be attempted')
    monkeypatch.setattr(skills,'grasp',forbidden)
    env=SensorEnv()
    result=StudentCExecutor().execute(Action(Skill.GRASP,'p0',{'pos':[.4,-.15,.88]}),env,
                                     ScenePerception(missing=missing,stale=stale))
    assert not result.success and result.error_code is code
    assert env.steps == 0 and env.stops > 0 and result.post_frame_id >= 0


def test_grasp_retries_once_with_new_frame_and_checks_lift(monkeypatch):
    attempts=[]
    def grasp(env,pos):
        attempts.append(np.asarray(pos).copy())
        if len(attempts)==1:return SkillResult(False,ErrorCode.GRASP_MISSED)
        env.attached=True
        return SkillResult(True)
    monkeypatch.setattr(skills,'grasp',grasp)
    monkeypatch.setattr(skills,'move_to',move_success)
    env,perception=SensorEnv(),ScenePerception()
    result=StudentCExecutor().execute(Action(Skill.GRASP,'p0'),env,perception)
    assert result.success and result.recovery_attempted and len(attempts)==2
    assert len(set(perception.frames))==2
    assert result.info['held_instance_id']=='p0' and result.info['attached_after']
    assert env.ee[2] >= 1.02


def test_grasp_lift_failure_preserves_attachment_and_failure(monkeypatch):
    def grasp(env,pos):
        env.attached=True
        return SkillResult(True)
    monkeypatch.setattr(skills,'grasp',grasp)
    monkeypatch.setattr(skills,'move_to',lambda *args:SkillResult(False,ErrorCode.TIMEOUT))
    result=StudentCExecutor().execute(Action(Skill.GRASP,'p0'),SensorEnv(),ScenePerception())
    assert not result.success and result.error_code is ErrorCode.TIMEOUT
    assert result.info['attached_after'] and result.info['held_instance_id']=='p0'


def test_false_motion_success_and_dropped_transport_do_not_pass(monkeypatch):
    monkeypatch.setattr(skills,'reach',lambda *args:SkillResult(True))
    result=StudentCExecutor().execute(Action(Skill.REACH,'p0'),SensorEnv(),ScenePerception())
    assert not result.success and result.info['ee_error_m'] >= skills.EE_POS_TOL
    def drop(env,pos):
        env.attached=False
        return move_success(env,pos)
    monkeypatch.setattr(skills,'move_to',drop)
    env=SensorEnv();env.attached=True
    result=StudentCExecutor().execute(Action(Skill.MOVE_TO,'p2'),env,ScenePerception())
    assert not result.success and result.error_code is ErrorCode.NOT_HOLDING


def test_place_does_not_overwrite_detach_failure_with_retreat(monkeypatch):
    env,c=SensorEnv(),StudentCExecutor();env.attached=True;c._held_id='p0'
    monkeypatch.setattr(skills,'move_to',move_success)
    monkeypatch.setattr(skills,'place',lambda env:SkillResult(False,ErrorCode.PLACE_FAILED,{'detail':'stuck'}))
    def forbidden(*args): raise AssertionError('Retreat must not mask release failure')
    monkeypatch.setattr(skills,'reach',forbidden)
    result=c.execute(Action(Skill.PLACE,'p2',{'object':'p0'}),env,ScenePerception())
    assert result.error_code is ErrorCode.PLACE_FAILED and result.info['detail']=='stuck'
    assert env.attached


def test_visibility_does_not_need_depth_but_holding_needs_the_tracked_id():
    env,c=SensorEnv(),StudentCExecutor()
    result=c.execute(Action(Skill.VERIFY,'p0',{'condition':'object_visible'}),env,
                     ScenePerception(GroundStatus.UNLOCALIZED))
    assert result.success
    env.attached=True;c._held_id='p0'
    assert c.execute(Action(Skill.VERIFY,'p0',{'condition':'holding'}),env,None).success
    assert not c.execute(Action(Skill.VERIFY,'p1',{'condition':'holding'}),env,None).success


def test_stop_requires_settling_and_preserves_attachment():
    env=SensorEnv();env.attached=True
    result=StudentCExecutor().execute(Action(Skill.STOP),env,None)
    assert result.success and result.info['stable_samples']==2 and env.attached
    env.drifting=True
    result=StudentCExecutor().execute(Action(Skill.STOP),env,None)
    assert not result.success and result.error_code is ErrorCode.TIMEOUT
    assert result.info['elapsed_sim_s'] <= 2.11


def test_real_student_c_transfer_with_demo_rgbd_a(tmp_path):
    from demo.run import run_episode
    result=run_episode(tmp_path/'c',students=['C'],video=False)
    assert result['claimed_success'] and result['actual_success'], result['episode']
    placements=[e['data']['result'] for e in result['events']
                if e['type']=='C.end' and e['data']['action']['skill']=='PLACE']
    assert len(placements)==1 and placements[0]['success']
    assert 'stable across' in placements[0]['info']['detail']


def test_place_reobserves_occlusion_but_does_not_accept_bad_geometry(monkeypatch):
    import executor.closed_loop as module
    from core.types import VerificationResult
    env,c=SensorEnv(),StudentCExecutor();env.attached=True;c._held_id='p0'
    monkeypatch.setattr(skills,'move_to',move_success)
    monkeypatch.setattr(skills,'reach',move_success)
    def release(env):
        env.attached=False
        return SkillResult(True)
    monkeypatch.setattr(skills,'place',release)
    checks=[]
    def verify(*args):
        checks.append(True)
        return VerificationResult(False,'object_in_region',
            'exact object or region instance is not visible' if len(checks)==1 else 'outside region',1)
    monkeypatch.setattr(module,'verify_placement',verify)
    result=c.execute(Action(Skill.PLACE,'p2',{'object':'p0'}),env,ScenePerception())
    assert not result.success and result.error_code is ErrorCode.PLACE_FAILED
    assert result.recovery_attempted and len(checks)==2
    assert result.info['detail']=='outside region' and result.info['attached_after'] is False


def test_transport_changes_base_pose_once_after_a_stalled_reach(monkeypatch):
    env,c=SensorEnv(),StudentCExecutor();env.attached=True;c._held_id='p0'
    calls=[]
    monkeypatch.setattr(skills,'move_to',lambda *args:SkillResult(False,ErrorCode.TIMEOUT))
    def park(env,pos):
        calls.append('park')
        env.base[0]=.1
        return SkillResult(True)
    monkeypatch.setattr(skills,'approach',park)
    monkeypatch.setattr(skills,'reach',move_success)
    result=c.execute(Action(Skill.MOVE_TO,'p2'),env,ScenePerception())
    assert result.success and result.recovery_attempted and calls==['park']
    assert result.info['base_repositioned'] and result.info['motion_recovery']['error_code']=='TIMEOUT'
    assert result.info['ee_error_m'] < skills.EE_POS_TOL and env.attached
