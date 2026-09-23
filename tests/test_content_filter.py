"""Provider content-filter refusals (HTTP 400) are not fatal; no network."""
import json

import pytest

from core.llm_client import APIError, ContentFiltered, FakeTransport, LLMClient, LLMConfig
from core.types import CONTENT_FILTERED_NOTE
from core.verification import verify_placement
from perception.config import VisionConfig
from perception.run import offline_observation, offline_wire
from perception.student_a import StudentAPerception

# The body DashScope returned in round 2 (c_2_06_cube, PLACE verification).
FILTERED = (400, {'error': {'code': 'data_inspection_failed', 'type': 'data_inspection_failed',
                            'message': 'Input data may contain inappropriate content. For details, see: '
                                       'https://www.alibabacloud.com/help/en/model-studio/error-code'
                                       '#inappropriate-content'}})


def client(tmp_path, replies):
    config = LLMConfig('test-vlm', 'https://fixture.invalid/v1', api_key='test-secret',
                       max_retries=2, retry_backoff_s=0, cache_dir=str(tmp_path / 'cache'))
    transport = FakeTransport(replies)
    return LLMClient(config, transport=transport), transport


def make_a(tmp_path, replies):
    config = VisionConfig(LLMConfig('test-vlm', 'https://fixture.invalid/v1', api_key='test-secret',
                                    max_retries=0, retry_backoff_s=0), str(tmp_path / 'audit'))
    transport = FakeTransport(replies)
    return StudentAPerception(config, LLMClient(config.llm, transport=transport)), transport


@pytest.mark.parametrize('body', [FILTERED[1],
                                  {'error': {'message': 'blocked', 'code': 'content_filter'}}])
def test_content_filter_400_returns_typed_result_not_cached(tmp_path, body):
    c, transport = client(tmp_path, [(400, body), FakeTransport.completion('{"ok": 1}')])
    r = c.call_vlm(b'png', 'describe', json_schema={'type': 'object'})
    assert isinstance(r, ContentFiltered) and r.parsed is None and r.attempts == 1
    assert 'HTTP 400' in r.detail and len(transport.requests) == 1  # a 400 is not retried
    assert not list((tmp_path / 'cache').glob('*.json'))
    assert c.call_vlm(b'png', 'describe', json_schema={'type': 'object'}).parsed == {'ok': 1}


def test_other_400_is_still_fatal(tmp_path):
    c, _ = client(tmp_path, [(400, {'error': {'message': 'invalid image format'}})])
    with pytest.raises(APIError, match='HTTP 400'):
        c.call_vlm(b'png', 'describe')


def test_a_maps_content_filter_to_empty_frame_and_retries_next_time(tmp_path):
    obs = offline_observation()
    a, transport = make_a(tmp_path, [FILTERED, FILTERED, FakeTransport.completion(json.dumps(offline_wire()))])
    scene = a.describe(obs)
    assert scene.objects == [] and scene.regions == [] and CONTENT_FILTERED_NOTE in scene.ambiguities
    assert scene.frame_id == obs.frame_id
    assert a.last_diagnostics['status'] == 'content_filtered'
    assert a.ground(obs, 'gray stone') is None
    # The refusal is not remembered: the same frame is sent again and can pass.
    assert len(a.describe(obs).objects) == 2 and len(transport.requests) == 3


class _Env:
    def __init__(self, obs):
        self.obs = obs

    def get_obs(self):
        return self.obs

    def step(self, n):
        pass

    def timestep(self):
        return 0.002

    def is_attached(self):
        return False


def test_verify_placement_is_undecided_on_content_filter(tmp_path):
    obs = offline_observation()
    a, _ = make_a(tmp_path, [FILTERED])
    result = verify_placement(_Env(obs), a, 'a0', 'a1')
    assert result.passed is None and result.source == 'vision'
    assert result.detail == CONTENT_FILTERED_NOTE and result.frame_id == obs.frame_id
