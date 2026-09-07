"""A interface and geometry checks with canned model responses; no API accuracy claim."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from core.llm_client import APIError, FakeTransport, LLMClient, LLMConfig, SchemaError
from core.types import GroundStatus
from perception.config import VisionConfig
from perception.depth_geometry import localize, pixel_box
from perception.run import offline_observation, offline_wire
from perception.student_a import StudentAPerception
from perception.vision_contract import validate_wire


@pytest.fixture(scope='module')
def capture():
    return offline_observation()


def make_a(tmp_path, wires):
    replies = [FakeTransport.completion(json.dumps(w)) if isinstance(w, dict) else w for w in wires]
    transport = FakeTransport(replies)
    config = VisionConfig(LLMConfig('test-vlm', 'https://fixture.invalid/v1', api_key='test-secret',
                                    max_retries=0, retry_backoff_s=0), str(tmp_path/'audit'))
    return StudentAPerception(config, LLMClient(config.llm, transport=transport)), transport


def changed(obs, frame):
    # Different RGB bytes force a new model response without changing object pixels.
    rgb=obs.rgb.copy();rgb[0,0,0]=frame
    return replace(obs, rgb=rgb, frame_id=frame, sim_time=obs.sim_time+frame)


def test_real_rgbd_geometry_from_hand_labelled_boxes(capture,tmp_path):
    a,transport=make_a(tmp_path,[offline_wire()])
    scene=a.describe(capture)
    assert len(scene.objects)==2 and len(scene.regions)==1
    expected={'stone':(.4,-.15,.875),'cube':(.4,.15,.875),'red_region':(.4,.3,.853)}
    for g in scene.objects+scene.regions:
        assert g.status is GroundStatus.LOCALIZED
        assert np.linalg.norm(np.asarray(g.pos_world)-expected[g.name])<.004
        assert g.instance_id not in expected and g.source=='qwen_vlm:fixture'
    payload=transport.requests[0]['payload']
    assert payload['response_format']['type']=='json_object'
    text=payload['messages'][-1]['content'][0]['text']
    assert 'pos_world' not in text and 'intrinsics' not in text
    assert a.last_diagnostics['real_api_requests']==0


def test_same_image_reuse_recomputes_depth_and_updates_frame(capture,tmp_path):
    a,transport=make_a(tmp_path,[offline_wire()])
    before=a.describe(capture)
    after=a.describe(replace(capture,depth=np.full_like(capture.depth,np.nan),frame_id=7,sim_time=2.))
    assert len(transport.requests)==1 and a.last_diagnostics['prediction_reused']
    assert [g.instance_id for g in before.objects]==[g.instance_id for g in after.objects]
    assert all(g.frame_id==7 and g.status is GroundStatus.UNLOCALIZED and g.pos_world is None
               for g in after.objects+after.regions)
    assert after.frame_id==7 and after.sim_time==2.


def test_ids_stay_stable_when_order_changes_missing_objects_not_carried(capture,tmp_path):
    wire=offline_wire();reordered=deepcopy(wire);reordered['detections'].reverse()
    missing={'detections':[], 'selected':[], 'answer':'empty'}
    a,_=make_a(tmp_path,[wire,reordered,missing,wire])
    first=a.describe(capture);second=a.describe(changed(capture,1))
    ids=lambda s:{g.name:g.instance_id for g in s.objects+s.regions}
    assert ids(first)==ids(second)
    assert a.describe(changed(capture,2)).objects==[]
    assert ids(a.describe(changed(capture,3)))==ids(first)
    a.reset();assert not a._tracks and not a._predictions


def test_ground_missing_ambiguous_and_known_exact_id(capture,tmp_path):
    missing=deepcopy(offline_wire());missing['selected']=[]
    ambiguous=deepcopy(offline_wire());ambiguous['selected']=[0,1]
    a,_=make_a(tmp_path,[missing,ambiguous,offline_wire()])
    assert a.ground(capture,'invisible bottle') is None
    g=a.ground(capture,'an object')
    assert g.status is GroundStatus.AMBIGUOUS and g.pos_world is None
    known=a.describe(capture).objects[0]
    assert a.ground(capture,known.instance_id).instance_id==known.instance_id


@pytest.mark.parametrize('mutation',[
    lambda w:w['detections'][0].update(bbox=[20,20,10,30]),
    lambda w:w.update(selected=[999]),
    lambda w:w.update(selected=[-1]),
    lambda w:w['detections'][0].update(confidence=1.1),
    lambda w:w.update(selected=[0,0]),
    lambda w:w['detections'][0].update(bbox=[-1,20,30,40]),
    lambda w:w['detections'][0].update(confidence=float('nan')),
])
def test_bad_wire_rejected(mutation):
    wire=offline_wire();mutation(wire)
    with pytest.raises((ValueError,SchemaError)):validate_wire(wire)


def test_one_schema_repair_retains_invalid_response(capture,tmp_path):
    bad=offline_wire();bad['selected']=[10]
    a,transport=make_a(tmp_path,[bad,offline_wire()])
    assert a.describe(capture).objects
    assert len(transport.requests)==2 and len(a.last_diagnostics['responses'])==2
    assert 'error' in a.last_diagnostics['responses'][0]
    b,t=make_a(tmp_path/'b',[bad,bad,offline_wire()])
    with pytest.raises(SchemaError,match='after one repair'):b.describe(capture)
    assert len(t.requests)==2 and b.last_diagnostics['status']=='error'


def test_provider_failure_is_not_missing_and_key_is_not_logged(capture,tmp_path):
    a,transport=make_a(tmp_path,[(401,{'error':{'message':'bad test-secret'}})])
    with pytest.raises(APIError,match='REDACTED'):a.describe(capture)
    assert len(transport.requests)==1
    for path in tmp_path.rglob('*.json'):
        assert 'test-secret' not in path.read_text()


def test_duplicate_identity_uncertainty_does_not_choose_nearest_tie(capture,tmp_path,monkeypatch):
    import perception.student_a as module
    wire=offline_wire();stone=wire['detections'][0]
    wire['detections']=[stone,{**stone,'bbox':[100,100,200,200]}];wire['selected']=[]
    positions=iter([(.4,-.1,.875),(.4,.1,.875),(.4,0,.875),(.4,0,.875)])
    monkeypatch.setattr(module,'localize',lambda *args:(next(positions),'test depth'))
    a,_=make_a(tmp_path,[wire,wire]);first=a.describe(capture);second=a.describe(changed(capture,1))
    assert len({g.instance_id for g in first.objects})==2
    assert all(g.status is GroundStatus.AMBIGUOUS and g.pos_world is None for g in second.objects)


def test_invalid_camera_or_clipped_box_stays_unlocalized(capture):
    box=pixel_box(offline_wire()['detections'][0]['bbox'])
    assert localize(replace(capture,intrinsics=np.zeros((3,3))),box,'stone','gray')[0] is None
    assert localize(capture,(0,0,20,20),'stone','gray')[0] is None


def test_vlm_config_does_not_inherit_b_model_or_schema_mode():
    config=VisionConfig.from_env({'DASHSCOPE_API_KEY':'secret',
        'EE4705_QWEN_BASE_URL':'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
        'EE4705_QWEN_MODEL':'qwen-plus','EE4705_QWEN_OUTPUT_MODE':'json_schema'})
    assert config.llm.model=='qwen3-vl-plus' and config.llm.structured_output_mode=='json_object'
    assert config.llm.enable_thinking is False and 'api_key' not in config.public_settings()
    with pytest.raises(ValueError):VisionConfig.from_env({'EE4705_QWEN_BASE_URL':'http://remote.test/v1'})


def test_live_adapter_cannot_replay_fixture_cache(capture,tmp_path):
    from core.llm_client import CacheMissError, RequestsTransport
    a,transport=make_a(tmp_path,[offline_wire()])
    a.config.llm=replace(a.config.llm,cache_dir=str(tmp_path/'cache'))
    a.client=LLMClient(a.config.llm,transport=transport)
    a.describe(capture)
    config=replace(a.config.llm,cache_only=True)
    b=StudentAPerception(VisionConfig(config,str(tmp_path/'cached_audit')),
                        LLMClient(config,transport=RequestsTransport()))
    with pytest.raises(CacheMissError):b.describe(capture)
    assert b.client.stats.live_requests==0 and b.last_diagnostics['status']=='error'
    c=StudentAPerception(VisionConfig(config,str(tmp_path/'fixture_audit')),
                        LLMClient(config,transport=FakeTransport([])))
    scene=c.describe(capture)
    assert c.client.stats.cache_hits==1 and c.last_diagnostics['response_source']=='fixture'
    assert all(g.source=='qwen_vlm:fixture' for g in scene.objects)
