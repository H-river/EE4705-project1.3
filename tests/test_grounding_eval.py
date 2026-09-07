from core.types import GroundedObject, GroundStatus
from eval.grounding import score_grounding


def case(kind='unique'):
    return {'candidate_gt_ids':[] if kind=='missing' else ['stone','stone2'] if kind=='ambiguous' else ['stone'],
            'expected_kind':kind, 'gt_bboxes':{'stone':[64,48,128,96],'stone2':[192,48,256,96]},
            'gt_centers_world':{'stone':[.4,.1,.875]}}


def wire(selected):
    return {'detections':[{'bbox':[100,100,200,200]},{'bbox':[300,100,400,200]}],'selected':selected}


def test_unique_selection_is_scored_by_box_not_perceived_id():
    g=GroundedObject('unrelated-id','stone',GroundStatus.LOCALIZED,pos_world=(.41,.1,.875))
    result=score_grounding(case(),g,wire([0]))
    assert result['target_selection_correct'] and abs(result['position_error_m']-.01)<1e-9
    assert not score_grounding(case(),g,wire([1]))['target_selection_correct']


def test_missing_and_ambiguity_need_matching_evidence():
    assert score_grounding(case('missing'),None,wire([]))['target_selection_correct']
    g=GroundedObject('a1','stone',GroundStatus.AMBIGUOUS)
    assert score_grounding(case('ambiguous'),g,wire([0,1]))['target_selection_correct']
    assert not score_grounding(case('ambiguous'),g,wire([0]))['target_selection_correct']
    assert not score_grounding(case('unique'),None,wire([]))['target_selection_correct']


def test_full_evaluation_keeps_fixture_results_out_of_live_score(standard_world,env,tmp_path,monkeypatch):
    import hashlib,json
    from pathlib import Path
    from core.llm_client import FakeTransport,LLMClient,LLMConfig
    from core.obs_store import persist_observation
    from eval.grounding import evaluate_dataset
    from perception.config import VisionConfig
    import perception.student_a as implementation
    dataset=tmp_path/'dataset';files=persist_observation(env.get_obs(),dataset)
    c={'id':'missing','target':'the invisible bottle','query':'What is visible?',
       'expected_kind':'missing','candidate_gt_ids':[],'gt_bboxes':{},'gt_centers_world':{},
       'files':{k:Path(v).name for k,v in files.items()},
       'sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in files.items()}}
    (dataset/'dataset.json').write_text(json.dumps({'cases':[c]}))
    config=VisionConfig(LLMConfig('fixture','https://fixture.invalid/v1',api_key='not-real',max_retries=0))
    monkeypatch.setattr(VisionConfig,'from_env',classmethod(lambda cls:config))
    cls=implementation.StudentAPerception
    transport=FakeTransport([FakeTransport.completion(json.dumps({'detections':[],'selected':[],'answer':'fixture'})) for _ in range(2)])
    monkeypatch.setattr(implementation,'StudentAPerception',lambda config:cls(config,LLMClient(config.llm,transport=transport)))
    result=evaluate_dataset(dataset/'dataset.json',tmp_path/'result')
    assert result['cases']==1 and result['correct']==0 and result['error_cases']==1
    assert result['position_error_samples']==0 and len(transport.requests)==2
    record=json.loads((tmp_path/'result/missing.json').read_text())
    assert record['response_sources']==['fixture','fixture'] and 'Non-live-origin' in record['error']
    # Integrity failure happens before any further model request.
    Path(files['rgb']).write_bytes(b'changed image')
    import pytest
    with pytest.raises(ValueError,match='hash mismatch'):
        evaluate_dataset(dataset/'dataset.json',tmp_path/'corrupt')
    assert len(transport.requests)==2
