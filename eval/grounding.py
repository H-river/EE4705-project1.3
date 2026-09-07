"""Capture a labelled A dataset, then evaluate the real adapter on those images.

Oracle labels stay in this evaluation module. A receives RGB-D and query only.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from core.obs_store import persist_observation
from core.oracle import EvalOracle, _bbox_iou
from core.types import GroundStatus, SceneConfig, SceneObjectSpec
from eval.logger import to_json_safe
from perception.depth_geometry import pixel_box
from perception.run import load_observation


def save(path, value):
    Path(path).write_text(json.dumps(to_json_safe(value), indent=2, allow_nan=False))


def empty_dir(path):
    path=Path(path).resolve();path.mkdir(parents=True,exist_ok=True)
    if any(path.iterdir()): raise ValueError(f'Choose an empty output directory: {path}')
    return path


def capture_dataset(out):
    from core.env import RobotEnv
    from core.world import SimWorld
    out=empty_dir(out)
    world=SimWorld();oracle=EvalOracle(world);cases=[]
    try:
        # Four views x five object/target setups. Labels are derived before any model run.
        for view,yaw in enumerate((-.15,.05,.25,.50)):
            for setup in range(5):
                number=view*5+setup
                dx=.01*(view-1);dy=.015*(setup-2)
                objects=[SceneObjectSpec('stone',(.4+dx,-.10+dy,.88)),
                         SceneObjectSpec('cube',(.43,.12+dy,.88)),
                         SceneObjectSpec('bottle',(.54,-.28,.915))]
                if setup==2: objects.append(SceneObjectSpec('stone2',(.53,.02,.88)))
                if setup==3: objects=[o for o in objects if o.name!='bottle']
                target,possible=(('the gray stone',['stone']) if setup==0 else
                                 ('the blue cube',['cube']) if setup==1 else
                                 ('the stone',['stone','stone2']) if setup==2 else
                                 ('the green bottle',['bottle']))
                world.reset(SceneConfig(seed=100+number,robot_init={'x':-.12-.01*view,'y':0.,'yaw':yaw},objects=objects))
                world.step(500)
                obs=RobotEnv(world).get_obs('head')
                folder=out/f'a_{number+1:02d}'
                files=persist_observation(obs,folder)
                boxes=oracle.gt_bboxes('head')
                candidates=[name for name in possible if name in boxes]
                kind='missing' if not candidates else 'unique' if len(candidates)==1 else 'ambiguous'
                case={'id':folder.name,'target':target,'query':'What supported objects and colors are visible?',
                      'expected_kind':kind,'candidate_gt_ids':candidates,
                      'gt_bboxes':boxes,'gt_centers_world':{name:oracle.object_pos(name).tolist() for name in boxes if name!='red_region'},
                      'scene_config':to_json_safe(asdict(SceneConfig(seed=100+number,
                        robot_init={'x':-.12-.01*view,'y':0.,'yaw':yaw},objects=objects))),
                      'files':{k:str(Path(v).relative_to(out)) for k,v in files.items()}}
                case['sha256']={k:hashlib.sha256((out/v).read_bytes()).hexdigest() for k,v in case['files'].items()}
                cases.append(case)
    finally: world.close()
    manifest={'version':1,'role':'A development dataset; not a model result',
              'label_source':'evaluation oracle at the same camera capture; >=30 visible pixels',
              'n_cases':len(cases),'cases':cases}
    save(out/'dataset.json',manifest)
    counts={kind:sum(c['expected_kind']==kind for c in cases) for kind in ('unique','missing','ambiguous')}
    print(json.dumps({'dataset':str(out/'dataset.json'),'cases':len(cases),'labels':counts,'model_calls':0},indent=2))
    return manifest


def score_grounding(case, grounded, wire):
    expected=set(case['candidate_gt_ids'])
    matched=[]
    for index in wire.get('selected',[]):
        detection=wire['detections'][index]
        box=pixel_box(detection['bbox'])
        candidates=sorted(((_bbox_iou(box,b),name) for name,b in case['gt_bboxes'].items()),reverse=True)
        matched.append(candidates[0][1] if candidates and candidates[0][0]>=.3 else None)
    status=('missing' if grounded is None else 'ambiguous' if grounded.status is GroundStatus.AMBIGUOUS else 'unique')
    correct=(status==case['expected_kind'] and set(matched)==expected and len(matched)==len(expected))
    error=None
    if correct and status=='unique' and grounded.pos_world is not None:
        target=next(iter(expected))
        error=float(np.linalg.norm(np.asarray(grounded.pos_world)-case['gt_centers_world'][target]))
    return {'target_selection_correct':correct,'predicted_kind':status,'selected_gt_matches':matched,
            'position_error_m':error,'metric_localized':grounded is not None and grounded.status is GroundStatus.LOCALIZED}


def evaluate_dataset(dataset,out):
    from perception.config import VisionConfig
    from perception.student_a import StudentAPerception
    dataset=Path(dataset).resolve();raw=dataset.read_bytes();manifest=json.loads(raw)
    config=VisionConfig.from_env()
    # Verify every capture before starting paid requests.
    for case in manifest['cases']:
        for key,relative in case['files'].items():
            if hashlib.sha256((dataset.parent/relative).read_bytes()).hexdigest()!=case['sha256'][key]:
                raise ValueError(f"Capture hash mismatch: {case['id']} {key}")
    out=empty_dir(out);config.audit_dir=str(out/'audit')
    module=StudentAPerception(config);records=[]
    save(out/'dataset_snapshot.json',manifest)
    for case in manifest['cases']:
        record={'id':case['id'],'target':case['target'],'expected_kind':case['expected_kind'],
                'target_selection_correct':False,'error':None,'position_error_m':None}
        start=time.monotonic();module.reset()
        try:
            obs=load_observation(*(dataset.parent/case['files'][k] for k in ('rgb','depth','meta')))
            scene=module.describe(obs,case['query'])
            scene_source=module.last_diagnostics['response_source']
            grounded=module.ground(obs,case['target'])
            record.update(score_grounding(case,grounded,module.last_diagnostics['wire']))
            record.update(scene=to_json_safe(scene),grounded=to_json_safe(grounded),
                          response_sources=[scene_source,module.last_diagnostics['response_source']],
                          audit_file=module.last_diagnostics['audit_file'])
            if any(s!='live' for s in record['response_sources']):
                record.update(target_selection_correct=False,position_error_m=None,error='Non-live-origin response; excluded from live success numerator')
        except Exception as exc:
            record['error']=str(exc)
        record['wall_s']=time.monotonic()-start
        records.append(record);save(out/(case['id']+'.json'),record)
        print(f"{case['id']}: {'PASS' if record['target_selection_correct'] else 'FAIL'} ({case['expected_kind']})",flush=True)
    by_kind={kind:{'correct':sum(r['target_selection_correct'] for r in records if r['expected_kind']==kind),
                   'total':sum(r['expected_kind']==kind for r in records)} for kind in ('unique','missing','ambiguous')}
    errors=[r['position_error_m'] for r in records if r['position_error_m'] is not None]
    summary={'cases':len(records),'correct':sum(r['target_selection_correct'] for r in records),
             'target_selection_accuracy':sum(r['target_selection_correct'] for r in records)/len(records),
             'by_kind':by_kind,'position_error_samples':len(errors),
             'mean_position_error_m':float(np.mean(errors)) if errors else None,
             'error_cases':sum(bool(r['error']) for r in records),
             'client_stats':module.client.stats.as_dict(),'dataset_sha256':hashlib.sha256(raw).hexdigest(),
             'vqa_accuracy_measured':False,'note':'All cases remain in the denominator. Scene answers are recorded for manual review; this score measures target selection only. Each trial resets tracking.'}
    save(out/'summary.json',summary);print(json.dumps(summary,indent=2))
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    capture=sub.add_parser('capture');capture.add_argument('--out',required=True)
    run=sub.add_parser('run');run.add_argument('--dataset',required=True);run.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='capture':capture_dataset(args.out)
        else:
            result=evaluate_dataset(args.dataset,args.out)
            return 0 if result['correct']==result['cases'] else 1
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f'A evaluation error: {exc}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
