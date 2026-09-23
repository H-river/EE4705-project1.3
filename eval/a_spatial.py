"""Spatial Reasoning Accuracy.

Tests two relation types, each with a keyword-checkable expected answer:
  - containment: "Is the <object> inside the red area?" -- yes/no, computed
    from the object's (x,y) vs. the region's known 0.08x0.08 half-extents.
  - relative distance: "Which object is closer to the robot: the <A> or the
    <B>?" -- ground truth is whichever has the smaller world-x coordinate
    (x increases with distance from the robot base in this project's scenes;
    confirm this against a captured image before trusting results).

Both ride on describe()'s free-text `answer` field -- there is no structured
spatial-relation output in the current wire format, so scoring is a keyword
check, not exact matching. This makes real live model calls; it is not a
reuse of an existing eval.grounding run.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from core.types import SceneConfig, SceneObjectSpec
from perception.config import VisionConfig
from perception.student_a import StudentAPerception


def to_json_safe(x):
    return json.loads(json.dumps(x, default=lambda o: o.value if hasattr(o, 'value') else asdict(o)))


def empty_dir(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Choose an empty output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    return out


def capture_spatial_dataset(out, n_containment=6, n_distance=6):
    from core.env import RobotEnv
    from core.oracle import EvalOracle
    from core.obs_store import persist_observation
    from core.world import SimWorld
    out = empty_dir(out)
    world = SimWorld(); oracle = EvalOracle(world)
    cases = []
    number = 0
    try:
        # Containment: stone at varying offsets from the region center -- half inside, half outside.
        for i in range(n_containment):
            inside = i % 2 == 0
            dx = (0.02 if inside else 0.20) * (1 if i % 4 < 2 else -1)
            dy = (0.02 if inside else 0.20) * (1 if i % 2 == 0 else -1)
            stone = SceneObjectSpec('stone', (.4 + dx, .30 + dy, .88))  # region center is (.4, .30, ~.85)
            cube = SceneObjectSpec('cube', (.43, .12, .88))
            world.reset(SceneConfig(seed=300 + number, robot_init={'x': -.12, 'y': 0., 'yaw': .1},
                                     objects=[stone, cube]))
            world.step(500)
            obs = RobotEnv(world).get_obs('head')
            folder = out / f's_{number+1:02d}'
            files = persist_observation(obs, folder)
            region_pos = oracle.object_pos('red_region')
            stone_pos = oracle.object_pos('stone')
            actually_inside = (abs(stone_pos[0] - region_pos[0]) <= 0.08
                               and abs(stone_pos[1] - region_pos[1]) <= 0.08)
            case = {'id': folder.name, 'relation': 'containment',
                    'query': 'Is the gray stone inside the red area? Answer yes or no.',
                    'expected_answer': 'yes' if actually_inside else 'no',
                    'files': {k: str(Path(v).relative_to(out)) for k, v in files.items()}}
            case['sha256'] = {k: hashlib.sha256((out / v).read_bytes()).hexdigest()
                               for k, v in case['files'].items()}
            cases.append(case); number += 1

        # Relative distance: cube and bottle at swapped depths.
        for i in range(n_distance):
            cube_closer = i % 2 == 0
            cube_x, bottle_x = (.35, .58) if cube_closer else (.58, .35)
            cube = SceneObjectSpec('cube', (cube_x, .12, .88))
            bottle = SceneObjectSpec('bottle', (bottle_x, -.20, .915))
            world.reset(SceneConfig(seed=400 + number, robot_init={'x': -.12, 'y': 0., 'yaw': .1},
                                     objects=[cube, bottle]))
            world.step(500)
            obs = RobotEnv(world).get_obs('head')
            folder = out / f's_{number+1:02d}'
            files = persist_observation(obs, folder)
            case = {'id': folder.name, 'relation': 'relative_distance',
                    'query': 'Which object is closer to the robot: the blue cube or the green bottle? '
                             'Answer with just the object name.',
                    'expected_answer': 'cube' if cube_closer else 'bottle',
                    'files': {k: str(Path(v).relative_to(out)) for k, v in files.items()}}
            case['sha256'] = {k: hashlib.sha256((out / v).read_bytes()).hexdigest()
                               for k, v in case['files'].items()}
            cases.append(case); number += 1
    finally:
        world.close()

    manifest = {'version': 1, 'role': 'A spatial-reasoning development dataset; not a model result',
                'n_cases': len(cases), 'cases': cases}
    (out / 'dataset.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'dataset': str(out / 'dataset.json'), 'cases': len(cases)}, indent=2))
    return manifest


def run_spatial(dataset_path, out, target=0.75):
    from core.types import Observation
    import numpy as np
    dataset_path = Path(dataset_path)
    out = empty_dir(out)
    dataset = json.loads(dataset_path.read_text())
    ds_dir = dataset_path.parent
    a = StudentAPerception(VisionConfig.from_env())
    results = []
    for case in dataset['cases']:
        a.reset()
        rgb = np.array(__import__('PIL.Image', fromlist=['Image']).open(ds_dir / case['files']['rgb']))
        depth = np.load(ds_dir / case['files']['depth'])
        meta = json.loads((ds_dir / case['files']['meta']).read_text())
        obs = Observation(rgb=rgb, depth=depth, intrinsics=np.array(meta['intrinsics']),
                           t_world_camera=np.array(meta['t_world_camera']), frame_id=0, sim_time=0.,
                           camera_name='head')
        try:
            scene = a.describe(obs, query=case['query'])
            answer = scene.caption.lower()
            correct = case['expected_answer'] in answer
            record = {'id': case['id'], 'relation': case['relation'], 'query': case['query'],
                      'expected': case['expected_answer'], 'answer': scene.caption, 'correct': correct}
        except Exception as exc:
            record = {'id': case['id'], 'relation': case['relation'], 'error': str(exc), 'correct': False}
        results.append(record)
        (out / f"{case['id']}.json").write_text(json.dumps(record, indent=2))

    n = len(results)
    correct = sum(r['correct'] for r in results)
    by_relation = {}
    for rel in ('containment', 'relative_distance'):
        rel_results = [r for r in results if r['relation'] == rel]
        by_relation[rel] = {'correct': sum(r['correct'] for r in rel_results), 'total': len(rel_results)}
    summary = {'cases': n, 'correct': correct, 'spatial_reasoning_accuracy': correct / n if n else 0.0,
               'target': target, 'target_met': (correct / n if n else 0.0) >= target,
               'by_relation': by_relation}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capture = sub.add_parser('capture'); capture.add_argument('--out', required=True)
    capture.add_argument('--n-containment', type=int, default=6)
    capture.add_argument('--n-distance', type=int, default=6)
    run = sub.add_parser('run'); run.add_argument('--dataset', required=True)
    run.add_argument('--out', required=True); run.add_argument('--target', type=float, default=0.75)
    args = parser.parse_args(argv)
    if args.command == 'capture':
        capture_spatial_dataset(args.out, args.n_containment, args.n_distance)
    else:
        run_spatial(args.dataset, args.out, args.target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
