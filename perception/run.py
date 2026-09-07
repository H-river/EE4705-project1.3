"""Test A on a saved RGB-D capture, or on an explicitly labelled offline fixture."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core.llm_client import FakeTransport, LLMClient, LLMConfig
from core.types import Observation
from perception.config import VisionConfig
from perception.student_a import StudentAPerception, write_json


def offline_observation():
    # This helper builds input imagery only. No oracle enters the A implementation.
    from core.env import RobotEnv
    from core.types import SceneConfig, SceneObjectSpec
    from core.world import SimWorld
    world = SimWorld()
    try:
        world.reset(SceneConfig(seed=0, robot_init={'x': -.12, 'y': 0., 'yaw': .15}, objects=[
            SceneObjectSpec('stone', (.4, -.15, .88)), SceneObjectSpec('cube', (.4, .15, .88)),
            SceneObjectSpec('bottle', (.55, -.35, .915))]))
        world.step(500)
        return RobotEnv(world).get_obs()
    finally:
        world.close()


def offline_wire():
    # Hand-labelled from the fixed camera image. This is NOT model inference.
    boxes = [('stone', 'gray', (515, 200, 579, 260)),
             ('cube', 'blue', (218, 163, 285, 238)),
             ('red_region', 'red', (23, 150, 220, 269))]
    return {'detections': [{'name': name, 'color': color,
            'bbox': [x1/640*1000, y1/480*1000, x2/640*1000, y2/480*1000], 'confidence': .9}
            for name, color, (x1, y1, x2, y2) in boxes], 'selected': [0],
            'answer': 'OFFLINE FIXTURE: a gray stone, a blue cube, and a red region are visible.'}


def load_observation(rgb, depth, meta):
    m = json.loads(Path(meta).read_text())
    return Observation(m['frame_id'], np.asarray(Image.open(rgb).convert('RGB')).copy(),
                       np.load(depth, allow_pickle=False), np.asarray(m['intrinsics']),
                       np.asarray(m['t_world_camera']), float(m['sim_time']), m['camera_name'],
                       m.get('capture_id', ''))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline-demo', action='store_true', help='Canned boxes, no API; fixed stone target')
    parser.add_argument('--rgb'); parser.add_argument('--depth'); parser.add_argument('--meta')
    parser.add_argument('--query', default='What supported objects are visible?')
    parser.add_argument('--target', default='gray stone')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.offline_demo and (any((args.rgb, args.depth, args.meta)) or args.target != 'gray stone'):
        parser.error('--offline-demo uses the fixed sample and gray stone target')
    if not args.offline_demo and not all((args.rgb, args.depth, args.meta)):
        parser.error('Provide --rgb, --depth and --meta, or use --offline-demo')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): parser.error('Choose an empty --out directory to preserve previous evidence')
    try:
        if args.offline_demo:
            obs = offline_observation()
            transport = FakeTransport([FakeTransport.completion(json.dumps(offline_wire())) for _ in range(2)])
            config = VisionConfig(LLMConfig('offline-fixture', 'https://fixture.invalid/v1',
                                           api_key='offline-fixture', max_retries=0), str(out/'audit'))
            module = StudentAPerception(config, LLMClient(config.llm, transport=transport))
        else:
            obs = load_observation(args.rgb, args.depth, args.meta)
            config = VisionConfig.from_env()
            config.audit_dir = str(out/'audit')
            module = StudentAPerception(config)
        scene = module.describe(obs, args.query)
        grounded = module.ground(obs, args.target)
        write_json(out/'scene.json', scene)
        write_json(out/'grounded.json', grounded)
        overlay = Image.fromarray(obs.rgb).copy(); draw = ImageDraw.Draw(overlay)
        for g in scene.objects + scene.regions:
            draw.rectangle(g.bbox_xyxy, outline='yellow', width=2)
            draw.text((g.bbox_xyxy[0], max(0, g.bbox_xyxy[1]-12)), f'{g.instance_id} {g.name} {g.status.value}', fill='white')
        overlay.save(out/'overlay.png')
        summary = {'source': module.last_diagnostics['response_source'], 'real_api_requests': (module.client.stats.live_requests if module.client.response_source == 'live' else 0),
                   'transport_requests': module.client.stats.live_requests,
                   'target': args.target, 'ground_status': grounded.status.value if grounded else 'NOT_FOUND',
                   'objects': len(scene.objects), 'regions': len(scene.regions),
                   'model_accuracy_measured': False, 'output': str(out)}
        write_json(out/'summary.json', summary)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        write_json(out/'error.json', {'error': str(exc)})
        print(f'Perception error: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
