# Owner: Student A
"""Qwen-VL boxes + public RGB-D geometry + conservative episode tracking.

The provider sees RGB only. Depth, camera calibration and tracking remain local.
An API failure raises an error; it is never converted into a missing-object claim.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from core.interfaces import Perception
from core.llm_client import APIError, LLMClient, SchemaError
from core.obs_store import persist_observation
from core.types import GroundedObject, GroundStatus, SceneDescription
from perception.config import VisionConfig
from perception.depth_geometry import localize, pixel_box
from perception.vision_contract import SCHEMA, SYSTEM, VERSION, validate_wire


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                   default=lambda x: x.value if hasattr(x, 'value') else asdict(x)))


class StudentAPerception(Perception):
    IMPLEMENTED = True
    LABEL = 'A: Qwen-VL + RGB-D geometry (live accuracy not yet validated)'

    def __init__(self, config=None, client=None):
        self.config = config or (VisionConfig(client.config) if client else VisionConfig.from_env())
        self.client = client or LLMClient(self.config.llm)
        self.audit_root = Path(self.config.audit_dir) / uuid.uuid4().hex[:12]
        self.last_diagnostics = None
        self.reset()

    def reset(self):
        self._tracks = {}
        self._next_id = 0
        self._predictions = OrderedDict()
        self.last_diagnostics = None

    def _new_id(self):
        instance_id = f'a{self._next_id}'
        self._next_id += 1
        return instance_id

    def _track(self, detections, obs, source):
        """One-to-one spatial association; uncertain duplicate identities stay ambiguous.

        A unique class/color can move across the scene and keep its ID. Multiple
        identical objects require separated 3D evidence (15 cm gate, 3 cm margin).
        These are association limits, not proof of permanent physical identity.
        """
        prepared = []
        for d in detections:
            box = pixel_box(d['bbox'])
            pos, detail = localize(obs, box, d['name'], d['color'])
            if d['confidence'] < .5:
                pos, detail = None, 'model confidence below 0.5'
            prepared.append((d, box, pos, detail))
        assigned = {}
        ambiguous = set()
        for key in sorted({(d['name'], d['color']) for d in detections}):
            indices = [i for i, (d, *_) in enumerate(prepared) if (d['name'], d['color']) == key]
            old = {k: v for k, v in self._tracks.items() if v['key'] == key}
            if len(indices) == len(old) == 1:
                assigned[indices[0]] = next(iter(old))
            elif not old:
                for i in indices:
                    assigned[i] = self._new_id()
            else:
                candidates = {}
                for i in indices:
                    pos = prepared[i][2]
                    distances = sorted((float(np.linalg.norm(np.array(pos) - t['pos'])), k)
                                       for k, t in old.items() if pos is not None and t['pos'] is not None)
                    if distances and distances[0][0] < .15 and (
                            len(distances) == 1 or distances[1][0] - distances[0][0] > .03):
                        candidates[i] = distances[0][1]
                for i in indices:
                    target = candidates.get(i)
                    if target and list(candidates.values()).count(target) == 1:
                        assigned[i] = target
                    else:
                        ambiguous.add(i)
                        assigned[i] = self._new_id()
        result = []
        for i, (d, box, pos, detail) in enumerate(prepared):
            instance_id = assigned[i]
            status = GroundStatus.LOCALIZED if pos is not None else GroundStatus.UNLOCALIZED
            if i in ambiguous:
                status, pos = GroundStatus.AMBIGUOUS, None
                detail = 'cannot associate identical objects confidently across frames'
            else:
                previous = self._tracks.get(instance_id, {})
                self._tracks[instance_id] = {'key': (d['name'], d['color']),
                                            'pos': pos if pos is not None else previous.get('pos')}
            region = d['name'] == 'red_region'
            result.append(GroundedObject(instance_id, d['name'], status,
                bbox_xyxy=box, pos_world=pos, confidence=d['confidence'],
                source=f'qwen_vlm:{source}',
                kind='region' if region else 'object', frame_id=obs.frame_id,
                attributes={'color': d['color'], 'localization': detail},
                region_half_extents_xy=(.08, .08) if region and pos is not None else None))
        return result

    def _observe(self, obs, query, grounding=False):
        buffer = io.BytesIO()
        Image.fromarray(obs.rgb).save(buffer, format='PNG')
        png = buffer.getvalue()
        digest = hashlib.sha256(png).hexdigest()
        key = (digest, query, grounding)
        audit_dir = self.audit_root / uuid.uuid4().hex[:12]
        files = persist_observation(obs, audit_dir)
        audit = {'version': VERSION, 'frame_id': obs.frame_id, 'sim_time': obs.sim_time,
                 'query': query, 'grounding': grounding, 'image_sha256': digest,
                 'input_files': files, 'settings': self.config.public_settings(),
                 'responses': [], 'prediction_reused': False}
        self.last_diagnostics = audit
        try:
            if key in self._predictions:
                wire, source = self._predictions[key]
                audit['prediction_reused'] = True
            else:
                prompt = ('Ground this target phrase: ' if grounding else 'Describe the scene and answer: ')
                prompt += json.dumps(query or 'What supported objects and regions are visible?')
                for attempt in range(2):
                    response = None
                    try:
                        response = self.client.call_vlm(png, prompt, system=SYSTEM, json_schema=SCHEMA)
                        wire = response.parsed
                        validate_wire(wire)
                        audit['responses'].append(asdict(response))
                        source = response.source
                        break
                    except (SchemaError, ValueError) as exc:
                        response = response or getattr(exc, 'response', None)
                        audit['responses'].append({'error': str(exc),
                                                   'response': asdict(response) if response else None})
                        if attempt == 1:
                            raise SchemaError('VLM output invalid after one repair: ' + str(exc)) from exc
                        prompt += '\nYour previous response was invalid: ' + str(exc) + '. Return corrected JSON.'
                self._predictions[key] = (wire, source)
                if len(self._predictions) > 16:
                    self._predictions.popitem(last=False)
            audit['response_source'] = source
            objects = self._track(wire['detections'], obs, source)
            ambiguities = [g.attributes['localization'] for g in objects if g.status is GroundStatus.AMBIGUOUS]
            if grounding and len(wire['selected']) > 1:
                ambiguities.append('Several visible detections match the target phrase')
            scene = SceneDescription([g for g in objects if g.kind == 'object'],
                                     [g for g in objects if g.kind == 'region'], wire['answer'],
                                     ambiguities, obs.frame_id, obs.sim_time)
            audit.update(wire=wire, scene=asdict(scene), status='ok')
            return scene, objects, wire['selected']
        except Exception as exc:
            error = str(exc).replace(self.config.llm.api_key, '[REDACTED]') if self.config.llm.api_key else str(exc)
            audit.update(status='error', error=error)
            if error != str(exc):
                raise APIError(error) from None
            raise
        finally:
            audit.setdefault('response_source', self.client.response_source)
            audit['client_stats'] = self.client.stats.as_dict()
            audit['real_api_requests'] = (self.client.stats.live_requests
                                           if self.client.response_source == 'live' else 0)
            audit['audit_file'] = str(audit_dir / 'audit.json')
            serialized = json.dumps(audit, default=lambda x: x.value if hasattr(x, 'value') else asdict(x))
            if self.config.llm.api_key:
                serialized = serialized.replace(self.config.llm.api_key, '[REDACTED]')
            write_json(audit_dir / 'audit.json', json.loads(serialized))

    def describe(self, obs, query=None):
        return self._observe(obs, query)[0]

    def ground(self, obs, target):
        if target in self._tracks:
            return self.describe(obs).find(target)
        _, objects, selected = self._observe(obs, target, grounding=True)
        if not selected:
            return None
        if len(selected) > 1:
            return replace(objects[selected[0]], status=GroundStatus.AMBIGUOUS, pos_world=None,
                           attributes={**objects[selected[0]].attributes,
                                       'candidates': ','.join(objects[i].instance_id for i in selected)})
        return objects[selected[0]]
