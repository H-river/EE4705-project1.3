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
from core.llm_client import APIError, ContentFiltered, LLMClient, SchemaError
from core.obs_store import persist_observation
from core.types import CONTENT_FILTERED_NOTE, GroundedObject, GroundStatus, SceneDescription
from perception.config import VisionConfig
from core.scene_geometry import table_top_z
from core.vocab import Vocab
from perception.depth_geometry import backproject, color_mask, localize, pixel_box
from perception.vision_contract import SCHEMA, SYSTEM, VERSION, validate_wire


# EXPERIMENTAL (night run round 2, committed separately so it can be dropped):
# a region whose box is clipped by the image edge used to be UNLOCALIZED, which
# made a correct PLACE unverifiable (c_2_01/02/08, smoke_2, a_demo). If the
# depth patch is mostly valid we localize the VISIBLE part instead. The centre
# is the visible part's centre, so it is biased toward the image when the
# region is clipped; attributes['pos_basis'] = 'depth_patch_partial' marks it.
PARTIAL_REGION_VALID_RATIO = .5
PARTIAL_REGION_MIN_PIXELS = 24


def partial_region_center(obs, bbox):
    """Centre of the visible part of a clipped red region, or (None, reason)."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None, 'empty region box'
    depth = obs.depth[y1:y2, x1:x2]
    finite = np.isfinite(depth) & (depth > 0)
    ratio = float(np.count_nonzero(finite)) / float(depth.size)
    if ratio < PARTIAL_REGION_VALID_RATIO:
        return None, f'clipped region depth patch only {ratio:.0%} valid'
    rgb = obs.rgb[y1:y2, x1:x2].astype(float)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    valid = finite & (r > 160) & (g < 110) & (b < 110) & (r - g > 100)
    if np.count_nonzero(valid) < PARTIAL_REGION_MIN_PIXELS:
        return None, 'too few red pixels in the clipped region box'
    k, t = obs.intrinsics, obs.t_world_camera
    if (not np.all(np.isfinite(k)) or not np.all(np.isfinite(t))
            or k[0, 0] <= 0 or k[1, 1] <= 0):
        return None, 'invalid camera geometry'
    yy, xx = np.mgrid[y1:y2, x1:x2]
    z = depth[valid]
    u, v = xx[valid], yy[valid]
    pc = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))
    cloud = pc @ t[:3, :3].T + t[:3, 3]
    if not np.all(np.isfinite(cloud)):
        return None, 'invalid camera geometry'
    z0 = np.percentile(cloud[:, 2], 30)
    cloud = cloud[np.abs(cloud[:, 2] - z0) < .006]
    if len(cloud) < PARTIAL_REGION_MIN_PIXELS:
        return None, 'clipped region support is not flat enough'
    center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2
    center[2] = np.median(cloud[:, 2])
    return tuple(float(c) for c in center), 'partial RGB-D fit of the visible region part'


# Round 6 (1a): every supported object/region rests on the table top or is held
# above it, so a detection whose estimated 3D centre lies outside this band
# (relative to the table top read from the scene MJCF) is a hallucination on
# the floor, a wall or the robot's shadow. It is dropped before tracking so it
# can neither claim an identity nor make a real instance AMBIGUOUS.
TABLE_BAND_M = (-.05, .30)
MIN_CENTER_PIXELS = 24


def estimate_center_z(obs, box, name, color, fit=None):
    """(z, basis) of a detection's 3D centre: the class-geometry fit when it
    succeeds, else the median height of the box's valid-depth pixels (colour
    pixels first). (None, reason) when the depth gives no evidence.
    ``fit`` is a precomputed localize() result for the same box."""
    pos, _ = fit if fit is not None else localize(obs, box, name, color)
    if pos is not None:
        return pos[2], 'class_fit'
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None, 'empty box'
    mask = color_mask(obs.rgb[y1:y2, x1:x2], color)
    for m, basis in ((mask, 'color_pixels'), (None, 'box_pixels')):
        if m is not None and np.count_nonzero(m) < MIN_CENTER_PIXELS:
            continue
        cloud = backproject(obs, box, m)
        if cloud is not None and len(cloud) >= MIN_CENTER_PIXELS:
            return float(np.median(cloud[:, 2])), basis
    return None, 'too few valid depth pixels'


def drop_off_table(wire, obs, fits=None):
    """Wire without off-table detections (``selected`` re-mapped), the audit
    records of what was dropped, and the kept detections' localize() fits."""
    top = table_top_z()
    if fits is None:
        fits = [localize(obs, pixel_box(d['bbox']), d['name'], d['color']) for d in wire['detections']]
    keep, dropped = [], []
    for i, d in enumerate(wire['detections']):
        z, basis = estimate_center_z(obs, pixel_box(d['bbox']), d['name'], d['color'], fits[i])
        if z is not None and not top + TABLE_BAND_M[0] <= z <= top + TABLE_BAND_M[1]:
            dropped.append({'index': i, 'detection': d, 'reason': 'off_table',
                            'center_z': z, 'center_basis': basis, 'table_top_z': top})
        else:
            keep.append(i)
    if not dropped:
        return wire, [], fits
    remap = {old: new for new, old in enumerate(keep)}
    return ({**wire, 'detections': [wire['detections'][i] for i in keep],
             'selected': [remap[i] for i in wire['selected'] if i in remap]}, dropped,
            [fits[i] for i in keep])


# Round 6 (step 2): the same edge fix for OBJECTS. A box touching the image
# boundary is still localized when at least half of its depth patch is valid:
# the median 3D point of the visible colour pixels is pushed along the view ray
# by the class's horizontal half extent (assets/objects.yaml), i.e. from the
# visible surface to the centre. Never for the held object, and only when the
# result rests on the table (a lifted object is being carried).
PARTIAL_OBJECT_VALID_RATIO = .5
PARTIAL_OBJECT_MIN_PIXELS = 24
PARTIAL_OBJECT_REST_MARGIN_M = .03


def _object_size(name, color):
    for e in Vocab().entries:
        if e.cls == name and e.kind == 'object' and e.attributes.get('color') in (color, None):
            return e.size_xyz
    return None


def partial_object_center(obs, bbox, name, color):
    """Centre of an object whose box is clipped by the image edge, or (None, reason)."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None, 'empty object box'
    size = _object_size(name, color)
    if size is None:
        return None, 'no public size for this class'
    depth = obs.depth[y1:y2, x1:x2]
    ratio = float(np.count_nonzero(np.isfinite(depth) & (depth > 0))) / float(depth.size)
    if ratio < PARTIAL_OBJECT_VALID_RATIO:
        return None, f'clipped object depth patch only {ratio:.0%} valid'
    mask = color_mask(obs.rgb[y1:y2, x1:x2], color)
    if mask is None or np.count_nonzero(mask) < PARTIAL_OBJECT_MIN_PIXELS:
        return None, 'too few object-colour pixels in the clipped box'
    cloud = backproject(obs, bbox, mask)
    if cloud is None or len(cloud) < PARTIAL_OBJECT_MIN_PIXELS:
        return None, 'too few valid depth pixels in the clipped box'
    visible = np.median(cloud, axis=0)
    ray = visible - obs.t_world_camera[:3, 3]
    center = visible + ray / np.linalg.norm(ray) * max(size[0], size[1]) / 2
    rest_z = table_top_z() + size[2] / 2
    if center[2] > rest_z + PARTIAL_OBJECT_REST_MARGIN_M:
        return None, 'clipped object is not resting on the table (possibly held)'
    if center[2] < rest_z - PARTIAL_OBJECT_REST_MARGIN_M:
        return None, 'clipped object centre is below the table top'
    return tuple(float(c) for c in center), 'partial RGB-D: visible part pushed by the class half extent'


def _projects_into(obs, point, box, margin):
    """True when a world point projects inside a pixel box grown by ``margin``."""
    k, t = obs.intrinsics, obs.t_world_camera
    cam = t[:3, :3].T @ (np.asarray(point, dtype=float) - t[:3, 3])
    if cam[2] <= .05:
        return False
    u, v = k[0, 0] * cam[0] / cam[2] + k[0, 2], k[1, 1] * cam[1] / cam[2] + k[1, 2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return x1 - mx <= u <= x2 + mx and y1 - my <= v <= y2 + my


QUERY_CLASSES = ('stone', 'cube', 'bottle', 'red_region')
QUERY_COLOURS = ('gray', 'dark_red', 'blue', 'green', 'red')


def colour_class_query(target):
    """('stone', 'dark_red') for 'dark_red stone' / 'dark red stone'; None for
    anything else (a bare class, an instance id, a free-form phrase)."""
    words = (target or '').strip().lower().split()
    if len(words) < 2 or words[-1] not in QUERY_CLASSES:
        return None
    colour = '_'.join(words[:-1])
    return (words[-1], colour) if colour in QUERY_COLOURS else None


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
        self._held_id = None  # instance believed held; set from the tracking hint
        self._hint = None
        self.last_diagnostics = None

    def _new_id(self):
        instance_id = f'a{self._next_id}'
        self._next_id += 1
        return instance_id

    def _track(self, detections, obs, source, fits=None):
        """One-to-one spatial association; uncertain duplicate identities stay ambiguous.

        A unique class/color can move across the scene and keep its ID. Multiple
        identical objects require separated 3D evidence (15 cm gate, 3 cm margin).
        These are association limits, not proof of permanent physical identity.

        """
        prepared = []
        for i, d in enumerate(detections):
            box = pixel_box(d['bbox'])
            pos, detail = fits[i] if fits is not None else localize(obs, box, d['name'], d['color'])
            partial = False
            if pos is None and detail == 'box touches image boundary':
                if d['name'] == 'red_region':
                    pos, detail = partial_region_center(obs, box)
                else:
                    pos, detail = partial_object_center(obs, box, d['name'], d['color'])
                partial = pos is not None
            if d['confidence'] < .5:
                pos, detail, partial = None, 'model confidence below 0.5', False
            prepared.append((d, box, pos, detail, partial))
        hint = getattr(self, '_hint', None)
        assigned, held_index = self._hint_assign(prepared, obs) if hint is not None else ({}, None)
        reserved = set(assigned.values())
        if hint is not None and hint.held_object_id:
            reserved.add(hint.held_object_id)  # the held instance never takes part in normal matching
        ambiguous = set()
        for key in sorted({(d['name'], d['color']) for d in detections}):
            indices = [i for i, (d, *_) in enumerate(prepared)
                       if (d['name'], d['color']) == key and i not in assigned]
            old = {k: v for k, v in self._tracks.items() if v['key'] == key and k not in reserved}
            if not indices:
                continue
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
                    elif len(indices) == 1:
                        # Only ONE detection of this class/colour in this frame:
                        # there is nothing in this frame to confuse it with, so
                        # it gets a fresh identity rather than AMBIGUOUS. Stale
                        # tracks (e.g. a phantom duplicate hallucinated in an
                        # earlier frame) must not poison every later frame.
                        assigned[i] = self._new_id()
                    else:
                        ambiguous.add(i)
                        assigned[i] = self._new_id()
        result = []
        for i, (d, box, pos, detail, partial) in enumerate(prepared):
            instance_id = assigned[i]
            if partial and instance_id == getattr(self, "_held_id", None):
                pos, detail, partial = None, 'held object: no partial localization', False
            if i == held_index:
                pos, detail, partial = tuple(hint.held_pos_world), 'held object: position from the tracking hint', False
            status = GroundStatus.LOCALIZED if pos is not None else GroundStatus.UNLOCALIZED
            memory_attrs = {} #Enable the program to "Remember"
            if i in ambiguous:
                status, pos = GroundStatus.AMBIGUOUS, None
                detail = 'cannot associate identical objects confidently across frames'
            else:
                previous = self._tracks.get(instance_id, {})
                self._tracks[instance_id] = {'key': (d['name'], d['color']),
                                             'pos': pos if pos is not None else previous.get('pos'),
                                             'last_localized_frame': obs.frame_id if pos is not None else previous.get('last_localized_frame'),
                                             'last_localized_sim_time': obs.sim_time if pos is not None else previous.get('last_localized_sim_time')
                }
                #This cause the program remember an object previous position and last seen time
                if pos is None and previous.get('last_localized_frame') is not None:
                    memory_attrs = {
                        'memory_last_localized_pos': previous['pos'],
                        'memory_last_localized_frame': previous['last_localized_frame'],
                        'memory_note': 'not confrimed this frame; last confrimed position shown for reference only'
                    }
            region = d['name'] == 'red_region'
            result.append(GroundedObject(instance_id, d['name'], status,
                bbox_xyxy=box, pos_world=pos, confidence=d['confidence'],
                source=f'qwen_vlm:{source}',
                kind='region' if region else 'object', frame_id=obs.frame_id,
                attributes={'color': d['color'], 'localization': detail,
                            **({'pos_basis': 'depth_patch_partial'} if partial and status is GroundStatus.LOCALIZED else {}),
                            **({'pos_basis': 'held_hint'} if i == held_index else {}),
                            **memory_attrs},
                region_half_extents_xy=(.08, .08) if region and pos is not None else None))
        return result

    # Round 6 (step 3): identities across a grasp and a release.
    HINT_RADIUS_M = .15
    HINT_BOX_MARGIN = .2

    def _hint_candidates(self, prepared, track_id, expected, obs, xy_only, taken):
        """Indices of detections of the track's class/colour that are near the
        expected position: within HINT_RADIUS_M (in xy for a release, whose
        expectation is the gripper), or, without a 3D position, whose box
        (+20%) contains the expected point's projection."""
        track = self._tracks[track_id]
        point = np.array(expected, dtype=float)
        if xy_only and track.get('pos') is not None:
            point[2] = track['pos'][2]  # the object rests where it was, not at gripper height
        found = []
        for i, (d, box, pos, *_ ) in enumerate(prepared):
            if i in taken or (d['name'], d['color']) != track['key']:
                continue
            if pos is not None:
                delta = np.array(pos) - point
                dist = float(np.linalg.norm(delta[:2] if xy_only else delta))
                if dist < self.HINT_RADIUS_M:
                    found.append((dist, i))
            elif _projects_into(obs, point, box, self.HINT_BOX_MARGIN):
                found.append((self.HINT_RADIUS_M, i))
        return sorted(found)

    def _hint_assign(self, prepared, obs):
        """Detections fixed by the tracking hint before the normal rules run.
        Returns ({index: instance_id}, index of the held detection or None)."""
        hint = getattr(self, '_hint', None)
        assigned, held_index = {}, None
        if hint is None:
            return assigned, held_index
        if hint.held_object_id in self._tracks and hint.held_pos_world is not None:
            found = self._hint_candidates(prepared, hint.held_object_id, hint.held_pos_world, obs, False, assigned)
            if found and (len(found) == 1 or found[1][0] - found[0][0] > .03):
                held_index = found[0][1]
                assigned[held_index] = hint.held_object_id
        released = hint.released
        if (released is not None and released.instance_id in self._tracks
                and released.instance_id != hint.held_object_id):
            found = self._hint_candidates(prepared, released.instance_id, released.expected_pos_world,
                                          obs, True, assigned)
            if found and (len(found) == 1 or found[1][0] - found[0][0] > .03):
                assigned[found[0][1]] = released.instance_id
        return assigned, held_index

    def recall(self, name, color=None):
        """Best-effort memory lookup — NOT part of the Perception contract.

        Returns the last confirmed position/time for a tracked class, or None
        if nothing of that class has ever been localized this episode. This is
        never a substitute for a live ground()/describe() call: a caller must
        re-confirm with a live observation before acting (grasping, placing)
        on anything returned here.
        """
        candidates = [(iid, t) for iid, t in self._tracks.items()
                      if t['key'][0] == name and (color is None or t['key'][1] == color)
                      and t.get('last_localized_frame') is not None]
        if not candidates:
            return None
        iid, t = max(candidates, key=lambda kv: kv[1]['last_localized_sim_time'])
        return {'instance_id': iid, 'pos_world': t['pos'],
                'last_localized_frame': t['last_localized_frame'],
                'last_localized_sim_time': t['last_localized_sim_time'],
                'confidence_note': 'memory only — not a live observation'}

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
                        if isinstance(response, ContentFiltered):
                            # The provider refused this image. Report an empty
                            # frame (not cached: the next frame may pass) so the
                            # caller re-observes instead of failing the episode.
                            audit['responses'].append(asdict(response))
                            audit.update(response_source=response.source, status='content_filtered',
                                         error=response.detail)
                            scene = SceneDescription([], [], '', [CONTENT_FILTERED_NOTE],
                                                     obs.frame_id, obs.sim_time)
                            audit['scene'] = asdict(scene)
                            return scene, [], []
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
            audit['wire'] = wire
            wire, audit['dropped_detections'], fits = drop_off_table(wire, obs)
            objects = self._track(wire['detections'], obs, source, fits)
            ambiguities = [g.attributes['localization'] for g in objects if g.status is GroundStatus.AMBIGUOUS]
            if grounding and len(wire['selected']) > 1:
                ambiguities.append('Several visible detections match the target phrase')
            scene = SceneDescription([g for g in objects if g.kind == 'object'],
                                     [g for g in objects if g.kind == 'region'], wire['answer'],
                                     ambiguities, obs.frame_id, obs.sim_time)
            audit.update(scene=asdict(scene), status='ok')
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

    def describe(self, obs, query=None, *, hint=None):
        # The latest hint stays in force for later describe()/ground() calls
        # made by the executor inside an action (they carry no hint).
        if hint is not None:
            self._hint = hint
            self._held_id = hint.held_object_id
        return self._observe(obs, query)[0]

    def ground(self, obs, target):
        if target in self._tracks:
            return self.describe(obs).find(target)
        _, objects, selected = self._observe(obs, target, grounding=True)
        wanted = colour_class_query(target)
        if wanted is not None:
            # Round 7: a '<colour> <class>' query (SEARCH with the goal colour)
            # is matched on BOTH fields of A's own detections, not left to the
            # model's selection alone: the model often selects every stone.
            matches = [i for i, g in enumerate(objects)
                       if (g.name, g.attributes.get('color')) == wanted]
            selected = [i for i in selected if i in matches] or matches
        if not selected:
            return None
        if len(selected) > 1:
            return replace(objects[selected[0]], status=GroundStatus.AMBIGUOUS, pos_world=None,
                           attributes={**objects[selected[0]].attributes,
                                       'candidates': ','.join(objects[i].instance_id for i in selected)})
        return objects[selected[0]]
