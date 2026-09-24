"""Final2 stage 1: a near-identical re-render reuses A's earlier answer.

Two captures of an unchanged simulation state can differ by a few pixels by
one level (EGL render noise), so the exact image-sha reuse missed them; C's
APPROACH capture right after the plan-time capture is the main case."""
from dataclasses import replace

import numpy as np

from tests.test_student_a_partial_object import frame, make_a  # noqa: F401  (fixture)


def noisy(obs, pixels, level, seed=0):
    rgb = obs.rgb.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(rgb.shape[0] * rgb.shape[1], pixels, replace=False)
    ys, xs = np.unravel_index(idx, rgb.shape[:2])
    rgb[ys, xs, 0] = np.clip(rgb[ys, xs, 0].astype(int) + level, 0, 255)
    return replace(obs, rgb=rgb, frame_id=obs.frame_id + 1)


def test_render_noise_reuses_the_answer(frame, tmp_path):
    obs, wire = frame
    a = make_a(tmp_path, [wire])  # exactly one model reply available
    first = a.describe(obs)
    second = a.describe(noisy(obs, 6, 1))
    assert a.last_diagnostics['prediction_reused'] and a.last_diagnostics['near_duplicate_of']
    assert second.frame_id == obs.frame_id + 1
    assert [g.instance_id for g in second.objects] == [g.instance_id for g in first.objects]


def test_a_real_change_is_sent_to_the_model(frame, tmp_path):
    obs, wire = frame
    a = make_a(tmp_path, [wire, wire, wire])
    a.describe(obs)
    a.describe(noisy(obs, 5000, 40))
    assert not a.last_diagnostics['prediction_reused']
    a.describe(noisy(obs, 10, 20, seed=1))  # few pixels, but a large change
    assert not a.last_diagnostics['prediction_reused']


def test_only_the_same_simulation_instant_is_reused(frame, tmp_path):
    obs, wire = frame
    a = make_a(tmp_path, [wire, wire])
    a.describe(obs)
    a.describe(replace(noisy(obs, 6, 1), sim_time=obs.sim_time + 0.2))
    assert not a.last_diagnostics['prediction_reused']
