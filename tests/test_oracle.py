# Owner: backbone (ALL)
"""EvalOracle tests: geometric association (never string matching),
visibility, region predicate, validation-time misuse guard."""

from __future__ import annotations

import numpy as np
import pytest

from core.env import RobotEnv
from core.mocks import GTPerception
from core.types import GroundedObject, GroundStatus, SceneDescription
from tests.conftest import standard_scene


def test_visibility_respects_view(world, oracle):
    world.reset(standard_scene())
    world.step(200)
    assert set(oracle.visible_objects()) == {"stone", "cube", "bottle"}
    # rotate away: nothing visible
    world.teleport_base(0.0, 0.0, 3.0)
    world.data.ctrl[2] = 3.0
    from core.rendering import mujoco

    mujoco.mj_forward(world.model, world.data)
    assert oracle.visible_objects() == []


def test_association_is_geometric_one_to_one(world, oracle):
    world.reset(standard_scene())
    world.step(200)
    env = RobotEnv(world)
    obs = env.get_obs()
    scene = GTPerception(oracle).describe(obs)
    # Perceived ids are p*, never equal to gt ids — the association must
    # still match them all geometrically.
    result = oracle.associate(scene, camera="onboard")
    assert set(result.matches.keys()) == {g.instance_id for g in scene.objects}
    assert set(result.matches.values()) == {"stone", "cube", "bottle"}
    assert all(iou >= 0.3 for iou in result.ious.values())
    # one-to-one: no gt object matched twice
    assert len(set(result.matches.values())) == len(result.matches)
    assert result.unmatched_gt == [] and result.unmatched_perceived == []


def test_association_never_uses_id_strings(world, oracle):
    """A perceived object whose instance_id EQUALS a gt id but whose bbox is
    elsewhere must NOT be associated with that gt object."""
    world.reset(standard_scene())
    world.step(200)
    fake = SceneDescription(
        objects=[GroundedObject(instance_id="stone", name="stone", status=GroundStatus.UNLOCALIZED,
                                bbox_xyxy=(0.0, 0.0, 10.0, 10.0), confidence=0.9)],
        sim_time=world.sim_time,
    )
    result = oracle.associate(fake, camera="onboard")
    assert result.matches == {}
    assert "stone" in result.unmatched_perceived


def test_association_flags_ambiguous_boxes(world, oracle):
    world.reset(standard_scene())
    world.step(200)
    gt = oracle.gt_bboxes("onboard")
    x1, y1, x2, y2 = gt["stone"]
    # a giant box covering stone AND cube similarly -> ambiguous outcome
    big = (min(x1, gt["cube"][0]) - 5, min(y1, gt["cube"][1]) - 5,
           max(x2, gt["cube"][2]) + 5, max(y2, gt["cube"][3]) + 5)
    fake = SceneDescription(
        objects=[GroundedObject("pX", "stone", GroundStatus.UNLOCALIZED, bbox_xyxy=big, confidence=0.5)],
        sim_time=world.sim_time,
    )
    result = oracle.associate(fake, camera="onboard")
    assert ("pX" in result.ambiguous) or ("pX" in result.unmatched_perceived)
    assert "pX" not in result.matches


def test_associate_guards_against_stale_state(world, oracle):
    world.reset(standard_scene())
    world.step(200)
    scene = SceneDescription(sim_time=world.sim_time)
    world.step(10)  # state moved on
    with pytest.raises(ValueError, match="sim state"):
        oracle.associate(scene)


def test_region_predicate_has_finite_bounds(world, oracle):
    world.reset(standard_scene())
    world.step(200)
    b = oracle.region_bounds()
    assert b.half_extents_xy[0] < 0.5 and b.half_extents_xy[1] < 0.5  # finite, not a plane
    assert b.support_z == pytest.approx(0.85, abs=1e-3)  # table top (assets/scene_common.xml)
    # on-table but outside bounds -> False
    world.teleport_body("stone", np.array([b.center_xy[0] - 0.3, b.center_xy[1], b.support_z + 0.03]))
    assert not oracle.object_in_region("stone")
    # inside bounds at support height -> True
    world.teleport_body("stone", np.array([b.center_xy[0], b.center_xy[1], b.support_z + 0.025]))
    assert oracle.object_in_region("stone")
    # hovering far above the region -> False
    world.teleport_body("stone", np.array([b.center_xy[0], b.center_xy[1], b.support_z + 0.40]))
    assert not oracle.object_in_region("stone")
