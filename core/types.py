# Owner: backbone (ALL)
"""Shared data contracts for the EE4705 Project 1.3 backbone.

All inter-module data crosses these dataclasses only.  Implementation modules
(perception/, planner/, executor/) must not import each other; they exchange
only the types defined here.

Contract semantics
==================

Coordinate frames and units
---------------------------
* World frame: right-handed, z up, units are meters.  Angles are radians.
* ``pos_world`` on a :class:`GroundedObject` is the estimated 3D position of
  the object's center, or a region's support-surface point, in the **world
  frame** (meters).  ``None`` means the
  object has not been localized in 3D (see :class:`GroundStatus`).
* Camera frame (public convention): +x right, +y down, +z forward (optical
  axis).  ``t_world_camera`` is the 4x4 homogeneous transform mapping points
  expressed in this camera frame into the world frame:
  ``p_world = T_world_camera @ [p_cam, 1]``.
* ``intrinsics`` is the 3x3 pinhole matrix K for the public camera frame.
  Pixel (u, v) has u increasing rightwards, v downwards.
* ``depth`` stores the metric z-distance in meters along the camera optical
  axis (NOT the ray length).  Invalid or background samples are ``NaN``.

Perceived IDs versus ground-truth IDs
-------------------------------------
Perception assigns its own opaque ``instance_id`` strings (e.g. ``"p3"``)
with a perception-side lifecycle: they are stable within an episode for the
same tracked instance and reset between episodes.  They are NEVER equal to
simulator body names or ground-truth object IDs, and no component may assume
any string relationship between the two ID spaces.  Only :class:`EvalOracle
<core.oracle.EvalOracle>` may associate perceived instances with ground-truth
objects, and it does so geometrically (projected-bbox IoU), never by string
comparison.

Association uncertainty versus localization failure
---------------------------------------------------
These are distinct failure signals, both carried by ``GroundedObject.status``:

* ``GroundStatus.AMBIGUOUS`` — perception cannot decide *which* of several
  candidate instances the referent is (association uncertainty).  The plan
  layer must not act on such a reference without clarification.
* ``GroundStatus.UNLOCALIZED`` — the instance is confidently identified and
  visible (2D evidence exists) but has no reliable 3D position
  (localization failure).  Visibility-only checks (e.g. VERIFY
  object_visible) may use it; metric skills (REACH, GRASP) may not.
* ``GroundStatus.NOT_FOUND`` — no visual evidence for the referent at all.

Observation ownership and lifetime
----------------------------------
An :class:`Observation` owns its arrays: producers must copy out of any
renderer-internal buffers before constructing it, and consumers must treat
the arrays as immutable.  Observations are retained by the
:class:`~core.obs_store.ObservationStore` for a bounded window (32 frames);
holding an ``Observation`` reference beyond that is allowed (it is a plain
value object) but ``frame_id`` lookups in the store may fail after eviction.
Anything needed for post-hoc inspection must be persisted via the store
before eviction.

Status, error and verification semantics
----------------------------------------
* ``ExecutionResult.success`` reports whether the *single action* achieved
  its own postcondition as far as the executor can tell.  It is never a
  claim about overall task success.
* ``error_code`` refines failures.  ``SEARCH_NOT_FOUND`` is a recoverable
  outcome (the target was not seen from any scanned viewpoint); fatal search
  errors (e.g. camera failure) use other codes and abort the episode.
* Task-level success may only be claimed after final verification
  (:class:`VerificationResult`) passes; completing an action list is never
  sufficient.  ``CLAIMED_SUCCESS`` is the system's own claim; ground-truth
  "actual success" is judged exclusively by the evaluation layer.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# v2: full-motion STOP and fresh, exact-instance placement verification.
CONTRACT_VERSION = 2

# Fixed public image resolution.
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


class Skill(enum.Enum):
    """The eight action primitives a Plan may contain."""

    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    REACH = "REACH"
    GRASP = "GRASP"
    MOVE_TO = "MOVE_TO"
    PLACE = "PLACE"
    VERIFY = "VERIFY"
    STOP = "STOP"


class PlanStatus(enum.Enum):
    """Planner verdict attached to a Plan.

    * ``READY`` — every action is executable with currently grounded
      knowledge; SEARCH is not allowed inside a READY plan.
    * ``NEEDS_SEARCH`` — the referent is not currently visible; the plan
      may begin with SEARCH actions.
    * ``NEEDS_CLARIFICATION`` — the instruction is ambiguous;
      ``clarification_question`` must be set and ``actions`` empty.
    * ``INFEASIBLE`` — the instruction is refused; ``reason`` must be set
      and ``actions`` empty.
    """

    READY = "READY"
    NEEDS_SEARCH = "NEEDS_SEARCH"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    INFEASIBLE = "INFEASIBLE"


class ErrorCode(enum.Enum):
    NONE = "NONE"
    GRASP_MISSED = "GRASP_MISSED"
    TARGET_LOST = "TARGET_LOST"
    UNREACHABLE = "UNREACHABLE"
    TIMEOUT = "TIMEOUT"
    SEARCH_NOT_FOUND = "SEARCH_NOT_FOUND"  # recoverable: scanned, target absent
    SEARCH_FATAL = "SEARCH_FATAL"  # non-recoverable search failure
    NOT_HOLDING = "NOT_HOLDING"
    ALREADY_HOLDING = "ALREADY_HOLDING"
    PLACE_FAILED = "PLACE_FAILED"
    VERIFY_FAILED = "VERIFY_FAILED"
    INVALID_ACTION = "INVALID_ACTION"
    PERCEPTION_ERROR = "PERCEPTION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GroundStatus(enum.Enum):
    """Grounding quality of a GroundedObject (see module docstring)."""

    LOCALIZED = "LOCALIZED"  # 2D evidence + valid pos_world
    UNLOCALIZED = "UNLOCALIZED"  # 2D evidence, no reliable 3D position
    AMBIGUOUS = "AMBIGUOUS"  # multiple indistinguishable candidates
    NOT_FOUND = "NOT_FOUND"  # no visual evidence


class TrialOutcome(enum.Enum):
    """Orchestrator-level outcome of one episode (the system's own view)."""

    CLAIMED_SUCCESS = "CLAIMED_SUCCESS"  # final verification passed
    FAILED = "FAILED"  # execution/verification failed within limits
    REFUSED = "REFUSED"  # planner declared the instruction infeasible
    CLARIFICATION_EXHAUSTED = "CLARIFICATION_EXHAUSTED"
    SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"  # replan/attempt/global bounds hit
    ERROR = "ERROR"  # unexpected exception; stopped safely
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class Observation:
    """One RGB-D capture.  Arrays are owned copies; treat as immutable.

    ``rgb``: uint8, shape (480, 640, 3).
    ``depth``: float32, shape (480, 640); metric optical-axis z in meters,
    NaN where invalid/background.
    ``intrinsics``: float64 (3, 3) pinhole K.
    ``t_world_camera``: float64 (4, 4) camera(+x right, +y down, +z fwd)
    to world transform.
    ``capture_id``: batch identifier shared (up to the camera suffix) by all
    observations of one atomic multi-camera capture.
    """

    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    t_world_camera: np.ndarray
    sim_time: float
    camera_name: str
    # Batch identity for (multi-)camera captures: ``ep<episode>_capture<n>_<camera>``.
    # Observations captured together share the ``ep.._capture..`` prefix and
    # the same ``sim_time``.  Empty for synthetic/legacy observations.
    capture_id: str = ""

    def __post_init__(self) -> None:
        if self.rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8 (480,640,3), got {self.rgb.dtype} {self.rgb.shape}")
        if self.depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"depth must have shape (480,640), got {self.depth.shape}")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must be 3x3")
        if self.t_world_camera.shape != (4, 4):
            raise ValueError("t_world_camera must be 4x4")


@dataclass(frozen=True)
class RobotState:
    """Proprioceptive snapshot returned by ``RobotEnv.get_robot_state()``
    (no ground truth about the scene).

    ``base_pose``: (x, y, yaw) of the sliding base, world frame.
    ``ee_pos`` / ``ee_quat``: end-effector site (2F-85 pinch point / TCP)
    position and orientation (w, x, y, z) in the world frame.
    ``gripper_opening``: MEASURED normalized opening per side,
    1.0 = fully open, 0.0 = closed (same normalization as ``set_gripper``).
    ``attached``: an attachment (weld or physical grasp) is currently held.
    ``last_attach_reason``: outcome of the most recent attach attempt
    ("attached", "no_contact", "single_pad_contact", ...).
    """

    sim_time: float
    base_pose: tuple[float, float, float]
    ee_pos: tuple[float, float, float]
    ee_quat: tuple[float, float, float, float]
    gripper_opening: dict[str, float]
    attached: bool
    last_attach_reason: str = ""


@dataclass
class GroundedObject:
    """A perceived object or region instance.

    ``instance_id`` is a perception-side opaque ID (see module docstring).
    ``name`` is the normalized class name from assets/objects.yaml.
    ``bbox_xyxy`` is (x_min, y_min, x_max, y_max) in pixels, or None when
    no 2D evidence exists (status NOT_FOUND).
    ``pos_world`` is the world-frame object center or region support-surface
    point in meters, or None.
    ``kind`` distinguishes manipulable objects from support regions.
    """

    instance_id: str
    name: str
    status: GroundStatus
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    pos_world: Optional[tuple[float, float, float]] = None
    confidence: float = 0.0
    source: str = "vlm"  # "vlm" | "gt"
    kind: str = "object"  # "object" | "region"
    frame_id: int = -1
    attributes: dict[str, str] = field(default_factory=dict)  # e.g. {"color": "gray"}
    # Optional perceived/declared axis-aligned region half-size in world meters.
    # For a region, pos_world is the support-surface point, not the visual slab center.
    region_half_extents_xy: Optional[tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.status is GroundStatus.LOCALIZED and self.pos_world is None:
            raise ValueError("LOCALIZED instance requires pos_world")
        if self.pos_world is not None:
            p = np.asarray(self.pos_world, dtype=float)
            if p.shape != (3,) or not np.all(np.isfinite(p)):
                raise ValueError(f"pos_world must be a finite 3-vector, got {self.pos_world}")
            self.pos_world = (float(p[0]), float(p[1]), float(p[2]))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.region_half_extents_xy is not None:
            half = np.asarray(self.region_half_extents_xy, dtype=float)
            if self.kind != "region" or half.shape != (2,) or not np.all(np.isfinite(half)) or np.any(half <= 0):
                raise ValueError("region_half_extents_xy requires a region and two finite positive lengths")
            self.region_half_extents_xy = tuple(float(v) for v in half)


@dataclass
class SceneDescription:
    """Perception -> Planner summary of one observation."""

    objects: list[GroundedObject] = field(default_factory=list)
    regions: list[GroundedObject] = field(default_factory=list)
    caption: str = ""
    ambiguities: list[str] = field(default_factory=list)
    frame_id: int = -1
    sim_time: float = 0.0

    def find(self, instance_id: str) -> Optional[GroundedObject]:
        for g in self.objects + self.regions:
            if g.instance_id == instance_id:
                return g
        return None


@dataclass
class Action:
    """One plan step.  ``target`` references a perceived ``instance_id``
    (for REACH/GRASP/APPROACH/MOVE_TO/PLACE/VERIFY) or a normalized class
    name (for SEARCH, which by definition has no grounded instance yet).
    ``params`` carries skill-specific parameters; see core.validation for
    the per-skill schema."""

    skill: Skill
    target: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)
    status: PlanStatus = PlanStatus.READY
    reason: Optional[str] = None
    clarification_question: Optional[str] = None
    raw_llm_output: str = ""

    @property
    def feasible(self) -> bool:
        return self.status is not PlanStatus.INFEASIBLE


@dataclass(frozen=True)
class PlanError:
    """One validation finding from core.validation.validate_plan."""

    code: str  # e.g. "MISSING_REFERENCE", "BAD_PARAM", see core.validation
    action_index: int  # -1 for plan-level errors
    message: str


@dataclass
class ExecutionContext:
    """Execution-side state snapshot used for plan validation and replanning.

    Owned by the orchestrator; passed by value into validate_plan.
    ``held_instance_id`` is the perceived ID the system believes it holds
    (None when not holding).  ``scene`` is the most recent SceneDescription.
    """

    scene: SceneDescription
    held_instance_id: Optional[str] = None
    last_release_instance_id: Optional[str] = None


@dataclass
class SkillResult:
    """Return type of low-level motion primitives (core.skills)."""

    success: bool
    error_code: ErrorCode = ErrorCode.NONE
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Executor -> Orchestrator result for one action."""

    action: Action
    success: bool
    error_code: ErrorCode = ErrorCode.NONE
    recovery_attempted: bool = False
    post_frame_id: int = -1  # frame_id of the observation taken after acting
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Outcome of task-level (vision-based) final verification."""

    passed: bool
    condition: str  # e.g. "object_in_region"
    detail: str = ""
    frame_id: int = -1


@dataclass
class SceneObjectSpec:
    """Trial-config placement of one ground-truth object."""

    name: str  # canonical model name, e.g. "stone", "stone2"
    pos: tuple[float, float, float]
    yaw: float = 0.0


@dataclass
class SceneConfig:
    """Deterministic scene setup for one trial.  ``seed`` drives all
    randomization (fault injection, tie-breaking noise); the environment
    itself adds no unseeded randomness."""

    seed: int = 0
    objects: list[SceneObjectSpec] = field(default_factory=list)
    robot_init: dict[str, float] = field(default_factory=dict)  # x, y, yaw
    # Visibility of the second stone is config-controlled: include/exclude
    # "stone2" in ``objects`` and choose its pos (in/out of initial view).


@dataclass
class ClarificationExchange:
    question: str
    response: Optional[str]  # None if no scripted response remained


@dataclass
class TrialRecord:
    """Serializable record of one trial run (see eval.logger for schema)."""

    trial_id: str
    contract_version: int = CONTRACT_VERSION
    instruction: str = ""
    outcome: str = TrialOutcome.ERROR.value
    claimed_success: bool = False
    actual_success: Optional[bool] = None  # oracle judgement; None if n/a
    module_config: dict[str, str] = field(default_factory=dict)
    infrastructure_check: bool = False  # True when any mock replaces a real module
    events: list[dict[str, Any]] = field(default_factory=list)
    clarifications: list[dict[str, Any]] = field(default_factory=list)
    api_stats: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
