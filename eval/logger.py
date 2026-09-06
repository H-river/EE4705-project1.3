# Owner: backbone (ALL)
"""Strict-JSON trial logging.

Every value is converted before serialization: enums -> their string value,
numpy arrays/scalars -> lists/plain numbers, dataclasses -> dicts, and
non-finite floats -> an explicit {"__nonfinite__": "<repr>"} marker (strict
JSON has no NaN/Infinity literals; json.dumps runs with allow_nan=False so
an invalid number can never be emitted).
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import pathlib
from typing import Any

import numpy as np

from core.types import TrialRecord


def to_json_safe(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return [to_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"__nonfinite__": repr(value)}
        return value
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_json_safe(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def write_trial_record(record: TrialRecord, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_json_safe(record)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def read_trial_record(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())
