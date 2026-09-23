# Owner: backbone (ALL)
"""Public static scene geometry: the table, read from the scene MJCF.

The table is fixed furniture in every trial (SceneConfig only moves the
robot and the manipulable objects), so its box is public knowledge in the
same sense as assets/objects.yaml.  Values are parsed from the ``table``
body and its ``table_top`` geom, never hardcoded.
"""

from __future__ import annotations

import functools
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

SCENE_COMMON = pathlib.Path(__file__).resolve().parent.parent / "assets" / "scene_common.xml"


def _vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.array(default, dtype=float)


@functools.lru_cache(maxsize=None)
def table_aabb(path: pathlib.Path = SCENE_COMMON) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """World-frame (lo, hi) corners of the table top box."""
    root = ET.parse(path).getroot()
    body = root.find(".//body[@name='table']")
    geom = body.find("geom[@name='table_top']") if body is not None else None
    if geom is None or geom.get("type", "sphere") != "box":
        raise ValueError(f"{path}: no box geom 'table_top' in body 'table'")
    if body.get("quat") or body.get("euler") or geom.get("quat") or geom.get("euler"):
        raise ValueError(f"{path}: rotated table is not supported")
    center = _vec(body.get("pos"), (0, 0, 0)) + _vec(geom.get("pos"), (0, 0, 0))
    half = _vec(geom.get("size"), (0, 0, 0))
    return tuple((center - half).tolist()), tuple((center + half).tolist())


def table_top_z() -> float:
    """Height of the table's top surface in the world frame."""
    return table_aabb()[1][2]


def table_center_xy() -> tuple[float, float]:
    lo, hi = table_aabb()
    return (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
