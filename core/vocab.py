# Owner: backbone (ALL)
"""Shared object vocabulary loaded from assets/objects.yaml.

Maps instruction phrases (synonyms) to normalized class names and default
attributes.  Used by the mocks; Student A/B may reuse it so grounding and
planning share one vocabulary.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

import yaml

DEFAULT_PATH = pathlib.Path(__file__).resolve().parent.parent / "assets" / "objects.yaml"

# Color aliases for attribute qualifiers in instructions/clarifications.
COLOR_ALIASES = {
    "gray": ["gray", "grey", "silver", "light"],
    "dark_red": ["dark red", "dark-red", "reddish", "dark", "brown", "maroon"],
    "blue": ["blue"],
    "green": ["green"],
    "red": ["red"],
}


@dataclass
class VocabEntry:
    gt_id: str  # ground-truth model id (body name) — never exposed to planners
    cls: str  # normalized class name
    kind: str  # "object" | "region"
    synonyms: list[str]
    attributes: dict[str, str]
    shape: str
    size_xyz: tuple[float, float, float]


class Vocab:
    def __init__(self, path: pathlib.Path = DEFAULT_PATH) -> None:
        raw = yaml.safe_load(path.read_text())
        self.entries: list[VocabEntry] = []
        for kind, section in (("object", "objects"), ("region", "regions")):
            for gt_id, spec in (raw.get(section) or {}).items():
                self.entries.append(
                    VocabEntry(
                        gt_id=gt_id,
                        cls=spec.get("class", gt_id),
                        kind=kind,
                        synonyms=[s.lower() for s in spec.get("synonyms", [gt_id])],
                        attributes=dict(spec.get("attributes", {})),
                        shape=spec.get("shape", "box"),
                        size_xyz=tuple(spec.get("size_xyz", (0.05, 0.05, 0.05))),
                    )
                )

    def entry_for_gt(self, gt_id: str) -> Optional[VocabEntry]:
        for e in self.entries:
            if e.gt_id == gt_id:
                return e
        return None

    def find_phrase(self, text: str, kind: str) -> Optional[tuple[str, Optional[str]]]:
        """Find the longest synonym of the given kind in ``text``.
        Returns (class_name, color_qualifier_or_None).  The qualifier is set
        when the matched synonym is specific to entries sharing one color."""
        text = text.lower()
        best: Optional[tuple[str, str]] = None  # (synonym, class)
        for e in self.entries:
            if e.kind != kind:
                continue
            for syn in e.synonyms:
                if syn in text and (best is None or len(syn) > len(best[0])):
                    best = (syn, e.cls)
        if best is None:
            return None
        syn, cls = best
        colors = {
            e.attributes.get("color")
            for e in self.entries
            if e.kind == kind and e.cls == cls and syn in e.synonyms
        }
        qualifier = colors.pop() if len(colors) == 1 and None not in colors else None
        # A synonym shared by all same-class entries carries no qualifier.
        all_colors = {
            e.attributes.get("color") for e in self.entries if e.kind == kind and e.cls == cls
        }
        if qualifier is not None and len(all_colors) == 1:
            qualifier = None  # only one variant exists; qualifier is meaningless
        return cls, qualifier

    @staticmethod
    def color_in_text(text: str) -> Optional[str]:
        """Extract an explicit color qualifier from free text (longest alias
        wins, so 'dark red' beats 'red')."""
        text = text.lower()
        best: Optional[tuple[str, str]] = None
        for color, aliases in COLOR_ALIASES.items():
            for alias in aliases:
                if alias in text and (best is None or len(alias) > len(best[0])):
                    best = (alias, color)
        return best[1] if best else None
