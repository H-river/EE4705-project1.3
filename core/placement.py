# Owner: backbone (ALL)
"""Shared placement tolerances; inputs may be estimated or oracle-measured.

No ground truth is read here. The evaluator and visual verifier supply
their own independent positions and region bounds.
"""

import numpy as np

MIN_SUPPORT_OFFSET_M = -0.005
MAX_SUPPORT_OFFSET_M = 0.12
STABILITY_DRIFT_M = 0.02


def position_in_region(pos, support_pos, half_extents_xy) -> bool:
    p, r, half = (np.asarray(x, dtype=float) for x in (pos, support_pos, half_extents_xy))
    if (p.shape != (3,) or r.shape != (3,) or half.shape != (2,)
            or not all(np.all(np.isfinite(x)) for x in (p, r, half)) or np.any(half <= 0)):
        return False
    delta = p - r
    return bool(np.all(np.abs(delta[:2]) <= half)
                and MIN_SUPPORT_OFFSET_M <= delta[2] <= MAX_SUPPORT_OFFSET_M)
