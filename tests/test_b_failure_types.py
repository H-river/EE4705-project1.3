"""Final2 6.2: failure-type classification of B errors."""
import pytest

from eval.b_failure_types import classify


@pytest.mark.parametrize("error,ids,kind", [
    ("status: expected INFEASIBLE, got REJECTED", None, "WRONG_STATUS"),
    ("Unexpected search: SEARCH blue cube", None, "WRONG_STATUS"),
    ("object_id: expected v194, got v514", {"v194", "v514"}, "WRONG_BINDING"),
    ("object_id: expected v194, got v999", {"v194"}, "HALLUCINATED_INSTANCE"),
    ("object_id: expected v662, got ", {"v662"}, "WRONG_BINDING"),
    ("Unknown perceived region ID 'red_region'", None, "HALLUCINATED_INSTANCE"),
    ("Unknown region class 'drawer'", None, "WRONG_GOAL"),
    ("Original goal must keep object_id='i536'", None, "WRONG_GOAL"),
    ("Goal object color mismatch", None, "WRONG_BINDING"),
    ("$.rationale: violates maxLength=300", None, "INVALID_PARAMS"),
    ("Action 2 (MOVE_TO): object='a1' is not allowed; expected an empty string", None, "INVALID_PARAMS"),
])
def test_classify(error, ids, kind):
    assert classify(error, ids) == kind
