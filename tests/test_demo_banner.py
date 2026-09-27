"""Video end banner: correct refusals are not shown as failures."""
from demo.recording import end_banner


def test_end_banner():
    assert end_banner(True, True) == "DONE / PASS"
    assert end_banner(False, False, refusal_correct=True) == "DONE / REFUSED (correct)"
    assert end_banner(False, False) == "DONE / FAIL"
    assert end_banner(True, False) == "DONE / FAIL"  # a false claim is never dressed up
