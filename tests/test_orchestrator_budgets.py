"""Final2 6.6: episode endings per orchestrator budget."""
from eval.orchestrator_budgets import ending


def rec(outcome, *limits):
    return {"outcome": outcome, "events": [{"type": "limit", "which": w} for w in limits]}


def test_endings():
    assert ending(rec("CLAIMED_SUCCESS")) == "success"
    assert ending(rec("SEARCH_EXHAUSTED")) == "max_search"
    assert ending(rec("LIMIT_EXCEEDED", "max_total_plans")) == "max_total_plans"
    assert ending(rec("LIMIT_EXCEEDED")) == "max_replans"
    assert ending(rec("FAILED")) == "max_replans"
    assert ending(rec("CLARIFICATION_EXHAUSTED")) == "max_clarify"
    assert ending(rec("REFUSED")) == "refused"
    assert ending(rec("ERROR")) == "error"
