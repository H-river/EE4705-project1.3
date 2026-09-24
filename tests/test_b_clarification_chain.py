"""E4: clarification-chain tool; the figure renders from a recorded chain (no model call)."""
from eval.b_clarification_chain import CASES, timeline_figure


def test_cases_are_ambiguous_with_a_scripted_answer():
    assert len(CASES) == 4
    for c in CASES:
        assert c["answer"] and c["expect_object"] in {o.instance_id for o in c["objects"]}


def test_timeline_renders(tmp_path):
    chain = {"id": "c1", "instruction": "move the stone", "answer": "the gray one", "final_status": "READY",
             "goal": {"object_id": "a2"}, "correct": True,
             "steps": [{"kind": "plan", "t0": 0, "t1": 5.1, "status": "NEEDS_CLARIFICATION", "question": "Which one?"},
                       {"kind": "answer", "t0": 5.1, "t1": 5.1, "text": "the gray one"},
                       {"kind": "plan", "t0": 5.1, "t1": 9.0, "status": "READY", "question": None}]}
    timeline_figure([chain, {**chain, "id": "c2"}], tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000
