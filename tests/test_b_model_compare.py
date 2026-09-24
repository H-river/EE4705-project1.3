"""E2: tier comparison reads b_benchmark folders and orders flash < plus < max."""
import json

from eval.b_model_compare import load, table


def _suite(root, name, correct, n=2):
    d = root / name
    for i in range(n):
        (d / f"test_{i}").mkdir(parents=True)
        (d / f"test_{i}" / "result.json").write_text(json.dumps(
            {"latency_s": 1.0 + i, "errors": [], "api_stats": {"attempts": 1, "prompt_tokens": 10,
                                                                "completion_tokens": 2}}))
    (d / "summary.json").write_text(json.dumps({"n_cases": n, "n_correct": correct, "accuracy": correct / n,
                                                "first_response_accuracy": correct / n}))


def test_load_orders_tiers_and_sums_tokens(tmp_path):
    for m in ("qwen3.7-max-x", "qwen3.7-flash-x", "qwen3.7-plus-x"):
        _suite(tmp_path, f"{m}_eval", 2)
    rows = load(tmp_path)
    assert [r["model"] for r in rows] == ["qwen3.7-flash-x", "qwen3.7-plus-x", "qwen3.7-max-x"]
    assert rows[0]["calls"] == 2 and rows[0]["prompt_tokens"] == 20 and rows[0]["latency_mean_s"] == 1.5
    assert "2/2 (100.0%)" in table(rows)
