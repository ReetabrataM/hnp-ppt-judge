from __future__ import annotations

from unittest.mock import patch

from pipeline.compare import compare_decks, profile_to_text


def _profile(problem: str, solution: str) -> dict:
    return {
        "problem": {"value": problem, "slides": [1]},
        "solution": {"value": solution, "slides": [2]},
        "tech_stack": {"value": "python", "slides": [3]},
        "claimed_results_metrics": {"value": "not stated", "slides": []},
        "demo_evidence": {"value": "not stated", "slides": []},
        "target_users": {"value": "users", "slides": [1]},
        "architecture": {"value": "not stated", "slides": []},
        "business_model": {"value": "not stated", "slides": []},
        "team": {"value": "3 students", "slides": [4]},
        "domain_track": {"value": "general", "slides": [1]},
    }


def test_profile_to_text_flattens_values():
    text = profile_to_text(_profile("farm prices", "sms alerts"))
    assert "farm prices" in text
    assert "sms alerts" in text
    assert "not stated" not in text


def test_compare_assigns_cluster_novelty_and_duplicates():
    decks = [
        {"deck_id": "a", "weighted_total": 18.0, "profile": _profile("farm prices", "sms alerts")},
        {"deck_id": "b", "weighted_total": 17.5, "profile": _profile("farm prices", "sms price alerts")},
        {"deck_id": "c", "weighted_total": 12.0, "profile": _profile("hospital triage", "ml chatbot")},
    ]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.995, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ]
    pairwise = {"preferred": "A", "reason": "clearer scope", "agrees_with_rubric_order": True}
    with patch("pipeline.compare.embeddings", return_value=vectors), patch(
        "pipeline.compare.text_completion", return_value=pairwise
    ) as mock_pair:
        out = compare_decks(decks)

    by_id = {d["deck_id"]: d for d in out}
    assert by_id["a"]["cluster_id"] == by_id["b"]["cluster_id"]
    assert by_id["c"]["cluster_id"] != by_id["a"]["cluster_id"]
    assert by_id["a"]["novelty"] < by_id["c"]["novelty"]
    assert by_id["a"]["near_duplicates"][0]["deck_id"] == "b"
    assert by_id["c"]["near_duplicates"] == []
    assert any(f.get("type") == "near_duplicate" for f in by_id["a"]["flags"])
    assert by_id["a"]["pairwise"]
    assert mock_pair.call_args.kwargs["provider"] == "groq"


def test_compare_skips_pairwise_without_scores():
    decks = [
        {"deck_id": "a", "profile": _profile("x", "y")},
        {"deck_id": "b", "profile": _profile("x", "z")},
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    with patch("pipeline.compare.embeddings", return_value=vectors), patch(
        "pipeline.compare.text_completion"
    ) as mock_pair:
        out = compare_decks(decks)
    mock_pair.assert_not_called()
    assert out[0]["pairwise"] == []
    assert out[1]["cluster_id"] is not None


def test_compare_handles_embedding_failure():
    with patch("pipeline.compare.embeddings", side_effect=RuntimeError("quota")):
        out = compare_decks([{"deck_id": "a", "profile": _profile("x", "y")}])
    assert out[0]["cluster_id"] == 0
    assert out[0]["novelty"] is None
    assert any(f.get("type") == "compare_failed" for f in out[0]["flags"])


def test_pairwise_budget_is_capped():
    decks = [
        {"deck_id": f"t{i}", "weighted_total": 20 - i, "profile": _profile(f"p{i}", f"s{i}")}
        for i in range(12)
    ]
    vectors = [[float(i), 1.0, 0.0] for i in range(12)]
    with patch("pipeline.compare.embeddings", return_value=vectors), patch(
        "pipeline.compare.text_completion",
        return_value={"preferred": "A", "reason": "ok", "agrees_with_rubric_order": True},
    ) as mock_pair, patch("pipeline.compare.MAX_PAIRWISE_CALLS", 3), patch(
        "pipeline.compare.TOP_TIER_SIZE", 10
    ):
        compare_decks(decks)
    assert mock_pair.call_count == 3
