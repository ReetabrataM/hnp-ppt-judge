from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.export import export_results, merge_output_directories


def deck(deck_id: str, team: str, total: float) -> dict:
    return {
        "deck_id": deck_id,
        "deck_name": f"{team}.pdf",
        "page_count": 8,
        "stage": "complete",
        "score_variance": 0.3,
        "needs_human_review": False,
        "flags": [],
        "profile": {
            "team_name": team,
            "strengths": ["Working prototype"],
            "weaknesses": ["Small validation sample"],
            "evidence_slides": [2, 4],
        },
        "scores": {
            "weighted_total": total,
            "confidence": "medium",
            "overall_reason": "Evidence is plausible.",
            "criteria": {
                "novelty": {
                    "score": 7.0,
                    "weight": 0.25,
                    "stdev": 0.2,
                    "reason": "Differentiator appears on slide 2.",
                    "evidence_slides": [2],
                },
                "feasibility": {
                    "score": 6.5,
                    "weight": 0.25,
                    "stdev": 0.3,
                    "reason": "Architecture shown on slide 4.",
                    "evidence_slides": [4],
                },
            },
        },
    }


def test_export_has_required_columns_evidence_and_excel_bom(tmp_path: Path) -> None:
    paths = export_results([deck("deck-1", "প্রগতি", 7.2)], output_dir=tmp_path)
    assert paths["summary"].read_bytes().startswith(b"\xef\xbb\xbf")
    summary = pd.read_csv(paths["summary"], encoding="utf-8-sig")
    details = pd.read_csv(paths["details"], encoding="utf-8-sig")
    assert {"team", "team_name", "weighted_total_10", "rank", "flags", "confidence", "needs_human_review"} <= set(summary.columns)
    assert {"reason", "strengths", "weaknesses", "evidence_slides", "deck_summary"} <= set(details.columns)
    assert summary.loc[0, "team_name"] == "প্রগতি"
    assert details["reason"].notna().all()
    assert all(json.loads(value) for value in details["evidence_slides"])
    assert details["deck_summary"].str.split().str.len().le(50).all()


def test_export_limits_profile_summary_to_fifty_words(tmp_path: Path) -> None:
    state = deck("deck-1", "Alpha", 7.2)
    state["profile"].update({
        "problem": {"value": "problem " * 30, "slides": [1]},
        "solution": {"value": "solution " * 30, "slides": [2]},
        "target_users": {"value": "students", "slides": [1]},
    })
    paths = export_results([state], output_dir=tmp_path)
    details = pd.read_csv(paths["details"], encoding="utf-8-sig")
    assert details.loc[0, "deck_summary"].split()[-1].endswith("…")
    assert len(details.loc[0, "deck_summary"].split()) == 50


def test_merge_removes_duplicate_decks_and_recalculates_rank(tmp_path: Path) -> None:
    machine_one = tmp_path / "machine-one"
    machine_two = tmp_path / "machine-two"
    merged = tmp_path / "merged"
    export_results([deck("deck-1", "Alpha", 6.0)], output_dir=machine_one)
    export_results([deck("deck-1", "Alpha", 8.0), deck("deck-2", "Beta", 7.0)], output_dir=machine_two)
    paths = merge_output_directories([machine_one, machine_two], merged)
    summary = pd.read_csv(paths["summary"], encoding="utf-8-sig")
    details = pd.read_csv(paths["details"], encoding="utf-8-sig")
    assert summary["deck_id"].tolist() == ["deck-1", "deck-2"]
    assert summary["rank"].tolist() == [1, 2]
    assert len(details[details["deck_id"] == "deck-1"]) == 2
