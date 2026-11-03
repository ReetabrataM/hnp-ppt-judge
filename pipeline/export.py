# pandas to Excel, CSV
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pipeline.state import DeckState


SUMMARY_FILE = "summary.csv"
DETAILS_FILE = "details.csv"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else ([] if value is None else [value])


def _append_flag(flags: list[dict[str, Any]], code: str, message: str) -> None:
    if not any(item.get("code") == code for item in flags):
        flags.append({"code": code, "severity": "medium", "message": message})


def _team_name(state: DeckState) -> str:
    profile = state.get("profile", {})
    return str(profile.get("team_name") or profile.get("title") or Path(str(state.get("deck_name", "Unknown team"))).stem)


def _weighted(score: dict[str, Any]) -> float | None:
    try:
        return round(float(score["score"]) * float(score["weight"]), 3)
    except (KeyError, TypeError, ValueError):
        return None


def _rank_summary(rows: list[dict[str, Any]]) -> None:
    ranked = [row for row in rows if row.get("weighted_total_10") is not None]
    ranked.sort(key=lambda row: (-float(row["weighted_total_10"]), str(row["team_name"]).casefold(), str(row["deck_id"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["metric_rank"] = rank


def _rows_for_state(state: DeckState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = dict(state.get("profile", {}))
    scores = dict(state.get("scores", {}))
    criteria = scores.get("criteria", {}) if isinstance(scores.get("criteria", {}), dict) else {}
    flags = [dict(item) for item in _list(state.get("flags", [])) if isinstance(item, dict)]
    evidence_fallback = _list(profile.get("evidence_slides", []))
    team_name = _team_name(state)
    summary: dict[str, Any] = {
        "deck_id": state.get("deck_id"),
        "team": team_name,
        "team_name": team_name,
        "deck_name": state.get("deck_name"),
        "status": state.get("stage", "complete"),
        "page_count": state.get("page_count"),
        "weighted_total_10": scores.get("weighted_total"),
        "confidence": scores.get("confidence", "low"),
        "score_variance": state.get("score_variance", 0.0),
        "rank": None,
        "metric_rank": None,
        "flags": "",
        "needs_human_review": bool(state.get("needs_human_review", False)),
        "strengths": _json(_list(profile.get("strengths", []))),
        "weaknesses": _json(_list(profile.get("weaknesses", []))),
        "comparison": _json(state.get("comparison", {})),
        "failure": state.get("failure"),
        "rank_note": "Sorting aid only; organizers make the final decision.",
    }
    details: list[dict[str, Any]] = []
    for criterion, item in criteria.items():
        score = dict(item) if isinstance(item, dict) else {}
        evidence = _list(score.get("evidence_slides", evidence_fallback))
        reason = str(score.get("reason") or "No model rationale was provided.")
        if not score.get("reason"):
            _append_flag(flags, "missing_score_reason", f"{criterion} has no model-supplied reason.")
        if not evidence:
            _append_flag(flags, "missing_score_evidence", f"{criterion} has no slide reference; review manually.")
            summary["needs_human_review"] = True
        summary[f"{criterion}_score"] = score.get("score")
        details.append(
            {
                "deck_id": state.get("deck_id"),
                "team": team_name,
                "team_name": team_name,
                "deck_name": state.get("deck_name"),
                "criterion": criterion,
                "score_10": score.get("score"),
                "criterion_stdev": score.get("stdev"),
                "weight": score.get("weight"),
                "weighted_contribution": _weighted(score),
                "reason": reason,
                "evidence_slides": _json(evidence),
                "strengths": _json(_list(profile.get("strengths", []))),
                "weaknesses": _json(_list(profile.get("weaknesses", []))),
                "overall_reason": scores.get("overall_reason", ""),
                "profile": _json(profile),
                "flags": "",
            }
        )
    summary["flags"] = _json(flags)
    for row in details:
        row["flags"] = summary["flags"]
    return summary, details


def _resolve_output_dir(settings: Any | None, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    candidate = getattr(settings, "output_dir", None)
    if candidate is None:
        raise ValueError("Pass output_dir or a settings object with output_dir.")
    return Path(candidate)


def export_results(states: Iterable[DeckState], settings: Any | None = None, *, output_dir: Path | None = None) -> dict[str, Path]:
    """Write summary.csv and details.csv with the Excel-compatible UTF-8 BOM."""
    destination = _resolve_output_dir(settings, output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state in states:
        deck_id = str(state.get("deck_id") or "")
        if not deck_id:
            raise ValueError("Every exported deck must have a stable deck_id.")
        if deck_id in seen:
            raise ValueError(f"Duplicate deck_id in this export: {deck_id}")
        seen.add(deck_id)
        summary, details = _rows_for_state(state)
        summary_rows.append(summary)
        detail_rows.extend(details)
    _rank_summary(summary_rows)
    summary_path = destination / SUMMARY_FILE
    details_path = destination / DETAILS_FILE
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(details_path, index=False, encoding="utf-8-sig")
    return {"summary": summary_path, "details": details_path}


def _read_required(directory: Path, filename: str) -> pd.DataFrame:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Expected {path}")
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def merge_output_directories(input_dirs: Iterable[Path], output_dir: Path) -> dict[str, Path]:
    """Merge teammate output folders. Later folders win duplicate deck_id conflicts."""
    folders = [Path(folder) for folder in input_dirs]
    if not folders:
        raise ValueError("Provide at least one output directory to merge.")
    summary = pd.concat([_read_required(folder, SUMMARY_FILE) for folder in folders], ignore_index=True)
    details = pd.concat([_read_required(folder, DETAILS_FILE) for folder in folders], ignore_index=True)
    for frame, columns in ((summary, {"deck_id", "weighted_total_10"}), (details, {"deck_id", "criterion"})):
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"Cannot merge: missing columns {sorted(missing)}")
    summary = summary.drop_duplicates(subset=["deck_id"], keep="last").copy()
    details = details.drop_duplicates(subset=["deck_id", "criterion"], keep="last").copy()
    summary["weighted_total_10"] = pd.to_numeric(summary["weighted_total_10"], errors="coerce")
    summary = summary.sort_values(["weighted_total_10", "team_name", "deck_id"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
    summary["rank"] = pd.Series([index + 1 if pd.notna(value) else pd.NA for index, value in enumerate(summary["weighted_total_10"])], dtype="Int64")
    summary["metric_rank"] = summary["rank"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILE
    details_path = output_dir / DETAILS_FILE
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    return {"summary": summary_path, "details": details_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Hack-n-Pitch CSV output folders.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Team output folders, in precedence order.")
    parser.add_argument("--output", type=Path, required=True, help="Destination for merged files.")
    args = parser.parse_args()
    paths = merge_output_directories(args.inputs, args.output)
    print(f"Wrote {paths['summary']} and {paths['details']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
