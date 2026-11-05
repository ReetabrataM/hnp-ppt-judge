"""Small state helpers shared by pipeline nodes."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, TypedDict


class DeckState(TypedDict, total=False):
    """State produced for a single PDF deck."""
    pdf_path: str
    deck_id: str
    ingest_ok: bool
    flags: list[dict[str, Any]]
    pages: list[dict[str, Any]]
    textless_pages: list[int]
    diagram_pages: list[int]
    vision_results: list[dict[str, Any]]
    profile: dict[str, Any]
    scores: dict[str, Any]
    score_variance: float
    score_attempts: int
    needs_human_review: bool


class BatchState(TypedDict, total=False):
    """Checkpointed LangGraph state for an entire input batch."""
    decks: list[DeckState]
    output_dir: str
    export_paths: dict[str, str]
    batch_errors: list[str]


def updated(state: dict[str, Any], **values: Any) -> dict[str, Any]:
    result = deepcopy(state)
    result.update(values)
    return result


def add_flag(
    state: dict[str, Any],
    code: str,
    message: str,
    *,
    severity: str = "warning",
    slide: int | None = None,
) -> dict[str, Any]:
    flags = list(state.get("flags", []))
    if not any(
        isinstance(flag, dict) and (flag.get("code") == code or flag.get("type") == code)
        for flag in flags
    ):
        flags.append({
            "type": code,
            "code": code,
            "message": message,
            "severity": severity,
            "slide": slide,
        })
    return updated(state, flags=flags)
