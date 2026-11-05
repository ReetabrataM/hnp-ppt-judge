"""LangGraph orchestration; provider calls remain inside pipeline.llm."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import config
from langgraph.graph import END, START, StateGraph

from pipeline.compare import compare_decks
from pipeline.export import export_results
from pipeline.extract import extract_deck
from pipeline.ingest import ingest_deck
from pipeline.integrity import check_integrity
from pipeline.profile import build_profile
from pipeline.state import BatchState, DeckState, add_flag, updated
from pipeline.vision import analyse_vision

logger = logging.getLogger(__name__)


def _process_each(state: BatchState, action: Callable[[dict[str, Any]], dict[str, Any]], stage: str) -> BatchState:
    decks: list[dict[str, Any]] = []
    for deck in state.get("decks", []):
        # A failed ingest is terminal for that deck, but never terminal for the batch.
        if stage != "ingest" and not deck.get("ingest_ok"):
            decks.append(deck)
            continue
        try:
            decks.append(action(deck))
        except Exception as exc:  # A bad deck must not stop the other 199 decks.
            logger.exception("%s failed for %s", stage, deck.get("deck_id", deck.get("pdf_path")))
            decks.append(add_flag(deck, f"{stage}_failed", f"{stage} failed: {exc}", severity="error"))
    return updated(state, decks=decks)


def ingest_node(state: BatchState) -> BatchState:
    return _process_each(state, ingest_deck, "ingest")


def extract_node(state: BatchState) -> BatchState:
    return _process_each(state, extract_deck, "extract")


def vision_node(state: BatchState) -> BatchState:
    return _process_each(state, analyse_vision, "vision")


def profile_node(state: BatchState) -> BatchState:
    return _process_each(state, build_profile, "profile")


def _score_one(deck: dict[str, Any]) -> dict[str, Any]:
    """Use issue #4 when present; retain a transparent safe state until then."""
    try:
        from pipeline import score as score_module
        scorer = getattr(score_module, "score_deck", None) or getattr(score_module, "score", None)
        if not callable(scorer):
            raise NotImplementedError("pipeline/score.py has not been implemented (issue #4).")
        return scorer(deck)
    except NotImplementedError as exc:
        return updated(add_flag(deck, "scoring_unavailable", str(exc), severity="info"),
                       stage="scoring_unavailable", needs_human_review=True)


def score_node(state: BatchState) -> BatchState:
    return _process_each(state, _score_one, "score")


def _needs_score_retry(state: BatchState) -> str:
    for deck in state.get("decks", []):
        variance = deck.get("score_variance")
        attempts = int(deck.get("score_attempts", 0))
        if isinstance(variance, (int, float)) and variance > config.SCORE_VARIANCE_THRESHOLD:
            if attempts < config.MAX_SCORE_RETRIES:
                return "score"
    return "compare"


def _mark_score_retry(state: BatchState) -> BatchState:
    decks: list[dict[str, Any]] = []
    for deck in state.get("decks", []):
        variance = deck.get("score_variance")
        attempts = int(deck.get("score_attempts", 0))
        if isinstance(variance, (int, float)) and variance > config.SCORE_VARIANCE_THRESHOLD:
            attempts += 1
            deck = updated(deck, score_attempts=attempts)
            if attempts >= config.MAX_SCORE_RETRIES:
                deck = add_flag(deck, "high_score_variance", "Score variance remained high after retries; human review required.", severity="warning")
                deck = updated(deck, needs_human_review=True)
        decks.append(deck)
    return updated(state, decks=decks)


def compare_node(state: BatchState) -> BatchState:
    try:
        return updated(state, decks=compare_decks(state.get("decks", [])))
    except Exception as exc:
        logger.exception("Batch comparison failed")
        return updated(state, batch_errors=[*state.get("batch_errors", []), f"compare failed: {exc}"])


def integrity_node(state: BatchState) -> BatchState:
    return updated(state, decks=check_integrity(state.get("decks", [])))


def export_node(state: BatchState) -> BatchState:
    try:
        paths = export_results(state.get("decks", []), output_dir=Path(state.get("output_dir", config.OUTPUT_DIR)))
        return updated(state, export_paths={name: str(path) for name, path in paths.items()})
    except Exception as exc:
        logger.exception("Export failed")
        return updated(state, batch_errors=[*state.get("batch_errors", []), f"export failed: {exc}"])


def _needs_vision(state: BatchState) -> str:
    for deck in state.get("decks", []):
        if deck.get("ingest_ok") and (deck.get("textless_pages") or deck.get("diagram_pages")):
            return "vision"
    return "profile"


def build_graph(*, checkpointer: Any | None = None):
    """Build the checkpointable batch graph used by ``main.py``."""
    graph = StateGraph(BatchState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("extract", extract_node)
    graph.add_node("vision", vision_node)
    graph.add_node("profile", profile_node)
    graph.add_node("score", score_node)
    graph.add_node("score_retry_guard", _mark_score_retry)
    graph.add_node("compare", compare_node)
    graph.add_node("integrity", integrity_node)
    graph.add_node("export", export_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "extract")
    graph.add_conditional_edges("extract", _needs_vision, {"vision": "vision", "profile": "profile"})
    graph.add_edge("vision", "profile")
    graph.add_edge("profile", "score")
    graph.add_edge("score", "score_retry_guard")
    graph.add_conditional_edges("score_retry_guard", _needs_score_retry, {"score": "score", "compare": "compare"})
    graph.add_edge("compare", "integrity")
    graph.add_edge("integrity", "export")
    graph.add_edge("export", END)
    return graph.compile(checkpointer=checkpointer)
