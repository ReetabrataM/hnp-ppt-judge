"""Cross-team embeddings, clustering, novelty, duplicates, and top-tier pairwise checks."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from pipeline.llm import embeddings, text_completion, untrusted_deck_content
from pipeline.state import add_flag, updated

logger = logging.getLogger(__name__)

CLUSTER_DISTANCE = 0.40
DUPLICATE_SIMILARITY = 0.92
NOVELTY_NEIGHBOURS = 3
MAX_PAIRWISE_CALLS = 12
TOP_TIER_SIZE = 10


def _deck_id(deck: dict[str, Any]) -> str:
    return str(deck.get("deck_id") or deck.get("team") or deck.get("pdf_path") or "unknown")


def _field_text(value: Any) -> str:
    if value is None or value == "not stated":
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(_field_text(item) for item in value if _field_text(item))
    if isinstance(value, dict):
        if "value" in value:
            return _field_text(value.get("value"))
        return ", ".join(f"{key}: {_field_text(val)}" for key, val in value.items() if _field_text(val))
    return str(value).strip()


def profile_to_text(profile: dict[str, Any] | None) -> str:
    if not isinstance(profile, dict):
        return ""
    parts = []
    for key, raw in profile.items():
        text = _field_text(raw)
        if text:
            parts.append(f"{key}: {text}")
    return "\n".join(parts)


def _total_score(deck: dict[str, Any]) -> float | None:
    if deck.get("weighted_total") is not None:
        return float(deck["weighted_total"])
    scores = deck.get("scores")
    if isinstance(scores, dict):
        if scores.get("total") is not None:
            return float(scores["total"])
        if scores.get("weighted_total") is not None:
            return float(scores["weighted_total"])
        numeric = [float(v) for v in scores.values() if isinstance(v, (int, float))]
        if numeric:
            return float(sum(numeric))
    return None


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _similarity_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = _l2_normalise(np.asarray(vectors, dtype=float))
    return np.clip(matrix @ matrix.T, -1.0, 1.0)


def _cluster_labels(similarity: np.ndarray) -> np.ndarray:
    n = similarity.shape[0]
    if n == 1:
        return np.array([0], dtype=int)
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=CLUSTER_DISTANCE,
    )
    return model.fit_predict(distance)


def _novelty_scores(similarity: np.ndarray, labels: np.ndarray) -> list[float]:
    n = similarity.shape[0]
    if n == 1:
        return [1.0]
    scores: list[float] = []
    for i in range(n):
        same = [j for j in range(n) if j != i and labels[j] == labels[i]]
        pool = same if same else [j for j in range(n) if j != i]
        k = min(NOVELTY_NEIGHBOURS, len(pool))
        nearest = sorted((float(similarity[i, j]) for j in pool), reverse=True)[:k]
        scores.append(float(np.mean([1.0 - sim for sim in nearest])))
    return scores


def _duplicate_pairs(decks: list[dict[str, Any]], similarity: np.ndarray) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    n = len(decks)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity[i, j])
            if sim >= DUPLICATE_SIMILARITY:
                pairs.append((_deck_id(decks[i]), _deck_id(decks[j]), sim))
    return pairs


def _attach_duplicates(
    decks: list[dict[str, Any]],
    pairs: list[tuple[str, str, float]],
) -> list[dict[str, Any]]:
    by_id = {_deck_id(deck): [] for deck in decks}
    for left, right, sim in pairs:
        by_id.setdefault(left, []).append({"deck_id": right, "similarity": round(sim, 4)})
        by_id.setdefault(right, []).append({"deck_id": left, "similarity": round(sim, 4)})

    updated_decks = []
    for deck in decks:
        deck_id = _deck_id(deck)
        peers = sorted(by_id.get(deck_id, []), key=lambda item: item["similarity"], reverse=True)
        next_deck = updated(deck, near_duplicates=peers)
        if peers:
            names = ", ".join(f"{p['deck_id']} ({p['similarity']:.3f})" for p in peers)
            next_deck = add_flag(
                next_deck,
                "near_duplicate",
                f"Near-duplicate of: {names}",
                severity="warning",
                slide=None,
            )
        updated_decks.append(next_deck)
    return updated_decks


def _profile_snapshot(deck: dict[str, Any]) -> dict[str, Any]:
    profile = deck.get("profile") or {}
    compact = {}
    for key, raw in profile.items() if isinstance(profile, dict) else []:
        compact[key] = _field_text(raw) or "not stated"
    return {
        "deck_id": _deck_id(deck),
        "weighted_total": _total_score(deck),
        "profile": compact,
    }


def _pairwise_top_tier(decks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        (deck for deck in decks if _total_score(deck) is not None),
        key=lambda deck: _total_score(deck) or 0.0,
        reverse=True,
    )[:TOP_TIER_SIZE]

    working = {_deck_id(deck): deck for deck in decks}
    notes_by_id: dict[str, list[dict[str, Any]]] = {deck_id: [] for deck_id in working}

    if len(ranked) < 2:
        return [updated(deck, pairwise=[]) for deck in decks]

    pairwise_prompt = (Path(__file__).resolve().parents[1] / "prompts" / "pairwise.txt").read_text(encoding="utf-8")
    calls = 0
    for index in range(len(ranked) - 1):
        if calls >= MAX_PAIRWISE_CALLS:
            logger.info("Pairwise call budget reached (%s); stopping.", MAX_PAIRWISE_CALLS)
            break
        a, b = ranked[index], ranked[index + 1]
        id_a, id_b = _deck_id(a), _deck_id(b)
        score_a, score_b = _total_score(a), _total_score(b)
        payload = {
            "deck_A": _profile_snapshot(a),
            "deck_B": _profile_snapshot(b),
            "rubric_order": "A_higher" if (score_a or 0) >= (score_b or 0) else "B_higher",
        }
        prompt = (
            pairwise_prompt
            + "\n\nComparison input:\n"
            + untrusted_deck_content(json.dumps(payload, ensure_ascii=False))
        )
        try:
            result = text_completion(prompt, json_output=True, provider="groq")
            if not isinstance(result, dict):
                raise ValueError("pairwise output was not an object")
            preferred = result.get("preferred")
            notes_by_id[id_a].append({
                "against": id_b,
                "preferred": preferred,
                "reason": result.get("reason"),
                "agrees_with_rubric_order": result.get("agrees_with_rubric_order"),
                "rubric_totals": {"A": score_a, "B": score_b},
            })
            notes_by_id[id_b].append({
                "against": id_a,
                "preferred": {"A": "B", "B": "A"}.get(str(preferred), preferred),
                "reason": result.get("reason"),
                "agrees_with_rubric_order": result.get("agrees_with_rubric_order"),
                "rubric_totals": {"A": score_b, "B": score_a},
            })
            if result.get("agrees_with_rubric_order") is False:
                working[id_a] = add_flag(
                    working[id_a],
                    "pairwise_disagreement",
                    f"Pairwise check disagreed with rubric order vs {id_b}: {result.get('reason')}",
                    severity="info",
                )
        except Exception as exc:
            logger.warning("Pairwise comparison failed for %s vs %s: %s", id_a, id_b, exc)
        calls += 1

    return [
        updated(working[_deck_id(deck)], pairwise=notes_by_id.get(_deck_id(deck), []))
        for deck in decks
    ]


def compare_decks(decks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not decks:
        return []

    texts = [profile_to_text(deck.get("profile")) or f"empty profile {_deck_id(deck)}" for deck in decks]
    try:
        vectors = embeddings(texts)
    except Exception as exc:
        logger.error("Embedding step failed: %s", exc)
        return [
            add_flag(updated(deck, cluster_id=0, novelty=None, near_duplicates=[], pairwise=[]),
                     "compare_failed", f"Embedding/clustering failed: {exc}", severity="error")
            for deck in decks
        ]

    if len(vectors) != len(decks):
        raise RuntimeError("Embedding count does not match deck count.")

    similarity = _similarity_matrix(vectors)
    labels = _cluster_labels(similarity)
    novelty = _novelty_scores(similarity, labels)
    pairs = _duplicate_pairs(decks, similarity)

    annotated = []
    for deck, label, signal in zip(decks, labels, novelty):
        annotated.append(updated(
            deck,
            cluster_id=int(label),
            novelty=round(float(signal), 4),
            near_duplicates=[],
            pairwise=[],
        ))

    annotated = _attach_duplicates(annotated, pairs)
    annotated = _pairwise_top_tier(annotated)
    return annotated


def compare(state: dict[str, Any]) -> dict[str, Any]:
    decks = state.get("decks")
    if not isinstance(decks, list):
        raise ValueError("compare expects state['decks'] to be a list of deck states.")
    return updated(state, decks=compare_decks(decks))
