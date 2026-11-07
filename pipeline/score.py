"""Rubric scoring with multi-run median, variance, and review signals."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml

import config
from pipeline.llm import text_completion, untrusted_deck_content
from pipeline.state import add_flag, updated


class ScoreValidationError(ValueError):
    pass


def load_rubric(path: Path | None = None) -> dict[str, Any]:
    rubric_path = path or config.RUBRIC_PATH
    data = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("criteria"), dict):
        raise ScoreValidationError("Rubric must define a criteria mapping.")
    weights = [float(spec.get("weight", 0)) for spec in data["criteria"].values()]
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ScoreValidationError(f"Rubric weights must sum to 1.0, got {sum(weights):.4f}.")
    return data


def _criterion_names(rubric: dict[str, Any]) -> list[str]:
    return list(rubric["criteria"].keys())


def _validate_run(raw: Any, rubric: dict[str, Any], page_count: int | None) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ScoreValidationError("Score output must be a JSON object.")
    names = _criterion_names(rubric)
    missing = [name for name in names if name not in raw]
    if missing:
        raise ScoreValidationError(f"Missing criteria: {missing}")
    low = int(rubric.get("scale_min", 1))
    high = int(rubric.get("scale_max", 5))
    normalised: dict[str, dict[str, Any]] = {}
    for name in names:
        item = raw[name]
        if not isinstance(item, dict):
            raise ScoreValidationError(f"{name} must be an object.")
        score = item.get("score")
        reason = item.get("reason", "")
        slides = item.get("slides", [])
        if not isinstance(score, (int, float)) or int(score) != score:
            raise ScoreValidationError(f"{name}.score must be an integer.")
        score = int(score)
        if score < low or score > high:
            raise ScoreValidationError(f"{name}.score must be between {low} and {high}.")
        if not isinstance(reason, str) or not reason.strip():
            raise ScoreValidationError(f"{name}.reason must be non-empty text.")
        if not isinstance(slides, list) or any(not isinstance(s, int) or s < 1 for s in slides):
            raise ScoreValidationError(f"{name}.slides must be positive page numbers.")
        if page_count is not None and any(s > page_count for s in slides):
            raise ScoreValidationError(f"{name}.slides references a page outside this deck.")
        normalised[name] = {
            "score": score,
            "reason": reason.strip(),
            "slides": sorted(set(slides)),
        }
    return normalised


def _build_prompt(rubric: dict[str, Any], state: dict[str, Any]) -> str:
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "score.txt").read_text(encoding="utf-8")
    evidence = {
        "profile": state.get("profile") or {},
        "page_text": [{"page": p["page"], "text": p.get("text", "")} for p in state.get("pages", [])],
        "vision_results": state.get("vision_results", []),
    }
    return (
        prompt
        + "\n\nRubric:\n"
        + json.dumps(rubric, ensure_ascii=False)
        + "\n\nDeck evidence:\n"
        + untrusted_deck_content(json.dumps(evidence, ensure_ascii=False))
    )


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pvariance(values))


def _aggregate_runs(runs: list[dict[str, dict[str, Any]]], rubric: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    names = _criterion_names(rubric)
    medians: dict[str, dict[str, Any]] = {}
    variances: dict[str, float] = {}
    for name in names:
        scores = [run[name]["score"] for run in runs]
        median_score = int(round(_median(scores)))
        variances[name] = round(_variance(scores), 4)
        best = min(runs, key=lambda run: (abs(run[name]["score"] - median_score), -len(run[name]["slides"])))
        medians[name] = {
            "score": median_score,
            "reason": best[name]["reason"],
            "slides": best[name]["slides"],
        }
    return medians, variances


def _weighted_total(scores: dict[str, dict[str, Any]], rubric: dict[str, Any]) -> float:
    total = 0.0
    for name, spec in rubric["criteria"].items():
        total += scores[name]["score"] * float(spec["weight"])
    return round(total, 4)


def _confidence(variances: dict[str, float], threshold: float) -> float:
    if not variances:
        return 0.0
    mean_var = sum(variances.values()) / len(variances)
    return round(max(0.0, min(1.0, 1.0 - (mean_var / threshold))), 4)


def score_deck(state: dict[str, Any], *, rubric: dict[str, Any] | None = None) -> dict[str, Any]:
    if not state.get("profile"):
        return add_flag(state, "score_skipped", "Scoring skipped because profile is missing.", severity="error")

    try:
        rubric = rubric or load_rubric()
    except Exception as exc:
        return add_flag(state, "rubric_invalid", f"Could not load rubric: {exc}", severity="error")

    attempt = int(state.get("score_attempt", 0)) + 1
    prompt = _build_prompt(rubric, state)
    runs: list[dict[str, dict[str, Any]]] = []
    errors: list[str] = []

    for index in range(config.SCORE_RUNS):
        try:
            raw = text_completion(
                prompt + f"\n\nIndependent scoring pass {index + 1} of {config.SCORE_RUNS}.",
                json_output=True,
                provider="groq",
            )
            runs.append(_validate_run(raw, rubric, state.get("page_count")))
        except Exception as exc:
            errors.append(str(exc))

    if not runs:
        return add_flag(
            updated(state, score_attempt=attempt, rescore=False),
            "score_failed",
            f"All scoring runs failed: {'; '.join(errors)}",
            severity="error",
        )

    scores, variances = _aggregate_runs(runs, rubric)
    max_variance = max(variances.values()) if variances else 0.0
    weighted = _weighted_total(scores, rubric)
    confidence = _confidence(variances, config.SCORE_VARIANCE_THRESHOLD)
    unstable = max_variance > config.SCORE_VARIANCE_THRESHOLD

    result = updated(
        state,
        scores=scores,
        score_runs=runs,
        score_variance=variances,
        score_max_variance=max_variance,
        weighted_total=weighted,
        confidence=confidence,
        score_attempt=attempt,
        rescore=False,
        needs_human_review=bool(state.get("needs_human_review", False)),
    )

    if unstable and attempt < config.SCORE_MAX_ATTEMPTS:
        return updated(result, rescore=True, needs_human_review=False)

    if unstable:
        result = updated(result, rescore=False, needs_human_review=True)
        return add_flag(
            result,
            "high_score_variance",
            f"Score variance still high after {attempt} attempts (max={max_variance}).",
            severity="warning",
        )

    return updated(result, rescore=False)


score = score_deck
