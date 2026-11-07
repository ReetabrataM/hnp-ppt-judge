"""Create and validate the common, evidence-linked team profile."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pipeline.llm import text_completion, untrusted_deck_content
from pipeline.state import add_flag, updated

PROFILE_FIELDS = (
    "problem", "target_users", "solution", "tech_stack", "architecture",
    "claimed_results_metrics", "demo_evidence", "business_model", "team", "domain_track",
)
PROFILE_SCHEMA = {field: {"value": "not stated", "slides": []} for field in PROFILE_FIELDS}


class ProfileValidationError(ValueError): pass


def validate_profile(profile: Any, page_count: int | None = None) -> dict[str, dict[str, Any]]:
    if not isinstance(profile, dict):
        raise ProfileValidationError("Profile must be a JSON object.")
    extra = set(profile) - set(PROFILE_FIELDS)
    if extra:
        raise ProfileValidationError(f"Profile has unsupported fields: {sorted(extra)}")
    normalised: dict[str, dict[str, Any]] = {}
    for field in PROFILE_FIELDS:
        raw = profile.get(field, {"value": "not stated", "slides": []})
        if not isinstance(raw, dict) or set(raw) - {"value", "slides"}:
            raise ProfileValidationError(f"{field} must contain only value and slides.")
        value = raw.get("value", "not stated")
        slides = raw.get("slides", [])
        if not isinstance(value, (str, list, dict)):
            raise ProfileValidationError(f"{field}.value must be text, list, or object.")
        if not isinstance(slides, list) or any(not isinstance(slide, int) or slide < 1 for slide in slides):
            raise ProfileValidationError(f"{field}.slides must be positive page numbers.")
        if page_count is not None and any(slide > page_count for slide in slides):
            raise ProfileValidationError(f"{field}.slides references a page outside this deck.")
        if value in ("", [], {}): value = "not stated"
        normalised[field] = {"value": value, "slides": sorted(set(slides))}
    return normalised


def build_profile(state: dict[str, Any]) -> dict[str, Any]:
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "profile.txt").read_text(encoding="utf-8")
    source = {
        "page_text": [{"page": p["page"], "text": p.get("text", "")} for p in state.get("pages", [])],
        "vision_results": state.get("vision_results", []),
    }
    request = (prompt + "\nRequired schema:\n" + json.dumps(PROFILE_SCHEMA) + "\nDeck evidence follows:\n" +
               untrusted_deck_content(json.dumps(source, ensure_ascii=False)))
    try:
        # Profile extraction is text-only. Use Groq directly so a temporary
        # Gemini vision/capacity outage does not block the whole deck batch.
        profile = validate_profile(text_completion(request, json_output=True, provider="groq"), state.get("page_count"))
    except Exception as exc:
        return add_flag(state, "profile_failed", f"Profile extraction/validation failed: {exc}", severity="error")
    return updated(state, profile=profile, profile_valid=True)


profile = build_profile
