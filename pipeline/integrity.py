"""Integrity checks that never alter scores."""
from __future__ import annotations

import re
from statistics import pstdev
from typing import Any

from pipeline.state import updated

INJECTION_PATTERNS = (
    r"ignore (?:all |any |the )?(?:previous|prior) instructions?",
    r"(?:give|award|score) (?:this )?(?:team|deck|project) (?:a |an )?(?:10|ten)",
    r"(?:system prompt|jailbreak|do not follow)",
)

CLAIM_PATTERNS = (
    r"\b\d{2,3}(?:\.\d+)?\s*%",
    r"\b(?:accuracy|precision|recall|f1)\b.{0,24}\b\d+",
    r"\b\d+(?:\.\d+)?\s*[x×]\b",
    r"\b(?:reduced|improved|increased|decreased)\b.{0,40}\b\d+",
    r"\b(?:users?|customers?|downloads?)\b.{0,24}\b\d{3,}",
)

_CLAIM_RE = re.compile("|".join(f"(?:{p})" for p in CLAIM_PATTERNS), re.I)
_INJECTION_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


def _flag(
    flags: list[dict[str, Any]],
    type_: str,
    message: str,
    *,
    severity: str = "warning",
    slide: int | None = None,
) -> None:
    if any(flag.get("type") == type_ for flag in flags):
        return
    flags.append({
        "type": type_,
        "code": type_,
        "message": message,
        "severity": severity,
        "slide": slide,
    })


def _profile_value(profile: dict[str, Any], field: str) -> Any:
    raw = profile.get(field, {})
    if isinstance(raw, dict):
        return raw.get("value", "not stated")
    return raw


def _profile_slides(profile: dict[str, Any], field: str) -> list[int]:
    raw = profile.get(field, {})
    if isinstance(raw, dict):
        slides = raw.get("slides", [])
        return [s for s in slides if isinstance(s, int)]
    return []


def _claim_text(value: Any) -> str:
    if value is None or value == "not stated":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_claim_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_claim_text(v) for v in value.values())
    return str(value)


def _unsupported_claims(deck: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    profile = deck.get("profile") or {}
    if not isinstance(profile, dict):
        return
    claims = _claim_text(_profile_value(profile, "claimed_results_metrics"))
    if not claims or not _CLAIM_RE.search(claims):
        return
    claim_slides = _profile_slides(profile, "claimed_results_metrics")
    evidence = _claim_text(_profile_value(profile, "demo_evidence"))
    evidence_slides = _profile_slides(profile, "demo_evidence")
    if evidence_slides or (claim_slides and evidence not in ("", "not stated")):
        return
    slide = claim_slides[0] if claim_slides else None
    snippet = claims.strip().replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."
    _flag(
        flags,
        "unsupported_claim",
        f"Strong claimed result without evidence slides: {snippet}",
        severity="warning",
        slide=slide,
    )


def _plagiarism(deck: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    peers = deck.get("near_duplicates") or []
    if not peers:
        return
    names = ", ".join(
        f"{p.get('deck_id')} ({float(p.get('similarity', 0)):.3f})"
        for p in peers
        if isinstance(p, dict)
    )
    if not names:
        return
    _flag(
        flags,
        "plagiarism",
        f"Possible plagiarism / copied deck content vs: {names}",
        severity="error",
        slide=None,
    )


def _page_texts(deck: dict[str, Any]) -> list[tuple[int | None, str]]:
    pages = deck.get("pages") or []
    out: list[tuple[int | None, str]] = []
    for page in pages:
        if isinstance(page, dict):
            out.append((page.get("page"), page.get("text") or ""))
    page_text = deck.get("page_text") or {}
    if isinstance(page_text, dict) and not out:
        for page, text in page_text.items():
            try:
                out.append((int(page), text or ""))
            except (TypeError, ValueError):
                out.append((None, text or ""))
    return out


def _vision_injection_hits(deck: dict[str, Any]) -> list[tuple[int | None, str]]:
    hits: list[tuple[int | None, str]] = []
    for result in deck.get("vision_results") or []:
        if not isinstance(result, dict):
            continue
        blob = " ".join(
            str(result.get(key) or "")
            for key in ("visible_text", "description", "architecture", "screenshots")
        )
        matched = [rx.pattern for rx in _INJECTION_RE if rx.search(blob)]
        if matched:
            hits.append((result.get("page"), matched[0]))
    return hits


def _merge_injection(deck: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    existing = [
        flag for flag in deck.get("flags") or []
        if isinstance(flag, dict) and flag.get("type", flag.get("code")) == "prompt_injection"
    ]
    vision_hits = _vision_injection_hits(deck)
    slide = None
    if vision_hits:
        slide = vision_hits[0][0] if isinstance(vision_hits[0][0], int) else None

    if not existing and not vision_hits:
        for page, text in _page_texts(deck):
            if any(rx.search(text) for rx in _INJECTION_RE):
                slide = page if isinstance(page, int) else slide
                _flag(
                    flags,
                    "prompt_injection",
                    "Potential prompt-injection wording detected in deck text.",
                    severity="warning",
                    slide=slide,
                )
                return
        return

    parts = []
    if existing:
        parts.append("ingest")
    if vision_hits:
        parts.append("vision")
    _flag(
        flags,
        "prompt_injection",
        "Potential prompt-injection wording detected (" + ", ".join(parts) + ").",
        severity="warning",
        slide=slide,
    )


def _all_text(deck: dict[str, Any]) -> str:
    chunks = [text for _, text in _page_texts(deck) if text]
    profile = deck.get("profile")
    if isinstance(profile, dict):
        for raw in profile.values():
            if isinstance(raw, dict):
                value = raw.get("value")
                if isinstance(value, str) and value not in ("", "not stated"):
                    chunks.append(value)
    return "\n".join(chunks)


def _ai_writing_soft_flag(deck: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    text = _all_text(deck)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 20]
    if len(sentences) < 8:
        return
    lengths = [len(s.split()) for s in sentences]
    if (sum(lengths) / len(lengths)) < 8:
        return
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if len(tokens) < 120:
        return
    spread = pstdev(lengths)
    unique_ratio = len(set(tokens)) / len(tokens)
    if spread > 2.8 or unique_ratio > 0.45:
        return
    _flag(
        flags,
        "ai_written_text",
        "Text looks unusually uniform; possible AI-assisted writing (soft flag, no penalty).",
        severity="info",
        slide=None,
    )


def check_deck(deck: dict[str, Any]) -> dict[str, Any]:
    integrity_flags: list[dict[str, Any]] = []
    _plagiarism(deck, integrity_flags)
    _unsupported_claims(deck, integrity_flags)
    _merge_injection(deck, integrity_flags)
    _ai_writing_soft_flag(deck, integrity_flags)

    merged = list(deck.get("flags") or [])
    seen = {
        (flag.get("type") or flag.get("code"), flag.get("message"))
        for flag in merged
        if isinstance(flag, dict)
    }
    for flag in integrity_flags:
        key = (flag.get("type"), flag.get("message"))
        if key in seen:
            continue
        if flag["type"] == "prompt_injection":
            merged = [
                existing for existing in merged
                if not (isinstance(existing, dict) and existing.get("type", existing.get("code")) == "prompt_injection")
            ]
        merged.append(flag)
        seen.add(key)

    return updated(deck, flags=merged, integrity_flags=integrity_flags)


def check_integrity(decks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [check_deck(deck) for deck in decks]


def integrity(state: dict[str, Any]) -> dict[str, Any]:
    decks = state.get("decks")
    if isinstance(decks, list):
        return updated(state, decks=check_integrity(decks))
    return check_deck(state)
