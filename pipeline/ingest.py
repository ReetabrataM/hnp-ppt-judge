"""Pre-flight PDF validation. This stage never calls an LLM."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import fitz
import config
from pipeline.state import add_flag, updated

INJECTION_PATTERNS = (
    r"ignore (?:all |any |the )?(?:previous|prior) instructions?",
    r"(?:give|award|score) (?:this )?(?:team|deck|project) (?:a |an )?(?:10|ten)",
    r"(?:system prompt|jailbreak|do not follow)",
)


def _path(state: dict[str, Any]) -> Path:
    value = state.get("pdf_path") or state.get("deck_path") or state.get("path")
    if not value:
        raise ValueError("State needs pdf_path (or deck_path/path).")
    return Path(value)


def _hidden_text(doc: fitz.Document) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for page_no, page in enumerate(doc, 1):
        rect = page.rect
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    bbox = fitz.Rect(span["bbox"])
                    colour = span.get("color", 0)
                    rgb = ((colour >> 16) & 255, (colour >> 8) & 255, colour & 255)
                    reasons = []
                    if min(rgb) > 240: reasons.append("near-white text")
                    if span.get("size", 999) < config.MIN_VISIBLE_FONT_SIZE: reasons.append("tiny font")
                    if not rect.intersects(bbox) or bbox.width <= 0 or bbox.height <= 0: reasons.append("outside page area")
                    if reasons:
                        findings.append({"page": page_no, "text": text[:200], "reasons": reasons})
    return findings


def ingest_deck(state: dict[str, Any]) -> dict[str, Any]:
    path = _path(state)
    result = updated(state, deck_id=state.get("deck_id", path.stem), ingest_ok=False)
    if path.suffix.lower() != ".pdf":
        return add_flag(result, "not_pdf", "Input must be a PDF.", severity="error")
    try:
        size = path.stat().st_size
    except OSError as exc:
        return add_flag(result, "unreadable_file", f"Cannot read PDF: {exc}", severity="error")
    if size > config.MAX_PDF_BYTES:
        return add_flag(result, "file_too_large", f"PDF is {size} bytes; limit is {config.MAX_PDF_BYTES}.", severity="error")
    try:
        with fitz.open(path) as doc:
            if doc.needs_pass:
                return add_flag(result, "password_protected", "PDF is password-protected.", severity="error")
            pages = doc.page_count
            if pages == 0:
                return add_flag(result, "empty_pdf", "PDF contains no pages.", severity="error")
            if pages > config.MAX_PDF_PAGES:
                return add_flag(result, "too_many_pages", f"PDF has {pages} pages; limit is {config.MAX_PDF_PAGES}.", severity="error")
            result = updated(result, page_count=pages, pdf_size_bytes=size)
            hidden = _hidden_text(doc)
            if hidden:
                result = add_flag(result, "hidden_text", "Potentially hidden text detected.")
                result = updated(result, hidden_text=hidden)
            visible = "\n".join(page.get_text() for page in doc)
            phrases = [pattern for pattern in INJECTION_PATTERNS if re.search(pattern, visible, re.I)]
            if phrases:
                result = add_flag(result, "prompt_injection", "Potential prompt-injection wording detected.")
                result = updated(result, injection_matches=phrases)
    except (fitz.FileDataError, RuntimeError, OSError) as exc:
        return add_flag(result, "corrupt_pdf", f"Unable to open PDF: {exc}", severity="error")
    return updated(result, ingest_ok=True)


ingest = ingest_deck
