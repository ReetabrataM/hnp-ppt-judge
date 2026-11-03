"""Text and high-resolution page-image extraction."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import fitz
import config
from pipeline.state import add_flag, updated


def extract_deck(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("ingest_ok"):
        return add_flag(state, "extraction_skipped", "Extraction skipped because ingest failed.", severity="error")
    path = Path(state.get("pdf_path") or state.get("deck_path") or state["path"])
    deck_id = str(state.get("deck_id", path.stem))
    output_dir = config.PAGES_DIR / deck_id
    output_dir.mkdir(parents=True, exist_ok=True)
    pages, textless, diagram_pages = [], [], []
    try:
        with fitz.open(path) as doc:
            matrix = fitz.Matrix(config.RENDER_DPI / 72, config.RENDER_DPI / 72)
            for number, page in enumerate(doc, 1):
                text = page.get_text("text").strip()
                image_path = output_dir / f"page_{number:03d}.png"
                page.get_pixmap(matrix=matrix, alpha=False).save(image_path)
                drawings = page.get_drawings()
                has_visuals = bool(page.get_images(full=True) or drawings)
                is_textless = len(text) < config.TEXTLESS_CHARACTER_LIMIT
                if is_textless: textless.append(number)
                if has_visuals: diagram_pages.append(number)
                pages.append({"page": number, "text": text, "image_path": str(image_path), "text_less": is_textless, "has_visuals": has_visuals})
    except (fitz.FileDataError, RuntimeError, OSError) as exc:
        return add_flag(state, "extraction_failed", f"Could not extract PDF: {exc}", severity="error")
    return updated(state, pages=pages, page_text={item["page"]: item["text"] for item in pages}, textless_pages=textless, diagram_pages=diagram_pages, pages_dir=str(output_dir))


extract = extract_deck
