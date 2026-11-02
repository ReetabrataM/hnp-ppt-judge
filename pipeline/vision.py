"""Selective slide vision: only visual or text-less pages consume vision quota."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from pipeline.llm import vision_completion
from pipeline.state import add_flag, updated


def analyse_vision(state: dict[str, Any]) -> dict[str, Any]:
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "vision.txt").read_text(encoding="utf-8")
    selected = set(state.get("textless_pages", [])) | set(state.get("diagram_pages", []))
    results = []
    for page in state.get("pages", []):
        if page["page"] not in selected:
            continue
        try:
            result = vision_completion(page["image_path"], f"{prompt}\nPage number: {page['page']}")
            if not isinstance(result, dict): raise ValueError("vision output was not an object")
            result["page"] = page["page"]
            results.append(result)
        except Exception as exc:
            state = add_flag(state, "vision_failed", f"Vision failed on page {page['page']}: {exc}")
    return updated(state, vision_results=results, vision_pages=sorted(selected))


vision = analyse_vision
