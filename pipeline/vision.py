"""Selective slide vision: only visual or text-less pages consume vision quota."""
from __future__ import annotations
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any
import config
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
            result = _vision_with_timeout(page["image_path"], f"{prompt}\nPage number: {page['page']}")
            if not isinstance(result, dict): raise ValueError("vision output was not an object")
            result["page"] = page["page"]
            results.append(result)
        except Exception as exc:
            state = add_flag(state, "vision_failed", f"Vision failed on page {page['page']}: {exc}")
            # A provider outage affects the remaining pages too. Do not spend
            # the batch's entire quota/retry budget repeating the same failed
            # request; profile extraction can continue from page text.
            break
    return updated(state, vision_results=results, vision_pages=sorted(selected))


vision = analyse_vision


def _vision_with_timeout(image_path: str, prompt: str) -> Any:
    """Bound a provider call so one unavailable vision endpoint cannot stall a batch."""
    result: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def call() -> None:
        try:
            result.put((True, vision_completion(image_path, prompt)))
        except Exception as exc:
            result.put((False, exc))

    worker = Thread(target=call, daemon=True)
    worker.start()
    worker.join(config.VISION_TIMEOUT_SECONDS)
    if worker.is_alive():
        raise TimeoutError(f"Vision request timed out after {config.VISION_TIMEOUT_SECONDS:g} seconds")
    successful, value = result.get_nowait()
    if not successful:
        raise value
    return value
