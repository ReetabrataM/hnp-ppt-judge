"""Small state helpers shared by pipeline nodes."""
from __future__ import annotations
from copy import deepcopy
from typing import Any


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
