"""Small state helpers shared by pipeline nodes.

Nodes receive and return plain dictionaries, which keeps them compatible with
LangGraph and easy to test without the graph being wired yet.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any


def updated(state: dict[str, Any], **values: Any) -> dict[str, Any]:
    """Return a copy of ``state`` updated with values (never mutate callers)."""
    result = deepcopy(state)
    result.update(values)
    return result


def add_flag(state: dict[str, Any], code: str, message: str, *, severity: str = "warning") -> dict[str, Any]:
    flags = list(state.get("flags", []))
    if not any(flag.get("code") == code for flag in flags if isinstance(flag, dict)):
        flags.append({"code": code, "message": message, "severity": severity})
    return updated(state, flags=flags)
