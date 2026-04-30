"""
Shared nodes and types for NPCs.
"""
from __future__ import annotations

from typing import Literal, List

from pydantic import BaseModel


class NPCResponse(BaseModel):
    message: str
    emotion: Literal["relaxed", "nervous", "panicking", "angry", "upset", "depressed", "defensive"]
    topic: str
    summary: str
    knowledge: List[str] | None = None
