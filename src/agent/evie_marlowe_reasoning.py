"""
Reasoning subgraph for Evie Marlowe.

Two nodes per turn:
  1. detect_conditions  — LLM classifies the player's intent into boolean flags
  2. apply_transitions  — pure Python updates memory and advances emotional state
"""

from __future__ import annotations

import os
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import StateGraph, START
from pydantic import BaseModel
from typing_extensions import TypedDict

# ── LLM ──────────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
    model=os.getenv("LMSTUDIO_MODEL", "gemma-4-e4b"),
    temperature=0,
)


# ── Condition detection model ─────────────────────────────────────────────────

class PlayerInputConditions(BaseModel):
    mentions_empathy: bool
    offers_protection: bool
    shows_blackmail_evidence: bool
    shows_locket: bool
    mentions_brennan: bool
    proves_alley_presence: bool
    accuses_evie: bool
    mentions_murder_weapon_points_to_brennan: bool
    threatens_arrest: bool
    presents_concrete_evidence: bool
    asks_about_victor_relationship: bool
    mocks_or_dismisses_evie: bool
    repeated_accusation_without_evidence: bool
    harsh_accusation: bool
    threatens_or_mocks: bool
    suggests_public_brennan_confrontation: bool
    betrays_or_threatens_evie: bool
    stops_accusing_and_asks_facts: bool
    apologizes_or_softens: bool


_condition_llm = _llm.with_structured_output(PlayerInputConditions)

_DETECTION_SYSTEM = SystemMessage("""
You are a condition-detection module for a detective mystery game.
The player is interrogating Evie Marlowe, a frightened witness to a murder.

Game context:
- Victor = the murder victim (Evie's former lover)
- Brennan = the dangerous killer Evie fears
- Blackmail photos = photographs that prove Brennan's guilt
- The locket = a piece of evidence linking Evie to the crime scene
- The alley = the location of Victor's murder

Evaluate the player's message and set each condition to true ONLY if it is
clearly and directly present in what the player said.
""")

type EvieMilestone = Literal[
    "empathy",
    "protection_offer",
    "blackmail_proof",
    "locket_evidence",
    "brennan_connection",
    "alley_presence",
    "murder_weapon",
    "direct_accusation"
]

# ── Subgraph state ────────────────────────────────────────────────────────────

class EvieReasoningState(TypedDict):
    # Input
    player_input: str
    # Evie's current emotional/behavioural node
    current_node: str
    # Numeric counters
    trust_level: int
    fear_level: int
    guilt_pressure: int
    suspicion_level: int
    # Milestone set (stored as list — TypedDict cannot hold sets)
    milestones: List[EvieMilestone]
    # Boolean memory flags
    player_has_offered_protection: bool
    player_has_found_blackmail_photos: bool
    player_has_shown_locket: bool
    player_has_mentioned_brennan: bool
    player_knows_evie_was_at_alley: bool
    player_has_accused_evie: bool
    player_has_revealed_murder_weapon: bool
    critical_info_revealed: bool
    final_clue_revealed: bool
    # Intermediate: populated by detect_conditions, consumed by apply_transitions
    detected_conditions: dict


# ── Milestone helpers ─────────────────────────────────────────────────────────

def _has_any(milestones: List[str], required: List[str]) -> bool:
    return any(m in milestones for m in required)


def _has_all(milestones: List[str], required: List[str]) -> bool:
    return all(m in milestones for m in required)


def _count_any(milestones: List[str], required: List[str]) -> int:
    return sum(1 for m in required if m in milestones)


# ── Node 1: condition detection (LLM) ────────────────────────────────────────

async def detect_conditions(state: EvieReasoningState) -> dict:
    print(state)
    conditions: PlayerInputConditions = await _condition_llm.ainvoke([
        _DETECTION_SYSTEM,
        HumanMessage(state["player_input"]),
    ])
    return {"detected_conditions": conditions.model_dump()}


# ── Node 2: memory update + state transition (pure Python) ───────────────────

def apply_transitions(state: EvieReasoningState) -> dict:
    print(state)
    c = state["detected_conditions"]
    milestones = state["milestones"]

    trust_level = state.get("trust_level")
    fear_level = state.get("fear_level")
    guilt_pressure = state.get("guilt_pressure")
    suspicion_level = state.get("suspicion_level")
    player_has_offered_protection = state.get("player_has_offered_protection")
    player_has_found_blackmail_photos = state.get("player_has_found_blackmail_photos")
    player_has_shown_locket = state.get("player_has_shown_locket")
    player_has_mentioned_brennan = state.get("player_has_mentioned_brennan")
    player_knows_evie_was_at_alley = state.get("player_knows_evie_was_at_alley")
    player_has_accused_evie = state.get("player_has_accused_evie")
    player_has_revealed_murder_weapon = state.get("player_has_revealed_murder_weapon")
    critical_info_revealed = state.get("critical_info_revealed")
    final_clue_revealed = state.get("final_clue_revealed")

    def add_milestone(m: str) -> None:
        if m not in milestones:
            milestones.append(m)

    # ── Update memory from detected conditions ────────────────────────────────

    if c["mentions_empathy"]:
        trust_level += 1
        add_milestone("empathy")

    if c["offers_protection"]:
        trust_level += 2
        fear_level -= 1
        player_has_offered_protection = True
        add_milestone("protection_offer")

    if c["shows_blackmail_evidence"]:
        player_has_found_blackmail_photos = True
        guilt_pressure += 2
        add_milestone("blackmail_proof")

    if c["shows_locket"]:
        player_has_shown_locket = True
        guilt_pressure += 2
        add_milestone("locket_evidence")

    if c["mentions_brennan"]:
        player_has_mentioned_brennan = True
        fear_level += 2
        add_milestone("brennan_connection")

    if c["proves_alley_presence"]:
        player_knows_evie_was_at_alley = True
        guilt_pressure += 2
        add_milestone("alley_presence")

    if c["accuses_evie"]:
        player_has_accused_evie = True
        suspicion_level += 2
        add_milestone("direct_accusation")

    if c["mentions_murder_weapon_points_to_brennan"]:
        player_has_revealed_murder_weapon = True
        add_milestone("murder_weapon")

    # ── Compute readiness thresholds ──────────────────────────────────────────

    guarded_ready = _has_any(milestones, [
        "empathy", "protection_offer", "blackmail_proof", "locket_evidence", "alley_presence",
    ])
    critical_ready = (
            _count_any(milestones, [
                "protection_offer", "blackmail_proof", "locket_evidence",
                "brennan_connection", "alley_presence", "murder_weapon",
            ]) >= 2
            or _has_all(milestones, ["direct_accusation", "blackmail_proof", "alley_presence"])
    )
    final_ready = (
                          critical_info_revealed and player_has_offered_protection
                  ) or (
                          critical_info_revealed and player_has_mentioned_brennan and player_has_found_blackmail_photos
                  )

    # ── State transition ──────────────────────────────────────────────────────

    next_node = _transition(state.get("current_node") or "POLITE_MASK", c, guarded_ready, critical_ready)

    if next_node in ("BROKEN_TRUSTING", "PROTECTED_WITNESS", "CORNERED_CONFESSION"):
        critical_info_revealed = True
    if next_node == "PROTECTED_WITNESS" and final_ready:
        final_clue_revealed = True

    return {
        "current_node": next_node,
        "trust_level": trust_level,
        "fear_level": fear_level,
        "guilt_pressure": guilt_pressure,
        "suspicion_level": suspicion_level,
        "milestones": milestones,
        "player_has_offered_protection": player_has_offered_protection,
        "player_has_found_blackmail_photos": player_has_found_blackmail_photos,
        "player_has_shown_locket": player_has_shown_locket,
        "player_has_mentioned_brennan": player_has_mentioned_brennan,
        "player_knows_evie_was_at_alley": player_knows_evie_was_at_alley,
        "player_has_accused_evie": player_has_accused_evie,
        "player_has_revealed_murder_weapon": player_has_revealed_murder_weapon,
        "critical_info_revealed": critical_info_revealed,
        "final_clue_revealed": final_clue_revealed,
    }


def _transition(current: str, c: dict, guarded_ready: bool, critical_ready: bool) -> str:
    if current == "POLITE_MASK":
        if c["accuses_evie"]:                  return "DEFENSIVE_DENIAL"
        if c["threatens_arrest"]:              return "PANICKED_RESISTANCE"
        if c["presents_concrete_evidence"]:    return "EVIDENCE_PRESSURE"
        if c["asks_about_victor_relationship"]: return "BITTER_REMEMBERING"
        if c["mentions_empathy"]:              return "WARY_SOFTENING"
        return "POLITE_MASK"

    if current == "WARY_SOFTENING":
        if c["mentions_brennan"]:  return "FEAR_SPIKE"
        if c["accuses_evie"]:      return "DEFENSIVE_DENIAL"
        if guarded_ready:          return "GUARDED_DISCLOSURE"
        return "POLITE_MASK"

    if current == "BITTER_REMEMBERING":
        if c["mocks_or_dismisses_evie"]: return "COLD_SHUTDOWN"
        if guarded_ready:                return "GUARDED_DISCLOSURE"
        return "WARY_SOFTENING"

    if current == "DEFENSIVE_DENIAL":
        if critical_ready:                            return "CORNERED_CONFESSION"
        if c["presents_concrete_evidence"]:           return "EVIDENCE_PRESSURE"
        if c["offers_protection"]:                    return "WARY_SOFTENING"
        if c["repeated_accusation_without_evidence"]: return "COLD_SHUTDOWN"
        return "DEFENSIVE_DENIAL"

    if current == "EVIDENCE_PRESSURE":
        if c["mentions_brennan"]:               return "FEAR_SPIKE"
        if critical_ready and c["accuses_evie"]: return "CORNERED_CONFESSION"
        if critical_ready:                      return "BROKEN_TRUSTING"
        if guarded_ready:                       return "GUARDED_DISCLOSURE"
        return "EVIDENCE_PRESSURE"

    if current == "GUARDED_DISCLOSURE":
        if critical_ready:          return "BROKEN_TRUSTING"
        if c["mentions_brennan"]:   return "FEAR_SPIKE"
        if c["threatens_arrest"]:   return "PANICKED_RESISTANCE"
        if c["harsh_accusation"]:   return "DEFENSIVE_DENIAL"
        return "GUARDED_DISCLOSURE"

    if current == "FEAR_SPIKE":
        if c["offers_protection"]:  return "PROTECTED_WITNESS"
        if critical_ready:          return "BROKEN_TRUSTING"
        if c["threatens_or_mocks"]: return "COLD_SHUTDOWN"
        return "GUARDED_DISCLOSURE"

    if current == "BROKEN_TRUSTING":
        if c["offers_protection"]: return "PROTECTED_WITNESS"
        if c["accuses_evie"]:      return "CORNERED_CONFESSION"
        return "BROKEN_TRUSTING"

    if current == "PROTECTED_WITNESS":
        if c["suggests_public_brennan_confrontation"]: return "FEAR_SPIKE"
        if c["betrays_or_threatens_evie"]:             return "COLD_SHUTDOWN"
        return "PROTECTED_WITNESS"

    if current == "CORNERED_CONFESSION":
        if c["offers_protection"]:          return "PROTECTED_WITNESS"
        if c["stops_accusing_and_asks_facts"]: return "BROKEN_TRUSTING"
        return "CORNERED_CONFESSION"

    if current == "COLD_SHUTDOWN":
        if c["apologizes_or_softens"]:       return "WARY_SOFTENING"
        if c["presents_concrete_evidence"]:  return "EVIDENCE_PRESSURE"
        return "COLD_SHUTDOWN"

    if current == "PANICKED_RESISTANCE":
        if c["offers_protection"]: return "FEAR_SPIKE"
        if critical_ready:         return "CORNERED_CONFESSION"
        if guarded_ready:          return "GUARDED_DISCLOSURE"
        return "PANICKED_RESISTANCE"

    return current  # unknown state: stay put


# ── Subgraph assembly ─────────────────────────────────────────────────────────

evie_reasoning_subgraph = (
    StateGraph(EvieReasoningState)
    .add_node("detect_conditions", detect_conditions)
    .add_node("apply_transitions", apply_transitions)
    .add_edge(START, "detect_conditions")
    .add_edge("detect_conditions", "apply_transitions")
    .compile()
)
