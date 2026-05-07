import { beforeEach, describe, expect, test, vi } from "vitest";

import type { EvieMilestone, EvieNode } from "../src/agent/evie_marlowe_reasoning.js";

// Capture mock before module load via vi.hoisted so it's available in the factory
const mockConditionInvoke = vi.hoisted(() => vi.fn());

vi.mock("@langchain/openai", () => ({
  ChatOpenAI: vi.fn().mockImplementation(() => ({
    withStructuredOutput: vi.fn().mockReturnValue({
      invoke: mockConditionInvoke,
    }),
  })),
}));

import { evieReasoningSubgraph } from "../src/agent/evie_marlowe_reasoning.js";

// ─── helpers ────────────────────────────────────────────────────────────────

const allFalse = {
  mentions_empathy: false,
  offers_protection: false,
  shows_blackmail_evidence: false,
  shows_locket: false,
  mentions_brennan: false,
  proves_alley_presence: false,
  accuses_evie: false,
  mentions_murder_weapon_points_to_brennan: false,
  threatens_arrest: false,
  presents_concrete_evidence: false,
  asks_about_victor_relationship: false,
  mocks_or_dismisses_evie: false,
  repeated_accusation_without_evidence: false,
  harsh_accusation: false,
  threatens_or_mocks: false,
  suggests_public_brennan_confrontation: false,
  betrays_or_threatens_evie: false,
  stops_accusing_and_asks_facts: false,
  apologizes_or_softens: false,
};

const baseState = {
  player_input: "Tell me about Victor.",
  current_node: "POLITE_MASK" as EvieNode,
  trust_level: 1,
  fear_level: 0,
  guilt_pressure: 0,
  suspicion_level: 0,
  milestones: [] as EvieMilestone[],
  player_has_offered_protection: false,
  player_has_found_blackmail_photos: false,
  player_has_shown_locket: false,
  player_has_mentioned_brennan: false,
  player_knows_evie_was_at_alley: false,
  player_has_accused_evie: false,
  player_has_revealed_murder_weapon: false,
  critical_info_revealed: false,
  final_clue_revealed: false,
  detected_conditions: {},
};

/** Invoke the apply_transitions node directly with given conditions and optional milestones. */
async function applyWith(
  current_node: EvieNode,
  conditions: Partial<typeof allFalse>,
  milestones: EvieMilestone[] = [],
) {
  return evieReasoningSubgraph.nodes["apply_transitions"].invoke({
    ...baseState,
    current_node,
    milestones,
    detected_conditions: { ...allFalse, ...conditions },
  });
}

// ─── tests ──────────────────────────────────────────────────────────────────

describe("evie reasoning subgraph", () => {
  beforeEach(() => vi.clearAllMocks());

  // ── apply_transitions (pure logic, no LLM) ──────────────────────────────

  describe("apply_transitions – POLITE_MASK", () => {
    test("stays in POLITE_MASK with no triggers", async () => {
      const r = await applyWith("POLITE_MASK", {});
      expect(r.current_node).toBe("POLITE_MASK");
    });

    test("accuses_evie → DEFENSIVE_DENIAL", async () => {
      const r = await applyWith("POLITE_MASK", { accuses_evie: true });
      expect(r.current_node).toBe("DEFENSIVE_DENIAL");
    });

    test("threatens_arrest → PANICKED_RESISTANCE", async () => {
      const r = await applyWith("POLITE_MASK", { threatens_arrest: true });
      expect(r.current_node).toBe("PANICKED_RESISTANCE");
    });

    test("presents_concrete_evidence → EVIDENCE_PRESSURE", async () => {
      const r = await applyWith("POLITE_MASK", { presents_concrete_evidence: true });
      expect(r.current_node).toBe("EVIDENCE_PRESSURE");
    });

    test("asks_about_victor_relationship → BITTER_REMEMBERING", async () => {
      const r = await applyWith("POLITE_MASK", { asks_about_victor_relationship: true });
      expect(r.current_node).toBe("BITTER_REMEMBERING");
    });

    test("mentions_empathy → WARY_SOFTENING", async () => {
      const r = await applyWith("POLITE_MASK", { mentions_empathy: true });
      expect(r.current_node).toBe("WARY_SOFTENING");
    });
  });

  describe("apply_transitions – WARY_SOFTENING", () => {
    test("mentions_brennan → FEAR_SPIKE", async () => {
      const r = await applyWith("WARY_SOFTENING", { mentions_brennan: true });
      expect(r.current_node).toBe("FEAR_SPIKE");
    });

    test("accuses_evie → DEFENSIVE_DENIAL", async () => {
      const r = await applyWith("WARY_SOFTENING", { accuses_evie: true });
      expect(r.current_node).toBe("DEFENSIVE_DENIAL");
    });

    test("guarded milestone in state → GUARDED_DISCLOSURE", async () => {
      const r = await applyWith("WARY_SOFTENING", {}, ["empathy"]);
      expect(r.current_node).toBe("GUARDED_DISCLOSURE");
    });

    test("no triggers → POLITE_MASK", async () => {
      const r = await applyWith("WARY_SOFTENING", {});
      expect(r.current_node).toBe("POLITE_MASK");
    });
  });

  describe("apply_transitions – BITTER_REMEMBERING", () => {
    test("mocks_or_dismisses_evie → COLD_SHUTDOWN", async () => {
      const r = await applyWith("BITTER_REMEMBERING", { mocks_or_dismisses_evie: true });
      expect(r.current_node).toBe("COLD_SHUTDOWN");
    });

    test("guarded milestone → GUARDED_DISCLOSURE", async () => {
      const r = await applyWith("BITTER_REMEMBERING", {}, ["empathy"]);
      expect(r.current_node).toBe("GUARDED_DISCLOSURE");
    });

    test("no triggers → WARY_SOFTENING", async () => {
      const r = await applyWith("BITTER_REMEMBERING", {});
      expect(r.current_node).toBe("WARY_SOFTENING");
    });
  });

  describe("apply_transitions – DEFENSIVE_DENIAL", () => {
    test("2 critical milestones → CORNERED_CONFESSION", async () => {
      const r = await applyWith("DEFENSIVE_DENIAL", {}, ["blackmail_proof", "locket_evidence"]);
      expect(r.current_node).toBe("CORNERED_CONFESSION");
    });

    test("presents_concrete_evidence → EVIDENCE_PRESSURE", async () => {
      const r = await applyWith("DEFENSIVE_DENIAL", { presents_concrete_evidence: true });
      expect(r.current_node).toBe("EVIDENCE_PRESSURE");
    });

    test("offers_protection → WARY_SOFTENING", async () => {
      const r = await applyWith("DEFENSIVE_DENIAL", { offers_protection: true });
      expect(r.current_node).toBe("WARY_SOFTENING");
    });

    test("repeated_accusation_without_evidence → COLD_SHUTDOWN", async () => {
      const r = await applyWith("DEFENSIVE_DENIAL", { repeated_accusation_without_evidence: true });
      expect(r.current_node).toBe("COLD_SHUTDOWN");
    });

    test("no triggers → DEFENSIVE_DENIAL", async () => {
      const r = await applyWith("DEFENSIVE_DENIAL", {});
      expect(r.current_node).toBe("DEFENSIVE_DENIAL");
    });
  });

  describe("apply_transitions – FEAR_SPIKE", () => {
    test("offers_protection → PROTECTED_WITNESS", async () => {
      const r = await applyWith("FEAR_SPIKE", { offers_protection: true });
      expect(r.current_node).toBe("PROTECTED_WITNESS");
    });

    test("2 critical milestones → BROKEN_TRUSTING", async () => {
      const r = await applyWith("FEAR_SPIKE", {}, ["blackmail_proof", "locket_evidence"]);
      expect(r.current_node).toBe("BROKEN_TRUSTING");
    });

    test("threatens_or_mocks → COLD_SHUTDOWN", async () => {
      const r = await applyWith("FEAR_SPIKE", { threatens_or_mocks: true });
      expect(r.current_node).toBe("COLD_SHUTDOWN");
    });

    test("no triggers → GUARDED_DISCLOSURE", async () => {
      const r = await applyWith("FEAR_SPIKE", {});
      expect(r.current_node).toBe("GUARDED_DISCLOSURE");
    });
  });

  describe("apply_transitions – COLD_SHUTDOWN", () => {
    test("apologizes_or_softens → WARY_SOFTENING", async () => {
      const r = await applyWith("COLD_SHUTDOWN", { apologizes_or_softens: true });
      expect(r.current_node).toBe("WARY_SOFTENING");
    });

    test("presents_concrete_evidence → EVIDENCE_PRESSURE", async () => {
      const r = await applyWith("COLD_SHUTDOWN", { presents_concrete_evidence: true });
      expect(r.current_node).toBe("EVIDENCE_PRESSURE");
    });

    test("no triggers → COLD_SHUTDOWN", async () => {
      const r = await applyWith("COLD_SHUTDOWN", {});
      expect(r.current_node).toBe("COLD_SHUTDOWN");
    });
  });

  describe("apply_transitions – GUARDED_DISCLOSURE", () => {
    test("mentions_brennan → FEAR_SPIKE", async () => {
      const r = await applyWith("GUARDED_DISCLOSURE", { mentions_brennan: true });
      expect(r.current_node).toBe("FEAR_SPIKE");
    });

    test("threatens_arrest → PANICKED_RESISTANCE", async () => {
      const r = await applyWith("GUARDED_DISCLOSURE", { threatens_arrest: true });
      expect(r.current_node).toBe("PANICKED_RESISTANCE");
    });

    test("harsh_accusation → DEFENSIVE_DENIAL", async () => {
      const r = await applyWith("GUARDED_DISCLOSURE", { harsh_accusation: true });
      expect(r.current_node).toBe("DEFENSIVE_DENIAL");
    });

    test("2 critical milestones → BROKEN_TRUSTING", async () => {
      const r = await applyWith("GUARDED_DISCLOSURE", {}, ["blackmail_proof", "locket_evidence"]);
      expect(r.current_node).toBe("BROKEN_TRUSTING");
    });

    test("no triggers → GUARDED_DISCLOSURE", async () => {
      const r = await applyWith("GUARDED_DISCLOSURE", {});
      expect(r.current_node).toBe("GUARDED_DISCLOSURE");
    });
  });

  describe("apply_transitions – BROKEN_TRUSTING", () => {
    test("offers_protection → PROTECTED_WITNESS", async () => {
      const r = await applyWith("BROKEN_TRUSTING", { offers_protection: true });
      expect(r.current_node).toBe("PROTECTED_WITNESS");
    });

    test("accuses_evie → CORNERED_CONFESSION", async () => {
      const r = await applyWith("BROKEN_TRUSTING", { accuses_evie: true });
      expect(r.current_node).toBe("CORNERED_CONFESSION");
    });

    test("no triggers → BROKEN_TRUSTING", async () => {
      const r = await applyWith("BROKEN_TRUSTING", {});
      expect(r.current_node).toBe("BROKEN_TRUSTING");
    });
  });

  describe("apply_transitions – PANICKED_RESISTANCE", () => {
    test("offers_protection → FEAR_SPIKE", async () => {
      const r = await applyWith("PANICKED_RESISTANCE", { offers_protection: true });
      expect(r.current_node).toBe("FEAR_SPIKE");
    });

    test("2 critical milestones → CORNERED_CONFESSION", async () => {
      const r = await applyWith("PANICKED_RESISTANCE", {}, ["blackmail_proof", "locket_evidence"]);
      expect(r.current_node).toBe("CORNERED_CONFESSION");
    });

    test("guarded milestone → GUARDED_DISCLOSURE", async () => {
      const r = await applyWith("PANICKED_RESISTANCE", {}, ["empathy"]);
      expect(r.current_node).toBe("GUARDED_DISCLOSURE");
    });
  });

  describe("apply_transitions – state variable updates", () => {
    test("mentions_empathy adds milestone and increments trust_level", async () => {
      const r = await applyWith("POLITE_MASK", { mentions_empathy: true });
      expect(r.milestones).toContain("empathy");
      expect(r.trust_level).toBe(2);
    });

    test("offers_protection adds milestone, +2 trust, -1 fear, sets flag", async () => {
      const r = await applyWith("POLITE_MASK", { offers_protection: true });
      expect(r.milestones).toContain("protection_offer");
      expect(r.trust_level).toBe(3);
      expect(r.fear_level).toBe(-1);
      expect(r.player_has_offered_protection).toBe(true);
    });

    test("mentions_brennan adds milestone, +2 fear, sets flag", async () => {
      const r = await applyWith("POLITE_MASK", { mentions_brennan: true });
      expect(r.milestones).toContain("brennan_connection");
      expect(r.fear_level).toBe(2);
      expect(r.player_has_mentioned_brennan).toBe(true);
    });

    test("accuses_evie adds milestone, +2 suspicion, sets flag", async () => {
      const r = await applyWith("POLITE_MASK", { accuses_evie: true });
      expect(r.milestones).toContain("direct_accusation");
      expect(r.suspicion_level).toBe(2);
      expect(r.player_has_accused_evie).toBe(true);
    });

    test("existing milestones are preserved alongside new ones", async () => {
      const r = await applyWith("POLITE_MASK", { mentions_empathy: true }, ["brennan_connection"]);
      expect(r.milestones).toContain("brennan_connection");
      expect(r.milestones).toContain("empathy");
    });

    test("critical_info_revealed set to true when entering BROKEN_TRUSTING", async () => {
      const r = await applyWith("FEAR_SPIKE", {}, ["blackmail_proof", "locket_evidence"]);
      expect(r.current_node).toBe("BROKEN_TRUSTING");
      expect(r.critical_info_revealed).toBe(true);
    });

    test("critical_info_revealed set to true when entering CORNERED_CONFESSION", async () => {
      const r = await applyWith("DEFENSIVE_DENIAL", {}, ["blackmail_proof", "locket_evidence"]);
      expect(r.current_node).toBe("CORNERED_CONFESSION");
      expect(r.critical_info_revealed).toBe(true);
    });

    test("triple milestone combo triggers critical_ready", async () => {
      // direct_accusation + blackmail_proof + alley_presence = critical_ready via hasAll rule
      const r = await applyWith(
        "DEFENSIVE_DENIAL",
        {},
        ["direct_accusation", "blackmail_proof", "alley_presence"],
      );
      expect(r.current_node).toBe("CORNERED_CONFESSION");
    });
  });

  // ── detect_conditions (LLM-dependent) ───────────────────────────────────

  describe("detect_conditions node", () => {
    test("invokes LLM with player input and stores result", async () => {
      const mockResult = { ...allFalse, mentions_empathy: true };
      mockConditionInvoke.mockResolvedValueOnce(mockResult);

      const r = await evieReasoningSubgraph.nodes["detect_conditions"].invoke({
        ...baseState,
        player_input: "I understand what you went through.",
      });

      expect(mockConditionInvoke).toHaveBeenCalledOnce();
      expect(r.detected_conditions).toMatchObject({ mentions_empathy: true });
    });

    test("all-false conditions are stored verbatim", async () => {
      mockConditionInvoke.mockResolvedValueOnce(allFalse);

      const r = await evieReasoningSubgraph.nodes["detect_conditions"].invoke(baseState);

      expect(r.detected_conditions).toMatchObject(allFalse);
    });
  });

  // ── full subgraph (end-to-end) ───────────────────────────────────────────

  describe("full subgraph", () => {
    test("mentions_empathy: detects conditions, transitions to WARY_SOFTENING, increments trust", async () => {
      mockConditionInvoke.mockResolvedValueOnce({ ...allFalse, mentions_empathy: true });

      const r = await evieReasoningSubgraph.invoke(baseState);

      expect(r.current_node).toBe("WARY_SOFTENING");
      expect(r.milestones).toContain("empathy");
      expect(r.trust_level).toBe(2);
    });

    test("no conditions: stays in POLITE_MASK", async () => {
      mockConditionInvoke.mockResolvedValueOnce(allFalse);

      const r = await evieReasoningSubgraph.invoke(baseState);

      expect(r.current_node).toBe("POLITE_MASK");
    });

    test("accuses_evie: transitions to DEFENSIVE_DENIAL and increments suspicion", async () => {
      mockConditionInvoke.mockResolvedValueOnce({ ...allFalse, accuses_evie: true });

      const r = await evieReasoningSubgraph.invoke(baseState);

      expect(r.current_node).toBe("DEFENSIVE_DENIAL");
      expect(r.suspicion_level).toBe(2);
    });
  });
});
