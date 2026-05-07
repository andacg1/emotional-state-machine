import { beforeEach, describe, expect, test, vi } from "vitest";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";

import type { EvieMilestone, EvieNode } from "../src/agent/evie_marlowe_reasoning.js";

// ─── mock dependencies before module load ────────────────────────────────────

const mockSubgraphInvoke = vi.hoisted(() => vi.fn());
const mockStructuredLlmInvoke = vi.hoisted(() => vi.fn());

vi.mock("./evie_marlowe_reasoning.js", () => ({
  evieReasoningSubgraph: { invoke: mockSubgraphInvoke },
}));

vi.mock("./llm.js", () => ({
  structuredLlm: { invoke: mockStructuredLlmInvoke },
}));

import {
  initializeEvie,
  politeMask,
  defensiveDenial,
  coldShutdown,
  type State,
} from "../src/agent/evie_marlowe.js";

// ─── shared fixtures ─────────────────────────────────────────────────────────

const humanMsg = new HumanMessage("Tell me about Victor.");

const baseState: State = {
  messages: [humanMsg],
  current_node: "POLITE_MASK" as EvieNode,
  milestones: [] as EvieMilestone[],
  trust_level: 1,
  fear_level: 0,
  guilt_pressure: 0,
  suspicion_level: 0,
  emotion: "relaxed",
  topic: "",
  summary: "",
  knowledge: null,
  draft_response: null,
  npc_name: "",
  setting: "",
  location: "",
  relationship_to_victim: "",
  public_identity: "",
  secret_identity: "",
  emotional_state: "relaxed",
  player_has_accused_evie: false,
  player_has_shown_locket: false,
  player_has_found_blackmail_photos: false,
  player_has_mentioned_brennan: false,
  player_has_proven_victor_was_blackmailing_evie: false,
  player_has_revealed_murder_weapon: false,
  player_knows_evie_was_at_alley: false,
  player_has_threatened_arrest: false,
  player_has_offered_protection: false,
  critical_info_revealed: false,
  final_clue_revealed: false,
};

const defaultSubgraphOutput = {
  player_input: humanMsg.content as string,
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

const defaultLlmResponse = {
  message: "Detective, men like Victor Vale collected enemies the way other men collected matchbooks.",
  emotion: "relaxed",
  topic: "Victor Vale",
  summary: "Evie deflected the question about Victor.",
  knowledge: null,
};

// ─── tests ───────────────────────────────────────────────────────────────────

describe("evie_marlowe", () => {
  beforeEach(() => vi.clearAllMocks());

  // ── initializeEvie ────────────────────────────────────────────────────────

  describe("initializeEvie", () => {
    test("defaults to POLITE_MASK when current_node is unset", async () => {
      const state = { ...baseState, current_node: undefined as unknown as EvieNode };
      const result = await initializeEvie(state);
      expect(result).toBeInstanceOf(Command);
      expect(result.goto).toContain("POLITE_MASK");
      expect(result.update).toMatchObject({ current_node: "POLITE_MASK" });
    });

    test("preserves an existing current_node", async () => {
      const state = { ...baseState, current_node: "DEFENSIVE_DENIAL" as EvieNode };
      const result = await initializeEvie(state);
      expect(result.goto).toContain("DEFENSIVE_DENIAL");
      expect(result.update).toMatchObject({ current_node: "DEFENSIVE_DENIAL" });
    });

    test("sets character identity defaults", async () => {
      const result = await initializeEvie(baseState);
      expect(result.update).toMatchObject({
        npc_name: "Evelyn 'Evie' Marlowe",
        setting: "Los Angeles, 1952",
        location: "Blue Dahlia Club",
      });
    });

    test("defaults trust_level to 1 when unset", async () => {
      const state = { ...baseState, trust_level: undefined as unknown as number };
      const result = await initializeEvie(state);
      expect(result.update).toMatchObject({ trust_level: 1 });
    });

    test("preserves existing trust_level", async () => {
      const state = { ...baseState, trust_level: 3 };
      const result = await initializeEvie(state);
      expect(result.update).toMatchObject({ trust_level: 3 });
    });

    test("defaults all boolean flags to false", async () => {
      const state = { ...baseState, player_has_accused_evie: undefined as unknown as boolean };
      const result = await initializeEvie(state);
      expect(result.update).toMatchObject({
        player_has_accused_evie: false,
        player_has_offered_protection: false,
        critical_info_revealed: false,
        final_clue_revealed: false,
      });
    });
  });

  // ── emotional nodes – same-node path ─────────────────────────────────────

  describe("emotional nodes – staying in the same node", () => {
    test("returns AIMessage when subgraph keeps current_node", async () => {
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "POLITE_MASK",
      });
      mockStructuredLlmInvoke.mockResolvedValueOnce(defaultLlmResponse);

      const result = await politeMask(baseState);

      expect(result).not.toBeInstanceOf(Command);
      expect(result.messages).toHaveLength(1);
      expect(result.messages[0]).toBeInstanceOf(AIMessage);
      expect((result.messages[0] as AIMessage).content).toBe(defaultLlmResponse.message);
    });

    test("merges subgraph state updates into the return value", async () => {
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "POLITE_MASK",
        trust_level: 2,
        milestones: ["empathy"] as EvieMilestone[],
      });
      mockStructuredLlmInvoke.mockResolvedValueOnce(defaultLlmResponse);

      const result = await politeMask(baseState);

      expect(result.trust_level).toBe(2);
      expect(result.milestones).toContain("empathy");
    });

    test("maps LLM response fields into state", async () => {
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "POLITE_MASK",
      });
      mockStructuredLlmInvoke.mockResolvedValueOnce({
        ...defaultLlmResponse,
        emotion: "nervous",
        topic: "the murder",
        summary: "Evie was nervous.",
        knowledge: ["Player showed the locket"],
      });

      const result = await politeMask(baseState);

      expect(result.emotion).toBe("nervous");
      expect(result.topic).toBe("the murder");
      expect(result.summary).toBe("Evie was nervous.");
      expect(result.knowledge).toEqual(["Player showed the locket"]);
    });

    test("null LLM knowledge is stored as null", async () => {
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "POLITE_MASK",
      });
      mockStructuredLlmInvoke.mockResolvedValueOnce({ ...defaultLlmResponse, knowledge: null });

      const result = await politeMask(baseState);

      expect(result.knowledge).toBeNull();
    });

    test("LLM receives the conversation messages from state", async () => {
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "POLITE_MASK",
      });
      mockStructuredLlmInvoke.mockResolvedValueOnce(defaultLlmResponse);

      await politeMask(baseState);

      const llmArgs: unknown[] = mockStructuredLlmInvoke.mock.calls[0][0] as unknown[];
      const hasOriginalMessage = llmArgs.some(
        (m) => m instanceof HumanMessage && m.content === humanMsg.content,
      );
      expect(hasOriginalMessage).toBe(true);
    });
  });

  // ── emotional nodes – transition path ────────────────────────────────────

  describe("emotional nodes – transitioning to a new node", () => {
    test("returns Command when subgraph changes current_node", async () => {
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "DEFENSIVE_DENIAL",
      });

      const result = await politeMask(baseState);

      expect(result).toBeInstanceOf(Command);
      expect((result as Command).goto).toContain("DEFENSIVE_DENIAL");
    });

    test("Command.update contains subgraph state fields", async () => {
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "WARY_SOFTENING",
        trust_level: 3,
        milestones: ["empathy"] as EvieMilestone[],
      });

      const result = await politeMask(baseState);

      expect(result).toBeInstanceOf(Command);
      const update = (result as Command).update as Record<string, unknown>;
      expect(update.trust_level).toBe(3);
      expect(update.milestones).toContain("empathy");
    });

    test("defensiveDenial returns Command on transition to CORNERED_CONFESSION", async () => {
      const state = { ...baseState, current_node: "DEFENSIVE_DENIAL" as EvieNode };
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "CORNERED_CONFESSION",
      });

      const result = await defensiveDenial(state);

      expect(result).toBeInstanceOf(Command);
      expect((result as Command).goto).toContain("CORNERED_CONFESSION");
    });

    test("coldShutdown stays in COLD_SHUTDOWN when no change", async () => {
      const state = { ...baseState, current_node: "COLD_SHUTDOWN" as EvieNode };
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "COLD_SHUTDOWN",
      });
      mockStructuredLlmInvoke.mockResolvedValueOnce(defaultLlmResponse);

      const result = await coldShutdown(state);

      expect(result).not.toBeInstanceOf(Command);
      expect(result.messages[0]).toBeInstanceOf(AIMessage);
    });
  });
});
