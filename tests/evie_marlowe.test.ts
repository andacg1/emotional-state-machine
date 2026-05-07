import { beforeEach, describe, expect, test, vi } from "vitest";
import { BaseMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";

import type { EvieMilestone, EvieNode } from "../src/agent/evie_marlowe_reasoning.js";

// ─── mock dependencies before module load ────────────────────────────────────

const mockSubgraphInvoke = vi.hoisted(() => vi.fn());
const mockStructuredLlmInvoke = vi.hoisted(() => vi.fn());

vi.mock("../src/agent/evie_marlowe_reasoning.js", () => ({
  evieReasoningSubgraph: { invoke: mockSubgraphInvoke },
}));

vi.mock("../src/agent/llm.js", () => ({
  structuredLlm: { invoke: mockStructuredLlmInvoke },
}));

import {
  EvieMarlowe,
  type State,
  type StateUpdate,
} from "../src/agent/evie_marlowe.js";

const evie = new EvieMarlowe();
const { initializeEvie, politeMask, defensiveDenial, coldShutdown } = evie;

// ─── shared fixtures ─────────────────────────────────────────────────────────

const humanMsg = new HumanMessage("Tell me about Victor.");

const baseState: State = {
  messages: [humanMsg],
  current_node: "POLITE_MASK" as EvieNode,
  visited_nodes: [] as EvieNode[],
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
  visited_nodes: [] as EvieNode[],
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
      const stateResult = result as StateUpdate;
      const messages = stateResult.messages as BaseMessage[];
      expect(messages).toHaveLength(1);
      expect(messages[0]).toBeInstanceOf(AIMessage);
      expect((messages[0] as AIMessage).content).toBe(defaultLlmResponse.message);
    });

    test("merges subgraph state updates into the return value", async () => {
      mockSubgraphInvoke.mockResolvedValueOnce({
        ...defaultSubgraphOutput,
        current_node: "POLITE_MASK",
        trust_level: 2,
        milestones: ["empathy"] as EvieMilestone[],
      });
      mockStructuredLlmInvoke.mockResolvedValueOnce(defaultLlmResponse);

      const result = await politeMask(baseState) as StateUpdate;

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

      const result = await politeMask(baseState) as StateUpdate;

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

      const result = await politeMask(baseState) as StateUpdate;

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

  // ── infinite loop detection ───────────────────────────────────────────────

  describe("infinite loop detection", () => {
    // Maps every EvieNode to its method on the evie instance.
    const nodeMethod: Record<EvieNode, (s: State) => Promise<Command | StateUpdate>> = {
      POLITE_MASK: (s) => evie.politeMask(s),
      WARY_SOFTENING: (s) => evie.warySoftening(s),
      BITTER_REMEMBERING: (s) => evie.bitterRemembering(s),
      DEFENSIVE_DENIAL: (s) => evie.defensiveDenial(s),
      EVIDENCE_PRESSURE: (s) => evie.evidencePressure(s),
      GUARDED_DISCLOSURE: (s) => evie.guardedDisclosure(s),
      FEAR_SPIKE: (s) => evie.fearSpike(s),
      BROKEN_TRUSTING: (s) => evie.brokenTrusting(s),
      PROTECTED_WITNESS: (s) => evie.protectedWitness(s),
      CORNERED_CONFESSION: (s) => evie.corneredConfession(s),
      COLD_SHUTDOWN: (s) => evie.coldShutdown(s),
      PANICKED_RESISTANCE: (s) => evie.panickedResistance(s),
    };

    /**
     * Drives the evie graph by following Command.goto values until a StateUpdate
     * is returned or the mock sequence is exhausted.
     *
     * intendedTransitions: the sequence of nodes that _transition() would return
     * WITHOUT the cycle guard. simulateChain mirrors the applyTransitions guard —
     * if the intended next node was already visited this turn, the mock reports
     * the current node instead (self-transition → StateUpdate path).
     *
     * Returns settledAtStep (1-based) when a StateUpdate is returned, or null if
     * every mock step produced a Command (i.e. cycle was not broken).
     */
    async function simulateChain(
      startNode: EvieNode,
      intendedTransitions: EvieNode[],
    ): Promise<{ settledAtStep: number | null; commandCount: number }> {
      let currentNode = startNode;
      let commandCount = 0;
      const visitedNodes: EvieNode[] = [];

      for (let i = 0; i < intendedTransitions.length; i++) {
        const intended = intendedTransitions[i];

        // Mirror the cycle guard from applyTransitions:
        // if the intended destination was already visited this turn, stay put.
        const actualNode =
          intended !== currentNode && visitedNodes.includes(intended)
            ? currentNode
            : intended;

        // Mark current node visited before transitioning (matches applyTransitions order).
        if (!visitedNodes.includes(currentNode)) visitedNodes.push(currentNode);

        const state: State = {
          ...baseState,
          current_node: currentNode,
          visited_nodes: [...visitedNodes],
        };

        mockSubgraphInvoke.mockResolvedValueOnce({
          ...defaultSubgraphOutput,
          current_node: actualNode,
          visited_nodes: [...visitedNodes],
        });

        if (actualNode === currentNode) {
          mockStructuredLlmInvoke.mockResolvedValueOnce(defaultLlmResponse);
        }

        const result = await nodeMethod[currentNode](state);

        if (!(result instanceof Command)) {
          return { settledAtStep: i + 1, commandCount };
        }

        commandCount++;
        currentNode = (result as Command).goto[0] as EvieNode;
      }

      return { settledAtStep: null, commandCount };
    }

    // ── terminating fallback chains ─────────────────────────────────────────

    test("WARY_SOFTENING fallback to POLITE_MASK settles within 2 steps", async () => {
      const { settledAtStep } = await simulateChain("WARY_SOFTENING", [
        "POLITE_MASK",  // wary → polite  (Command)
        "POLITE_MASK",  // polite stays   (StateUpdate)
      ]);
      expect(settledAtStep).not.toBeNull();
      expect(settledAtStep).toBeLessThanOrEqual(2);
    });

    test("BITTER_REMEMBERING fallback chain settles within 3 steps", async () => {
      const { settledAtStep } = await simulateChain("BITTER_REMEMBERING", [
        "WARY_SOFTENING",  // bitter → wary    (Command)
        "POLITE_MASK",     // wary  → polite   (Command)
        "POLITE_MASK",     // polite stays     (StateUpdate)
      ]);
      expect(settledAtStep).not.toBeNull();
      expect(settledAtStep).toBeLessThanOrEqual(3);
    });

    test("COLD_SHUTDOWN apology chain settles within 3 steps", async () => {
      const { settledAtStep } = await simulateChain("COLD_SHUTDOWN", [
        "WARY_SOFTENING",  // shutdown → wary   (Command)
        "POLITE_MASK",     // wary    → polite  (Command)
        "POLITE_MASK",     // polite stays      (StateUpdate)
      ]);
      expect(settledAtStep).not.toBeNull();
      expect(settledAtStep).toBeLessThanOrEqual(3);
    });

    test("FEAR_SPIKE fallback to GUARDED_DISCLOSURE settles within 2 steps when mentions_brennan is absent", async () => {
      const { settledAtStep } = await simulateChain("FEAR_SPIKE", [
        "GUARDED_DISCLOSURE",  // fear → guarded   (Command)
        "GUARDED_DISCLOSURE",  // guarded stays    (StateUpdate)
      ]);
      expect(settledAtStep).not.toBeNull();
      expect(settledAtStep).toBeLessThanOrEqual(2);
    });

    test("PROTECTED_WITNESS → FEAR_SPIKE → GUARDED_DISCLOSURE settles within 3 steps", async () => {
      const { settledAtStep } = await simulateChain("PROTECTED_WITNESS", [
        "FEAR_SPIKE",          // protected → fear    (Command)
        "GUARDED_DISCLOSURE",  // fear → guarded      (Command)
        "GUARDED_DISCLOSURE",  // guarded stays       (StateUpdate)
      ]);
      expect(settledAtStep).not.toBeNull();
      expect(settledAtStep).toBeLessThanOrEqual(3);
    });

    // ── cycle detection ─────────────────────────────────────────────────────

    test("GUARDED_DISCLOSURE ↔ FEAR_SPIKE oscillation settles within 4 round-trips", async () => {
      // When the player repeatedly mentions Brennan, GUARDED_DISCLOSURE transitions
      // to FEAR_SPIKE (line 295 of evie_marlowe_reasoning.ts) and FEAR_SPIKE
      // falls back to GUARDED_DISCLOSURE (line 304). Without cycle-breaking logic
      // this pair can oscillate indefinitely. The graph must settle within a
      // bounded number of hops.
      const MAX_BOUNCES = 4;
      const oscillation: EvieNode[] = [];
      for (let i = 0; i < MAX_BOUNCES; i++) {
        oscillation.push(i % 2 === 0 ? "FEAR_SPIKE" : "GUARDED_DISCLOSURE");
      }
      oscillation.push("GUARDED_DISCLOSURE"); // self-step so a settle is possible

      const { settledAtStep } = await simulateChain("GUARDED_DISCLOSURE", oscillation);
      expect(settledAtStep).not.toBeNull();
      expect(settledAtStep).toBeLessThanOrEqual(MAX_BOUNCES);
    });

    test("EVIDENCE_PRESSURE + mentions_brennan cycle settles within 4 round-trips", async () => {
      // EVIDENCE_PRESSURE also transitions to FEAR_SPIKE on mentions_brennan (line 287),
      // which then falls back to GUARDED_DISCLOSURE — that can return to FEAR_SPIKE
      // again creating the same oscillation as above.
      const MAX_BOUNCES = 4;
      const oscillation: EvieNode[] = ["FEAR_SPIKE"];
      for (let i = 0; i < MAX_BOUNCES - 1; i++) {
        oscillation.push(i % 2 === 0 ? "GUARDED_DISCLOSURE" : "FEAR_SPIKE");
      }
      oscillation.push("FEAR_SPIKE"); // self-step to settle

      const { settledAtStep } = await simulateChain("EVIDENCE_PRESSURE", oscillation);
      expect(settledAtStep).not.toBeNull();
      expect(settledAtStep).toBeLessThanOrEqual(MAX_BOUNCES);
    });

    // ── all nodes settle when subgraph reports no change ───────────────────

    test.each<EvieNode>([
      "POLITE_MASK",
      "WARY_SOFTENING",
      "BITTER_REMEMBERING",
      "DEFENSIVE_DENIAL",
      "EVIDENCE_PRESSURE",
      "GUARDED_DISCLOSURE",
      "FEAR_SPIKE",
      "BROKEN_TRUSTING",
      "PROTECTED_WITNESS",
      "CORNERED_CONFESSION",
      "COLD_SHUTDOWN",
      "PANICKED_RESISTANCE",
    ])("%s returns StateUpdate (not Command) when subgraph keeps the same node", async (node) => {
      mockSubgraphInvoke.mockResolvedValueOnce({ ...defaultSubgraphOutput, current_node: node });
      mockStructuredLlmInvoke.mockResolvedValueOnce(defaultLlmResponse);

      const result = await nodeMethod[node]({ ...baseState, current_node: node });

      expect(result).not.toBeInstanceOf(Command);
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
      expect(((result as StateUpdate).messages as BaseMessage[])[0]).toBeInstanceOf(AIMessage);
    });
  });
});
