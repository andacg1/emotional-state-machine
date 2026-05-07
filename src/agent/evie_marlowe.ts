import { AIMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import {
  Annotation,
  Command,
  MessagesAnnotation,
} from "@langchain/langgraph";

import {
  evieReasoningSubgraph,
  type EvieMilestone,
  type EvieNode,
  type EvieReasoningState,
} from "./evie_marlowe_reasoning.js";
import { structuredLlm } from "./llm.js";
import evieData from "./evie_marlowe.json" with { type: "json" };

type Emotion =
  | "relaxed"
  | "nervous"
  | "panicking"
  | "angry"
  | "upset"
  | "depressed";

interface NodeBehavior {
  emotionalState: string;
  allowedInformation: string[];
  forbiddenInformation: string[];
}

export class EvieMarlowe {
  private readonly systemMessages: SystemMessage[];

  constructor() {
    this.systemMessages = [new SystemMessage(evieData.systemPrompt)];
  }

  private knowledgeSystemMessage(
    currentKnowledge: string[] | null | undefined,
  ): SystemMessage {
    const existing =
      currentKnowledge && currentKnowledge.length > 0
        ? currentKnowledge.map((k) => `- ${k}`).join("\n")
        : "None yet.";
    return new SystemMessage(
      evieData.knowledgeMessageTemplate.replace("{{existing}}", existing),
    );
  }

  private buildNodePrompt(args: {
    currentNode: EvieNode;
    emotionalState: string;
    milestones: EvieMilestone[];
    memorySummary: string;
    allowedInformation: string[];
    forbiddenInformation: string[];
    playerInput: string;
  }): SystemMessage {
    const milestonesText = args.milestones.join("\n");
    const allowed = args.allowedInformation.join("\n");
    const forbidden = args.forbiddenInformation.join("\n");
    const content = evieData.nodePromptTemplate
      .replace("{{currentNode}}", args.currentNode)
      .replace("{{emotionalState}}", args.emotionalState)
      .replace("{{milestonesText}}", milestonesText)
      .replace("{{memorySummary}}", args.memorySummary)
      .replace("{{allowed}}", allowed)
      .replace("{{forbidden}}", forbidden)
      .replace("{{playerInput}}", args.playerInput);
    return new SystemMessage(content);
  }

  static addSetReducer(
    current: EvieMilestone[] | undefined,
    update: EvieMilestone[] | undefined,
  ): EvieMilestone[] {
    const base = current ?? [];
    const incoming = update ?? [];
    const seen = new Set(base);
    return [...base, ...incoming.filter((x) => !seen.has(x))];
  }

  private static lastPlayerInput(messages: BaseMessage[]): string {
    if (!messages || messages.length === 0) return "";
    const last = messages[messages.length - 1];
    const content = last.content;
    return typeof content === "string" ? content : JSON.stringify(content);
  }

  private buildReasoningState(state: State): EvieReasoningState {
    return {
      player_input: EvieMarlowe.lastPlayerInput(state.messages),
      current_node: state.current_node ?? "POLITE_MASK",
      trust_level: state.trust_level ?? 0,
      fear_level: state.fear_level ?? 0,
      guilt_pressure: state.guilt_pressure ?? 0,
      suspicion_level: state.suspicion_level ?? 0,
      milestones: state.milestones ?? [],
      visited_nodes: state.visited_nodes ?? [],
      player_has_offered_protection: state.player_has_offered_protection ?? false,
      player_has_found_blackmail_photos:
        state.player_has_found_blackmail_photos ?? false,
      player_has_shown_locket: state.player_has_shown_locket ?? false,
      player_has_mentioned_brennan: state.player_has_mentioned_brennan ?? false,
      player_knows_evie_was_at_alley: state.player_knows_evie_was_at_alley ?? false,
      player_has_accused_evie: state.player_has_accused_evie ?? false,
      player_has_revealed_murder_weapon:
        state.player_has_revealed_murder_weapon ?? false,
      critical_info_revealed: state.critical_info_revealed ?? false,
      final_clue_revealed: state.final_clue_revealed ?? false,
      detected_conditions: {},
    };
  }

  initializeEvie = async (state: State): Promise<Command> => {
    const goto = state.current_node ?? "POLITE_MASK";
    return new Command({
      update: {
        current_node: goto,
        npc_name: evieData.initializationData.npc_name,
        setting: evieData.initializationData.setting,
        location: evieData.initializationData.location,
        relationship_to_victim: evieData.initializationData.relationship_to_victim,
        public_identity: evieData.initializationData.public_identity,
        secret_identity: evieData.initializationData.secret_identity,
        emotional_state: state.emotional_state ?? "relaxed",
        trust_level: state.trust_level ?? 1,
        fear_level: state.fear_level ?? 0,
        suspicion_level: state.suspicion_level ?? 0,
        guilt_pressure: state.guilt_pressure ?? 0,
        player_has_accused_evie: state.player_has_accused_evie ?? false,
        player_has_shown_locket: state.player_has_shown_locket ?? false,
        player_has_found_blackmail_photos:
          state.player_has_found_blackmail_photos ?? false,
        player_has_mentioned_brennan: state.player_has_mentioned_brennan ?? false,
        player_has_proven_victor_was_blackmailing_evie:
          state.player_has_proven_victor_was_blackmailing_evie ?? false,
        player_has_revealed_murder_weapon:
          state.player_has_revealed_murder_weapon ?? false,
        player_knows_evie_was_at_alley:
          state.player_knows_evie_was_at_alley ?? false,
        player_has_threatened_arrest: state.player_has_threatened_arrest ?? false,
        player_has_offered_protection: state.player_has_offered_protection ?? false,
        critical_info_revealed: state.critical_info_revealed ?? false,
        final_clue_revealed: state.final_clue_revealed ?? false,
        milestones: state.milestones ?? [],
      },
      goto,
    });
  };

  private async runEmotionalNode(
    state: State,
    behavior: NodeBehavior,
  ): Promise<Command | StateUpdate> {
    const subgraphOutput = (await evieReasoningSubgraph.invoke(
      this.buildReasoningState(state),
    )) as EvieReasoningState;
    const knowledgeMsg = this.knowledgeSystemMessage(state.knowledge);
    const playerInput = EvieMarlowe.lastPlayerInput(state.messages);

    const nodePrompt = this.buildNodePrompt({
      currentNode: subgraphOutput.current_node,
      emotionalState: behavior.emotionalState,
      milestones: state.milestones ?? [],
      memorySummary:
        typeof knowledgeMsg.content === "string" ? knowledgeMsg.content : "",
      allowedInformation: behavior.allowedInformation,
      forbiddenInformation: behavior.forbiddenInformation,
      playerInput,
    });


    if (subgraphOutput.current_node === state.current_node) {
      const response = await structuredLlm.invoke([
        ...this.systemMessages,
        nodePrompt,
        knowledgeMsg,
        ...state.messages,
      ]);
      return {
        ...subgraphOutput,
        messages: [new AIMessage(response.message)],
        emotion: response.emotion as Emotion,
        topic: response.topic,
        summary: response.summary,
        knowledge: response.knowledge ?? null,
        visited_nodes: [],
      } satisfies Partial<StateUpdate> as StateUpdate;
    }

    return new Command({
      update: subgraphOutput,
      goto: subgraphOutput.current_node,
    });
  }

  politeMask = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.POLITE_MASK);

  warySoftening = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.WARY_SOFTENING);

  bitterRemembering = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.BITTER_REMEMBERING);

  defensiveDenial = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.DEFENSIVE_DENIAL);

  evidencePressure = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.EVIDENCE_PRESSURE);

  guardedDisclosure = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.GUARDED_DISCLOSURE);

  fearSpike = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.FEAR_SPIKE);

  brokenTrusting = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.BROKEN_TRUSTING);

  protectedWitness = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.PROTECTED_WITNESS);

  corneredConfession = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.CORNERED_CONFESSION);

  coldShutdown = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.COLD_SHUTDOWN);

  panickedResistance = (state: State) =>
    this.runEmotionalNode(state, evieData.nodes.PANICKED_RESISTANCE);
}

export const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  current_node: Annotation<EvieNode>(),
  visited_nodes: Annotation<EvieNode[]>(),
  emotion: Annotation<Emotion>(),
  milestones: Annotation<EvieMilestone[]>({
    reducer: EvieMarlowe.addSetReducer,
    default: () => [],
  }),
  topic: Annotation<string>(),
  summary: Annotation<string>(),
  knowledge: Annotation<string[] | null>(),
  draft_response: Annotation<string | null>(),

  npc_name: Annotation<string>(),
  setting: Annotation<string>(),
  location: Annotation<string>(),
  relationship_to_victim: Annotation<string>(),
  public_identity: Annotation<string>(),
  secret_identity: Annotation<string>(),
  emotional_state: Annotation<string>(),

  trust_level: Annotation<number>(),
  fear_level: Annotation<number>(),
  suspicion_level: Annotation<number>(),
  guilt_pressure: Annotation<number>(),

  player_has_accused_evie: Annotation<boolean>(),
  player_has_shown_locket: Annotation<boolean>(),
  player_has_found_blackmail_photos: Annotation<boolean>(),
  player_has_mentioned_brennan: Annotation<boolean>(),
  player_has_proven_victor_was_blackmailing_evie: Annotation<boolean>(),
  player_has_revealed_murder_weapon: Annotation<boolean>(),
  player_knows_evie_was_at_alley: Annotation<boolean>(),
  player_has_threatened_arrest: Annotation<boolean>(),
  player_has_offered_protection: Annotation<boolean>(),
  critical_info_revealed: Annotation<boolean>(),
  final_clue_revealed: Annotation<boolean>(),
});

export type State = typeof StateAnnotation.State;
export type StateUpdate = typeof StateAnnotation.Update;
