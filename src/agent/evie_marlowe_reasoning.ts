import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { Annotation, StateGraph, START } from "@langchain/langgraph";
import { z } from "zod";

const _llm = new ChatOpenAI({
  model: process.env.LMSTUDIO_MODEL ?? "gemma-4-e4b",
  apiKey: process.env.LMSTUDIO_API_KEY ?? "lm-studio",
  temperature: 0,
  configuration: {
    baseURL: process.env.LMSTUDIO_BASE_URL ?? "http://localhost:1234/v1",
  },
});

export const PlayerInputConditionsSchema = z.object({
  mentions_empathy: z.boolean(),
  offers_protection: z.boolean(),
  shows_blackmail_evidence: z.boolean(),
  shows_locket: z.boolean(),
  mentions_brennan: z.boolean(),
  proves_alley_presence: z.boolean(),
  accuses_evie: z.boolean(),
  mentions_murder_weapon_points_to_brennan: z.boolean(),
  threatens_arrest: z.boolean(),
  presents_concrete_evidence: z.boolean(),
  asks_about_victor_relationship: z.boolean(),
  mocks_or_dismisses_evie: z.boolean(),
  repeated_accusation_without_evidence: z.boolean(),
  harsh_accusation: z.boolean(),
  threatens_or_mocks: z.boolean(),
  suggests_public_brennan_confrontation: z.boolean(),
  betrays_or_threatens_evie: z.boolean(),
  stops_accusing_and_asks_facts: z.boolean(),
  apologizes_or_softens: z.boolean(),
});

export type PlayerInputConditions = z.infer<typeof PlayerInputConditionsSchema>;

const _conditionLlm = _llm.withStructuredOutput(PlayerInputConditionsSchema, {
  name: "PlayerInputConditions",
});

const _DETECTION_SYSTEM = new SystemMessage(`
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
`);

export type EvieMilestone =
  | "empathy"
  | "protection_offer"
  | "blackmail_proof"
  | "locket_evidence"
  | "brennan_connection"
  | "alley_presence"
  | "murder_weapon"
  | "direct_accusation";

export type EvieNode =
  | "POLITE_MASK"
  | "WARY_SOFTENING"
  | "BITTER_REMEMBERING"
  | "DEFENSIVE_DENIAL"
  | "EVIDENCE_PRESSURE"
  | "GUARDED_DISCLOSURE"
  | "FEAR_SPIKE"
  | "BROKEN_TRUSTING"
  | "PROTECTED_WITNESS"
  | "CORNERED_CONFESSION"
  | "COLD_SHUTDOWN"
  | "PANICKED_RESISTANCE";

export const EvieReasoningStateAnnotation = Annotation.Root({
  player_input: Annotation<string>(),
  current_node: Annotation<EvieNode>(),
  trust_level: Annotation<number>(),
  fear_level: Annotation<number>(),
  guilt_pressure: Annotation<number>(),
  suspicion_level: Annotation<number>(),
  milestones: Annotation<EvieMilestone[]>(),
  player_has_offered_protection: Annotation<boolean>(),
  player_has_found_blackmail_photos: Annotation<boolean>(),
  player_has_shown_locket: Annotation<boolean>(),
  player_has_mentioned_brennan: Annotation<boolean>(),
  player_knows_evie_was_at_alley: Annotation<boolean>(),
  player_has_accused_evie: Annotation<boolean>(),
  player_has_revealed_murder_weapon: Annotation<boolean>(),
  critical_info_revealed: Annotation<boolean>(),
  final_clue_revealed: Annotation<boolean>(),
  detected_conditions: Annotation<PlayerInputConditions | Record<string, never>>(),
});

export type EvieReasoningState = typeof EvieReasoningStateAnnotation.State;
export type EvieReasoningUpdate = typeof EvieReasoningStateAnnotation.Update;

const _hasAny = (milestones: EvieMilestone[], required: EvieMilestone[]): boolean =>
  required.some((m) => milestones.includes(m));

const _hasAll = (milestones: EvieMilestone[], required: EvieMilestone[]): boolean =>
  required.every((m) => milestones.includes(m));

const _countAny = (milestones: EvieMilestone[], required: EvieMilestone[]): number =>
  required.reduce((acc, m) => acc + (milestones.includes(m) ? 1 : 0), 0);

async function detectConditions(
  state: EvieReasoningState,
): Promise<Partial<EvieReasoningUpdate>> {
  const conditions = await _conditionLlm.invoke([
    _DETECTION_SYSTEM,
    new HumanMessage(state.player_input),
  ]);
  return { detected_conditions: conditions };
}

function applyTransitions(state: EvieReasoningState): Partial<EvieReasoningUpdate> {
  const c = state.detected_conditions as PlayerInputConditions;
  const milestones: EvieMilestone[] = [...state.milestones];

  let trust_level = state.trust_level;
  let fear_level = state.fear_level;
  let guilt_pressure = state.guilt_pressure;
  let suspicion_level = state.suspicion_level;
  let player_has_offered_protection = state.player_has_offered_protection;
  let player_has_found_blackmail_photos = state.player_has_found_blackmail_photos;
  let player_has_shown_locket = state.player_has_shown_locket;
  let player_has_mentioned_brennan = state.player_has_mentioned_brennan;
  let player_knows_evie_was_at_alley = state.player_knows_evie_was_at_alley;
  let player_has_accused_evie = state.player_has_accused_evie;
  let player_has_revealed_murder_weapon = state.player_has_revealed_murder_weapon;
  let critical_info_revealed = state.critical_info_revealed;
  let final_clue_revealed = state.final_clue_revealed;

  const addMilestone = (m: EvieMilestone): void => {
    if (!milestones.includes(m)) milestones.push(m);
  };

  if (c.mentions_empathy) {
    trust_level += 1;
    addMilestone("empathy");
  }

  if (c.offers_protection) {
    trust_level += 2;
    fear_level -= 1;
    player_has_offered_protection = true;
    addMilestone("protection_offer");
  }

  if (c.shows_blackmail_evidence) {
    player_has_found_blackmail_photos = true;
    guilt_pressure += 2;
    addMilestone("blackmail_proof");
  }

  if (c.shows_locket) {
    player_has_shown_locket = true;
    guilt_pressure += 2;
    addMilestone("locket_evidence");
  }

  if (c.mentions_brennan) {
    player_has_mentioned_brennan = true;
    fear_level += 2;
    addMilestone("brennan_connection");
  }

  if (c.proves_alley_presence) {
    player_knows_evie_was_at_alley = true;
    guilt_pressure += 2;
    addMilestone("alley_presence");
  }

  if (c.accuses_evie) {
    player_has_accused_evie = true;
    suspicion_level += 2;
    addMilestone("direct_accusation");
  }

  if (c.mentions_murder_weapon_points_to_brennan) {
    player_has_revealed_murder_weapon = true;
    addMilestone("murder_weapon");
  }

  const guarded_ready = _hasAny(milestones, [
    "empathy",
    "protection_offer",
    "blackmail_proof",
    "locket_evidence",
    "alley_presence",
  ]);
  const critical_ready =
    _countAny(milestones, [
      "protection_offer",
      "blackmail_proof",
      "locket_evidence",
      "brennan_connection",
      "alley_presence",
      "murder_weapon",
    ]) >= 2 ||
    _hasAll(milestones, ["direct_accusation", "blackmail_proof", "alley_presence"]);
  const final_ready =
    (critical_info_revealed && player_has_offered_protection) ||
    (critical_info_revealed &&
      player_has_mentioned_brennan &&
      player_has_found_blackmail_photos);

  const next_node = _transition(
    state.current_node ?? "POLITE_MASK",
    c,
    guarded_ready,
    critical_ready,
  );

  if (
    next_node === "BROKEN_TRUSTING" ||
    next_node === "PROTECTED_WITNESS" ||
    next_node === "CORNERED_CONFESSION"
  ) {
    critical_info_revealed = true;
  }
  if (next_node === "PROTECTED_WITNESS" && final_ready) {
    final_clue_revealed = true;
  }

  return {
    current_node: next_node,
    trust_level,
    fear_level,
    guilt_pressure,
    suspicion_level,
    milestones,
    player_has_offered_protection,
    player_has_found_blackmail_photos,
    player_has_shown_locket,
    player_has_mentioned_brennan,
    player_knows_evie_was_at_alley,
    player_has_accused_evie,
    player_has_revealed_murder_weapon,
    critical_info_revealed,
    final_clue_revealed,
  };
}

function _transition(
  current: EvieNode,
  c: PlayerInputConditions,
  guarded_ready: boolean,
  critical_ready: boolean,
): EvieNode {
  switch (current) {
    case "POLITE_MASK":
      if (c.accuses_evie) return "DEFENSIVE_DENIAL";
      if (c.threatens_arrest) return "PANICKED_RESISTANCE";
      if (c.presents_concrete_evidence) return "EVIDENCE_PRESSURE";
      if (c.asks_about_victor_relationship) return "BITTER_REMEMBERING";
      if (c.mentions_empathy) return "WARY_SOFTENING";
      return "POLITE_MASK";

    case "WARY_SOFTENING":
      if (c.mentions_brennan) return "FEAR_SPIKE";
      if (c.accuses_evie) return "DEFENSIVE_DENIAL";
      if (guarded_ready) return "GUARDED_DISCLOSURE";
      return "POLITE_MASK";

    case "BITTER_REMEMBERING":
      if (c.mocks_or_dismisses_evie) return "COLD_SHUTDOWN";
      if (guarded_ready) return "GUARDED_DISCLOSURE";
      return "WARY_SOFTENING";

    case "DEFENSIVE_DENIAL":
      if (critical_ready) return "CORNERED_CONFESSION";
      if (c.presents_concrete_evidence) return "EVIDENCE_PRESSURE";
      if (c.offers_protection) return "WARY_SOFTENING";
      if (c.repeated_accusation_without_evidence) return "COLD_SHUTDOWN";
      return "DEFENSIVE_DENIAL";

    case "EVIDENCE_PRESSURE":
      if (c.mentions_brennan) return "FEAR_SPIKE";
      if (critical_ready && c.accuses_evie) return "CORNERED_CONFESSION";
      if (critical_ready) return "BROKEN_TRUSTING";
      if (guarded_ready) return "GUARDED_DISCLOSURE";
      return "EVIDENCE_PRESSURE";

    case "GUARDED_DISCLOSURE":
      if (critical_ready) return "BROKEN_TRUSTING";
      if (c.mentions_brennan) return "FEAR_SPIKE";
      if (c.threatens_arrest) return "PANICKED_RESISTANCE";
      if (c.harsh_accusation) return "DEFENSIVE_DENIAL";
      return "GUARDED_DISCLOSURE";

    case "FEAR_SPIKE":
      if (c.offers_protection) return "PROTECTED_WITNESS";
      if (critical_ready) return "BROKEN_TRUSTING";
      if (c.threatens_or_mocks) return "COLD_SHUTDOWN";
      return "GUARDED_DISCLOSURE";

    case "BROKEN_TRUSTING":
      if (c.offers_protection) return "PROTECTED_WITNESS";
      if (c.accuses_evie) return "CORNERED_CONFESSION";
      return "BROKEN_TRUSTING";

    case "PROTECTED_WITNESS":
      if (c.suggests_public_brennan_confrontation) return "FEAR_SPIKE";
      if (c.betrays_or_threatens_evie) return "COLD_SHUTDOWN";
      return "PROTECTED_WITNESS";

    case "CORNERED_CONFESSION":
      if (c.offers_protection) return "PROTECTED_WITNESS";
      if (c.stops_accusing_and_asks_facts) return "BROKEN_TRUSTING";
      return "CORNERED_CONFESSION";

    case "COLD_SHUTDOWN":
      if (c.apologizes_or_softens) return "WARY_SOFTENING";
      if (c.presents_concrete_evidence) return "EVIDENCE_PRESSURE";
      return "COLD_SHUTDOWN";

    case "PANICKED_RESISTANCE":
      if (c.offers_protection) return "FEAR_SPIKE";
      if (critical_ready) return "CORNERED_CONFESSION";
      if (guarded_ready) return "GUARDED_DISCLOSURE";
      return "PANICKED_RESISTANCE";

    default:
      return current;
  }
}

export const evieReasoningSubgraph = new StateGraph(EvieReasoningStateAnnotation)
  .addNode("detect_conditions", detectConditions)
  .addNode("apply_transitions", applyTransitions)
  .addEdge(START, "detect_conditions")
  .addEdge("detect_conditions", "apply_transitions")
  .compile();
