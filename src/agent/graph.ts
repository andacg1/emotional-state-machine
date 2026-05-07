import "dotenv/config";

import {
  MemorySaver,
  MessagesAnnotation,
  MessagesZodState,
  START,
  StateGraph
} from "@langchain/langgraph";

import { EvieMarlowe, StateAnnotation } from "./evie_marlowe.js";

const checkpointer = new MemorySaver();
const evie = new EvieMarlowe();
const {
  initializeEvie,
  politeMask,
  warySoftening,
  bitterRemembering,
  defensiveDenial,
  evidencePressure,
  guardedDisclosure,
  fearSpike,
  brokenTrusting,
  protectedWitness,
  corneredConfession,
  coldShutdown,
  panickedResistance,
} = evie;

export const graph = new StateGraph(StateAnnotation)
  .addNode("INITIALIZE", initializeEvie, {
    ends: [
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
    ],
  })
  .addNode("POLITE_MASK", politeMask)
  .addNode("WARY_SOFTENING", warySoftening)
  .addNode("BITTER_REMEMBERING", bitterRemembering)
  .addNode("DEFENSIVE_DENIAL", defensiveDenial)
  .addNode("EVIDENCE_PRESSURE", evidencePressure)
  .addNode("GUARDED_DISCLOSURE", guardedDisclosure)
  .addNode("FEAR_SPIKE", fearSpike)
  .addNode("BROKEN_TRUSTING", brokenTrusting)
  .addNode("PROTECTED_WITNESS", protectedWitness)
  .addNode("CORNERED_CONFESSION", corneredConfession)
  .addNode("COLD_SHUTDOWN", coldShutdown)
  .addNode("PANICKED_RESISTANCE", panickedResistance)
  .addEdge(START, "INITIALIZE")
  .addEdge("INITIALIZE", "POLITE_MASK")
  ;

// Patch schema so Studio detects the messages key and enables the Chat tab.
// Annotation.Root() doesn't support jsonSchemaExtra, so we attach a
// duck-typed schema definition that mirrors all fields and adds langgraph_type.
(graph as any)._schemaRuntimeDefinition = {
  getJsonSchema: () => ({
    type: 'object',
    properties: {
      messages: { type: 'array', items: {}, langgraph_type: 'messages', default: [] },
      // ... include your other state fields here so they show up in Studio
    },
  }),
  getInputJsonSchema: () => ({
    type: 'object',
    properties: {
      messages: { type: 'array', items: {}, langgraph_type: 'messages' },
      // ... same fields without defaults
    },
  }),
};

graph.compile({ checkpointer, name: "NPC Graph" })
