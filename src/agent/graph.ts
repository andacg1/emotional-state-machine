import "dotenv/config";

import { MemorySaver, START, StateGraph } from "@langchain/langgraph";

import {
  StateAnnotation,
  bitterRemembering,
  brokenTrusting,
  coldShutdown,
  corneredConfession,
  defensiveDenial,
  evidencePressure,
  fearSpike,
  guardedDisclosure,
  initializeEvie,
  panickedResistance,
  politeMask,
  protectedWitness,
  warySoftening,
} from "./evie_marlowe.js";

const checkpointer = new MemorySaver();

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
  .compile({ checkpointer, name: "NPC Graph" });
