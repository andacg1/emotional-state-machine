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

export type NodeCluster = "trusting" | "hostile" | "neutral";

export const NODE_CLUSTER: Record<EvieNode, NodeCluster> = {
  POLITE_MASK:         "neutral",
  WARY_SOFTENING:      "trusting",
  BITTER_REMEMBERING:  "neutral",
  DEFENSIVE_DENIAL:    "hostile",
  EVIDENCE_PRESSURE:   "neutral",
  GUARDED_DISCLOSURE:  "neutral",
  FEAR_SPIKE:          "neutral",
  BROKEN_TRUSTING:     "trusting",
  PROTECTED_WITNESS:   "trusting",
  CORNERED_CONFESSION: "hostile",
  COLD_SHUTDOWN:       "hostile",
  PANICKED_RESISTANCE: "hostile",
};

export const NODE_LABEL: Record<EvieNode, string> = {
  POLITE_MASK:         "Polite Mask",
  WARY_SOFTENING:      "Wary Softening",
  BITTER_REMEMBERING:  "Bitter Remembering",
  DEFENSIVE_DENIAL:    "Defensive Denial",
  EVIDENCE_PRESSURE:   "Evidence Pressure",
  GUARDED_DISCLOSURE:  "Guarded Disclosure",
  FEAR_SPIKE:          "Fear Spike",
  BROKEN_TRUSTING:     "Broken Trusting",
  PROTECTED_WITNESS:   "Protected Witness",
  CORNERED_CONFESSION: "Cornered Confession",
  COLD_SHUTDOWN:       "Cold Shutdown",
  PANICKED_RESISTANCE: "Panicked Resistance",
};

// ─── Sprite images ────────────────────────────────────────────────────────────
import politeMask         from "../assets/EvieMarlowe-PoliteMask.png";
import warySoftening      from "../assets/EvieMarlowe-WarySoftening.png";
import bitterRemembering  from "../assets/EvieMarlowe-BitterRemembering.png";
import defensiveDenial    from "../assets/EvieMarlowe-DefensiveDenial.png";
import evidencePressure   from "../assets/EvieMarlowe-EvidencePressure.png";
import guardedDisclosure  from "../assets/EvieMarlowe-GuardedDisclosure.png";
import fearSpike          from "../assets/EvieMarlowe-FearSpike.png";
import brokenTrusting     from "../assets/EvieMarlowe-BrokenTrusting.png";
import protectedWitness   from "../assets/EvieMarlowe-ProtectedWitness.png";
import corneredConfession from "../assets/EvieMarlowe-CorneredConfession.png";
import coldShutdown       from "../assets/EvieMarlowe-ColdShutdown.png";
import panickedResistance from "../assets/EvieMarlowe-PanickedResistance.png";

export const SPRITES: Record<EvieNode, string> = {
  POLITE_MASK:         politeMask,
  WARY_SOFTENING:      warySoftening,
  BITTER_REMEMBERING:  bitterRemembering,
  DEFENSIVE_DENIAL:    defensiveDenial,
  EVIDENCE_PRESSURE:   evidencePressure,
  GUARDED_DISCLOSURE:  guardedDisclosure,
  FEAR_SPIKE:          fearSpike,
  BROKEN_TRUSTING:     brokenTrusting,
  PROTECTED_WITNESS:   protectedWitness,
  CORNERED_CONFESSION: corneredConfession,
  COLD_SHUTDOWN:       coldShutdown,
  PANICKED_RESISTANCE: panickedResistance,
};
