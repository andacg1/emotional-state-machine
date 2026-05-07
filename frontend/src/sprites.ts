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
  POLITE_MASK:        "neutral",
  WARY_SOFTENING:     "trusting",
  BITTER_REMEMBERING: "neutral",
  DEFENSIVE_DENIAL:   "hostile",
  EVIDENCE_PRESSURE:  "neutral",
  GUARDED_DISCLOSURE: "neutral",
  FEAR_SPIKE:         "neutral",
  BROKEN_TRUSTING:    "trusting",
  PROTECTED_WITNESS:  "trusting",
  CORNERED_CONFESSION:"hostile",
  COLD_SHUTDOWN:      "hostile",
  PANICKED_RESISTANCE:"hostile",
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

// Composed, smirking — the performance face
const POLITE_MASK = `
     ___________
    /    ___    \\
   | ( -   - )  |
   |   \\ ~ /   |
    \\   '---'  /
   __|_________|__
  /   )       (   \\
 | . /  ~*~*~  \\ . |
 |  /___________\\  |
  \\/   |     |   \\/
       |     |
      _|_____|_
     /_________\\`.trim();

// Cautious, tired — mask slipping just slightly
const WARY_SOFTENING = `
      __________
     /   _  _   \\
    |  ( .  . )  |
    |   \\ - /   |
     \\   '---' /
   __|_________|__
  /  ,)       (,  \\
 | . / ~~~~~*~~ \\ . |
 |  /___________\\  |
  \\/   |     |   \\/
       |     |
      _|_____|_
     /_________\\`.trim();

// Sad, looking into the distance — turned slightly
const BITTER_REMEMBERING = `
      __________
     /   _ .    \\
    |  ( .  ,)  |
    |    \\ _ /  |
     \\    '---' /
   __|__________|_
  / ,(          \\  \\
 |./ ~~~~~~~~~~~~ \\ |
 | /______________ \\|
  /    |      |    \\
       |      |
      _|______|_
     /__________\\`.trim();

// Angry, defiant — arms crossed, jaw set
const DEFENSIVE_DENIAL = `
      __________
     /  _     _  \\
    | (>.<  >.<)  |
    |    \\ = /   |
     \\    '---' /
   __|___________|__
  / .-X         X-. \\
 | . /===========\\ . |
 |  /=============\\  |
  \\/    |     |    \\/
   \\____|     |____/
       _|     |_
      /__________\\`.trim();

// Tense, calculating — leaning forward, narrowed eyes
const EVIDENCE_PRESSURE = `
      __________
     /  ___  _  \\
    | (>._  ._<) |
    |    \\ _ /  |
     \\    '---' /
    _|___________|_
   /  |           | \\
  | . /  - - - -  \\ . |
  | /~~~~~~~~~~~~~~~\\ |
   /     |     |     \\
         |     |
        _|_____|_
       /__________\\`.trim();

// Serious, reluctant — leaning in across a table
const GUARDED_DISCLOSURE = `
      __________
     /   _ _    \\
    |  ( - - )  |
    |    \\ o /  |
     \\    '---' /
    __|_________|__
   /  /           \\ \\
  | ./  ~~~~~~~~~  \\. |
  |//_______________\\\\|
   /    |       |    \\
 _/     |       |     \\_
/________________________\\`.trim();

// Frightened — wide eyes, hunched, shaking
const FEAR_SPIKE = `
      __________
     / .  O  O  . \\
    |  ( O   O )  |
    |    \\ ! /   |
     \\    '---'  /
  ____|___________|____
 /  ,(               ), \\
|  /   * * * * * * *  \\ |
| / * * * * * * * * * \\ |
 \\/  ,---|       |---, \\/
      \\  |       |  /
       \\_|_______|_/`.trim();

// Fragile, honest — shoulders dropped, tear-streaked
const BROKEN_TRUSTING = `
      __________
     /   _ _    \\
    |  ( ' ' )  |
    |    \\ . /  |
     \\  . '---' /
    __|_________|__
   / ,(           ), \\
  |. /  . . . . .  \\. |
  | /_______________\\ |
   /    |       |    \\
        |       |
       _|_______|_
      /___________\\`.trim();

// Earnest, scared but willing — reaching forward
const PROTECTED_WITNESS = `
      __________
     /   _ _    \\
    |  ( ^ ^ )  |
    |    \\ - /  |
     \\    '---' /
    _|___________|____
   / /           |>   \\
  |./  ~~~~~~~~~  >    |
  |/_______________>   |
   /    |       /      \\
        |      /
       _|_____/
      /________\\`.trim();

// Desperate — cornered against a wall, breathless
const CORNERED_CONFESSION = `
      __________
     / . _ _ . . \\
    | ( >_< )  . |
    |   \\ ! /    |
     \\ . '---' . /
  ____|___________|____
 /  ,( XXXXXXXXXX ), \\
|  / XXXXXXXXXXXXX \\ |
| /XXXXXXXXXXXXXXXX\\ |
 \\/ .|       |. \\/
   /||       ||\\
  / ||       || \\
 /__|_________|__\\`.trim();

// Ice cold — turned three-quarters away, completely closed
const COLD_SHUTDOWN = `
      _________
     /  .   _  \\
    | .  ( - ) . |
    |     \\ /   |
     \\ .  '--' /
   __|___________|
  /  /            \\
 | ./  _________  |
 |./  |         | |
  /   |_________|  \\
 /    |         |   \\
/_____|_________|____\\`.trim();

// Frantic, unraveling — everything at once
const PANICKED_RESISTANCE = `
      ___________
     / .  _ _  . \\
    | . (>_<) . . |
    | .  \\ ! / .  |
     \\ . '---' . /
  ___|_____________|___
 /  /! ! ! ! ! ! !\\ \\
| ./! ! ! ! ! ! ! !\\ .|
|./! ! ! ! ! ! ! ! !\\ |
 /! |         | ! ! \\
/!  |         |  ! !\\
\\!__|_________|__! !/
 \\___________________/`.trim();

export const SPRITES: Record<EvieNode, string> = {
  POLITE_MASK,
  WARY_SOFTENING,
  BITTER_REMEMBERING,
  DEFENSIVE_DENIAL,
  EVIDENCE_PRESSURE,
  GUARDED_DISCLOSURE,
  FEAR_SPIKE,
  BROKEN_TRUSTING,
  PROTECTED_WITNESS,
  CORNERED_CONFESSION,
  COLD_SHUTDOWN,
  PANICKED_RESISTANCE,
};
