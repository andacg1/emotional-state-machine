import { useEffect, useState } from "react";
import { SPRITES, NODE_LABEL, NODE_CLUSTER, type EvieNode } from "../sprites";

interface Props {
  node: EvieNode;
  milestones: string[];
  trustLevel: number;
  fearLevel: number;
}

const MILESTONE_LABELS: Record<string, string> = {
  empathy:          "empathy shown",
  protection_offer: "protection offered",
  blackmail_proof:  "blackmail proven",
  locket_evidence:  "locket found",
  brennan_connection: "brennan named",
  alley_presence:   "alley confirmed",
  murder_weapon:    "weapon revealed",
  direct_accusation:"accused",
};

export function AsciiSprite({ node, milestones, trustLevel, fearLevel }: Props) {
  const [displayNode, setDisplayNode] = useState<EvieNode>(node);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    if (node === displayNode) return;
    setFading(true);
    const t = setTimeout(() => {
      setDisplayNode(node);
      setFading(false);
    }, 350);
    return () => clearTimeout(t);
  }, [node, displayNode]);

  const cluster = NODE_CLUSTER[displayNode];
  const clusterClass = `cluster-${cluster}`;

  return (
    <div className="sprite-panel">
      <div className="sprite-header">
        <span className="sprite-name">EVELYN "EVIE" MARLOWE</span>
        <span className="sprite-location">Blue Dahlia Club · LA · 1952</span>
      </div>

      <div className={`sprite-frame ${fading ? "fading" : ""}`}>
        <div className="scanlines" />
        <img
          className="sprite-img"
          src={SPRITES[displayNode]}
          alt={NODE_LABEL[displayNode]}
        />
      </div>

      <div className={`node-badge ${clusterClass}`}>
        {NODE_LABEL[displayNode].toUpperCase()}
      </div>

      <div className="stat-bars">
        <StatBar label="TRUST" value={trustLevel} max={5} color="#5a9e6f" />
        <StatBar label="FEAR" value={fearLevel} max={5} color="#c0513a" />
      </div>

      {milestones.length > 0 && (
        <div className="milestones">
          <div className="milestones-label">EVIDENCE OBTAINED</div>
          <div className="milestone-list">
            {milestones.map((m) => (
              <span key={m} className="milestone-badge">
                {MILESTONE_LABELS[m] ?? m}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = Math.min(Math.max(value / max, 0), 1) * 100;
  return (
    <div className="stat-row">
      <span className="stat-label">{label}</span>
      <div className="stat-track">
        <div className="stat-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="stat-value">{value}/{max}</span>
    </div>
  );
}
