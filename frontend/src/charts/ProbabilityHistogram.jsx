/** Distribuição das probabilidades das predições reais na janela — a "forma" que o
 * time de risco acompanha: a maioria deve ficar perto de 0 (legítima); qualquer volume
 * crescendo perto do threshold é sinal de atenção antes mesmo do PSI se mover. */
export default function ProbabilityHistogram({ events, threshold = 0.724, bins = 10, width = 280, height = 90 }) {
  const win = events || [];
  const counts = new Array(bins).fill(0);
  for (const e of win) {
    const idx = Math.min(bins - 1, Math.floor(e.fraud_probability * bins));
    counts[idx] += 1;
  }
  const max = Math.max(1, ...counts);
  const barW = width / bins;
  const thresholdBin = Math.min(bins - 1, Math.floor(threshold * bins));

  if (win.length === 0) {
    return <div style={{ fontSize: 11, color: "var(--text-faint)" }}>sem predições na janela ainda</div>;
  }

  return (
    <svg width={width} height={height + 16} style={{ display: "block" }}>
      {counts.map((c, i) => {
        const h = (c / max) * height;
        const isAboveThreshold = i >= thresholdBin;
        return (
          <rect
            key={i}
            x={i * barW + 1}
            y={height - h}
            width={barW - 2}
            height={h}
            fill={isAboveThreshold ? "var(--danger)" : "var(--ok)"}
            opacity={0.85}
          />
        );
      })}
      <line x1={thresholdBin * barW} x2={thresholdBin * barW} y1={0} y2={height} stroke="var(--accent)" strokeDasharray="3 3" />
      <text x={2} y={height + 12} fontSize="9" className="mono" fill="var(--text-faint)">
        0.0
      </text>
      <text x={width - 18} y={height + 12} fontSize="9" className="mono" fill="var(--text-faint)">
        1.0
      </text>
    </svg>
  );
}
