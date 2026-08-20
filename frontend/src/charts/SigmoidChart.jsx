import { useMemo } from "react";
import { motion } from "framer-motion";

/**
 * A função que o modelo realmente usa para transformar "soma das árvores" (score bruto,
 * pode ser qualquer número) em "probabilidade" (0 a 1): sigmoid(z) = 1/(1+e^-z). Plota a
 * curva real e marca onde ESTA transação caiu nela — a mesma pergunta que aparece toda
 * vez que alguém pergunta "o que esse número de 0 a 1 significa".
 */
export default function SigmoidChart({ rawScore, probability, threshold = 0.724, width = 320, height = 180 }) {
  const margin = { top: 14, right: 16, bottom: 28, left: 38 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const domain = 8; // z de -8 a 8 cobre a saturação da sigmoide nos dois lados
  const x = (z) => margin.left + ((z + domain) / (2 * domain)) * innerW;
  const y = (p) => margin.top + (1 - p) * innerH;
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  const path = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const z = -domain + (2 * domain * i) / 60;
      pts.push(`${i === 0 ? "M" : "L"} ${x(z).toFixed(1)} ${y(sigmoid(z)).toFixed(1)}`);
    }
    return pts.join(" ");
  }, [width, height]);

  const clampedZ = rawScore == null ? null : Math.max(-domain, Math.min(domain, rawScore));
  const pointColor = probability >= threshold ? "var(--danger)" : "var(--ok)";

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {/* eixos + grade */}
      <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="var(--border-strong)" />
      <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="var(--border-strong)" />
      {[0, 0.25, 0.5, 0.75, 1].map((p) => (
        <g key={p}>
          <line x1={margin.left} x2={width - margin.right} y1={y(p)} y2={y(p)} stroke="var(--border)" strokeDasharray="2 3" />
          <text x={margin.left - 6} y={y(p)} dy="0.32em" textAnchor="end" fontSize="9.5" className="mono" fill="var(--text-faint)">
            {p.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={width / 2} y={height - 6} textAnchor="middle" fontSize="9.5" className="mono" fill="var(--text-faint)">
        score bruto (z) — soma das árvores
      </text>

      {/* linha do threshold */}
      <line x1={margin.left} x2={width - margin.right} y1={y(threshold)} y2={y(threshold)} stroke="var(--accent)" strokeDasharray="4 3" strokeWidth={1.2} />
      <text x={width - margin.right} y={y(threshold) - 4} textAnchor="end" fontSize="9" className="mono" fill="var(--accent-text)">
        threshold {threshold}
      </text>

      {/* a curva sigmoide real */}
      <path d={path} fill="none" stroke="var(--text-muted)" strokeWidth={1.8} />

      {/* onde esta transação caiu */}
      {clampedZ !== null && (
        <>
          <line x1={x(clampedZ)} x2={x(clampedZ)} y1={y(probability)} y2={height - margin.bottom} stroke={pointColor} strokeDasharray="3 3" strokeWidth={1} />
          <motion.circle
            key={rawScore}
            initial={{ r: 0 }}
            animate={{ r: 5.5 }}
            transition={{ duration: 0.35, type: "spring" }}
            cx={x(clampedZ)}
            cy={y(probability)}
            fill={pointColor}
            stroke="var(--surface)"
            strokeWidth={2}
          />
          <text x={x(clampedZ)} y={y(probability) - 12} textAnchor="middle" fontSize="10.5" fontWeight="700" className="mono" fill={pointColor}>
            {probability.toFixed(3)}
          </text>
        </>
      )}
    </svg>
  );
}
