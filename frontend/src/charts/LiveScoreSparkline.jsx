/** Mini-versão da LiveMonitorChart, para caber dentro do nó "Monitoramento" do grafo. */
export default function LiveScoreSparkline({ events, threshold = 0.724, width = 130, height = 26, windowSize = 20 }) {
  const data = (events || []).slice(-windowSize);
  if (data.length < 2) {
    return <div style={{ fontSize: 9, color: "var(--text-faint)", height }}>aguardando /predict…</div>;
  }
  const stepX = width / (windowSize - 1);
  const offset = windowSize - data.length;
  const y = (p) => height - p * height;
  const points = data.map((d, i) => `${(offset + i) * stepX},${y(d.fraud_probability)}`).join(" ");

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <line x1={0} x2={width} y1={y(threshold)} y2={y(threshold)} stroke="var(--accent)" strokeDasharray="2 2" strokeWidth={1} />
      <polyline points={points} fill="none" stroke="var(--text-faint)" strokeWidth={1} />
      {data.map((d, i) => (
        <circle
          key={d.ts}
          cx={(offset + i) * stepX}
          cy={y(d.fraud_probability)}
          r={i === data.length - 1 ? 2.6 : 1.6}
          fill={d.decision === "BLOQUEAR" ? "var(--danger)" : "var(--ok)"}
        />
      ))}
    </svg>
  );
}
