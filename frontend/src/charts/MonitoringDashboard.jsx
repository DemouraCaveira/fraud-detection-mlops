/**
 * Um pequeno painel "estilo Datadog": não é só um gráfico solto, é o tipo de tela que um
 * time de risco realmente olharia — vários indicadores lado a lado, cada um com o
 * estado (ok/atenção) visível sem precisar interpretar nada. Tudo computado a partir de
 * eventos reais recebidos via /events/live, numa janela deslizante.
 */
export default function MonitoringDashboard({ events, windowSize = 40 }) {
  const win = (events || []).slice(-windowSize);
  const n = win.length;

  const blocked = win.filter((e) => e.decision === "BLOQUEAR").length;
  const blockRate = n ? (blocked / n) * 100 : null;
  const avgLatency = n ? win.reduce((s, e) => s + e.latency_ms, 0) / n : null;
  const sorted = [...win].sort((a, b) => a.latency_ms - b.latency_ms);
  const p95Latency = n ? sorted[Math.floor(0.95 * (n - 1))]?.latency_ms : null;

  let throughputPerMin = null;
  if (n >= 2) {
    const spanS = (win[win.length - 1].ts - win[0].ts) || 1;
    throughputPerMin = (n / spanS) * 60;
  }

  const tiles = [
    { label: "predições (janela)", value: n ? n : "—", tone: "neutral" },
    { label: "taxa de bloqueio", value: blockRate == null ? "—" : `${blockRate.toFixed(1)}%`, tone: blockRate > 15 ? "warn" : "ok" },
    { label: "latência média", value: avgLatency == null ? "—" : `${avgLatency.toFixed(2)} ms`, tone: avgLatency > 5 ? "warn" : "ok" },
    { label: "latência p95", value: p95Latency == null ? "—" : `${p95Latency.toFixed(2)} ms`, tone: p95Latency > 8 ? "warn" : "ok" },
    { label: "throughput", value: throughputPerMin == null ? "—" : `${throughputPerMin.toFixed(0)}/min`, tone: "neutral" },
  ];

  const TONE_COLOR = { ok: "var(--ok)", warn: "var(--danger)", neutral: "var(--text)" };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(96px, 1fr))", gap: 8 }}>
      {tiles.map((t) => (
        <div key={t.label} style={{ background: "var(--surface-alt)", borderRadius: 8, padding: "8px 10px", border: "1px solid var(--border)" }}>
          <div style={{ fontSize: 9, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.04em" }}>{t.label}</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: TONE_COLOR[t.tone], marginTop: 2 }}>
            {t.value}
          </div>
        </div>
      ))}
    </div>
  );
}
