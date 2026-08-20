function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

/** Cards com as métricas reais de /evaluation-summary (src/evaluate.py) — sem cálculo no frontend. */
export default function MetricsPanel({ summary }) {
  if (!summary) {
    return <div style={{ fontSize: 12.5, color: "var(--text-faint)" }}>rode o replay ou aguarde o carregamento…</div>;
  }
  const m = summary.test_metrics_main_model || {};
  const items = [
    { label: "PR-AUC (teste)", value: fmt(m.average_precision) },
    { label: "ROC-AUC (teste)", value: fmt(m.roc_auc) },
    { label: "CV PR-AUC (média)", value: `${fmt(summary.cv_pr_auc_mean_lightgbm)} ± ${fmt(summary.cv_pr_auc_std_lightgbm)}` },
    { label: "gap generalização temporal", value: fmt(summary.temporal_generalization_gap_lightgbm) },
  ];
  return (
    <div>
      {items.map((it) => (
        <div className="stat-row" key={it.label}>
          <span className="label">{it.label}</span>
          <span className="value mono">{it.value}</span>
        </div>
      ))}
    </div>
  );
}
