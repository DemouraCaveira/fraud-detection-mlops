/**
 * Duas perguntas que sempre voltam: "como eu versiono o artefato?" e "o que
 * exatamente dispara um retreino?". As respostas aqui não são teóricas — vêm de
 * /model-info (a ficha real deste modelo) e /drift-report (o PSI real medido).
 */
export default function VersioningPanel({ modelInfo, driftReport }) {
  if (!modelInfo) {
    return <div style={{ fontSize: 11.5, color: "var(--text-faint)" }}>carregando ficha do modelo…</div>;
  }

  const rows = driftReport?.real_drift_treino_vs_teste || [];
  const maxPsi = rows.reduce((m, r) => Math.max(m, r.psi ?? 0), 0);
  const thresholds = driftReport?.thresholds || { moderado: 0.1, significativo: 0.25 };
  const armed = maxPsi >= thresholds.significativo;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div>
        <div style={{ fontSize: 10, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>
          versão do artefato em produção
        </div>
        <div className="mono" style={{ fontSize: 12, color: "var(--text)", lineHeight: 1.7 }}>
          <div>
            <span style={{ color: "var(--text-faint)" }}>id da versão </span>= treinado_em → <b>{modelInfo.treinado_em}</b>
          </div>
          <div>
            <span style={{ color: "var(--text-faint)" }}>fingerprint </span>= {modelInfo.n_arvores} árvores, hiperparâmetros do Optuna (json versionado junto)
          </div>
        </div>
        <p style={{ fontSize: 11, color: "var(--text-muted)", margin: "4px 0 0" }}>
          Não existe um "número de versão" arbitrário — a versão real é o par <b>(artefato .joblib, metadados json)</b> que sai
          junto do mesmo run de treino. Promover = mover esse par intacto de hom para prod; nunca editar um em produção.
        </p>
      </div>

      <div style={{ borderTop: "1px dashed var(--border)", paddingTop: 10 }}>
        <div style={{ fontSize: 10, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>
          gatilho de retreino
        </div>
        <div
          className="mono"
          style={{
            fontSize: 12, padding: "8px 10px", borderRadius: 6,
            background: armed ? "var(--danger-soft)" : "var(--ok-soft)",
            color: armed ? "var(--danger)" : "var(--ok)",
          }}
        >
          {armed ? "⚠ ativo agora" : "calmo agora"} — PSI máx. real = {maxPsi.toFixed(3)} (limiar {thresholds.significativo})
        </div>
        <p style={{ fontSize: 11, color: "var(--text-muted)", margin: "6px 0 0" }}>
          Dois gatilhos independentes, o que vier primeiro: <b>agendado</b> (ex.: toda semana, mesmo sem drift) ou{" "}
          <b>por desvio</b> (PSI/KS agregado cruza {thresholds.significativo}, mesmo fora do agendamento). Nenhuma transação
          individual decide isso sozinha.
        </p>
      </div>
    </div>
  );
}
