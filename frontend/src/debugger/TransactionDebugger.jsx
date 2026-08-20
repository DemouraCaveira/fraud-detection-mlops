import { useState } from "react";
import { motion } from "framer-motion";
import SigmoidChart from "../charts/SigmoidChart";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";

function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

const STEP_META = [
  {
    key: "raw",
    title: "1. Dado bruto recebido",
    file: "POST /predict",
    code: 'transaction = {\n  "Time": <segundos>,\n  "Amount": <R$>,\n  "V1..V28": [<PCA anonimizado>]\n}',
    explain: "O payload chega exatamente como viria de um sistema de pagamento — nenhum processamento ainda.",
  },
  {
    key: "features",
    title: "2. Feature engineering",
    file: "src/preprocessing.py · engineer_features()",
    code: "hour = (Time % 86400) / 3600\nhour_sin = sin(2π · hour / 24)\nhour_cos = cos(2π · hour / 24)\namount_log = log1p(Amount)",
    explain: "Time e Amount viram atributos que o modelo consegue usar melhor: hora do dia em forma cíclica (sin/cos, para meia-noite ficar 'perto' de 23h) e o log do valor (para outliers não dominarem a escala).",
  },
  {
    key: "scale",
    title: "3. Normalização",
    file: "src/preprocessing.py · apply_scaler() (RobustScaler)",
    code: "Amount_scaled = (Amount - mediana_treino) / IQR_treino",
    explain: "RobustScaler em vez de StandardScaler porque Amount tem outliers extremos (transações muito altas) que distorceriam média/desvio-padrão.",
  },
  {
    key: "model",
    title: "4. Inferência — LightGBM",
    file: "models/lightgbm_fraud.joblib (202 árvores)",
    code: "z = soma_das_202_arvores(X)     # score bruto, qualquer numero\nfraud_probability = 1 / (1 + e**-z)  # sigmoide espreme p/ 0..1",
    explain: "As árvores nunca devolvem uma probabilidade — devolvem um número qualquer (soma das 202 folhas). A sigmoide é quem espreme esse número para o intervalo 0–1. Veja abaixo exatamente onde esta transação caiu na curva.",
  },
  {
    key: "shap",
    title: "5. Explicabilidade — SHAP",
    file: "shap.TreeExplainer — calculado ao vivo para ESTA predição",
    code: "shap_values = explainer(X)\ntop_factors = maiores |shap_value|",
    explain: "Quais atributos mais empurraram a probabilidade para cima (fraude) ou para baixo (legítima) nesta transação específica — não a média do modelo, o caso real.",
  },
  {
    key: "decision",
    title: "6. Decisão final",
    file: "deploy/api.py · predict()",
    code: "decision = 'BLOQUEAR' if fraud_probability >= threshold else 'APROVAR'",
    explain: "O threshold (0.724) não é 0.5 — foi escolhido otimizando o custo real de falso positivo vs. falso negativo (ver src/evaluate.py).",
  },
  {
    key: "deploy",
    title: "7. Deploy — onde \"o modelo\" está, de verdade",
    file: "main.py · deploy/api.py",
    code: 'model = joblib.load("models/lightgbm_fraud.joblib")  # 1x, na subida\nuvicorn.run("deploy.api:app", port=8001)           # fica no ar\n\n# a transação de cima chegou aqui: POST /predict',
    explain: '"O modelo" não é um serviço misterioso rodando em algum lugar — é um arquivo de ~poucos KB com estas 202 árvores. Ele é carregado UMA vez quando o processo sobe (main.py) e fica em memória; toda predição real (inclusive esta) reusa esse mesmo objeto, sem reler o disco a cada chamada.',
  },
  {
    key: "monitor",
    title: "8. Monitoramento — o que esta predição alimenta",
    file: "deploy/api.py → /events/live · monitoring/drift_monitor.py",
    code: "_live_events.append({fraud_probability, decision, latency_ms, ts})\n\n# em LOTE, nao por transacao:\nPSI = comparar(distribuicao_hoje, distribuicao_treino)",
    explain: "Esta predição específica acabou de ser publicada em tempo real — é o ponto que aparece no gráfico de monitoramento ao vivo. Mas o PSI/KS que decide se o modelo está desviando NÃO é calculado por transação — é uma comparação de distribuição, em lote, entre um período de referência e o período atual.",
  },
  {
    key: "retrain",
    title: "9. Retreino — quando isso realmente dispara",
    file: "referência — run_pipeline.py, disparado por PSI real",
    code: "if psi_maximo >= 0.25:\n    disparar_retreino()   # na hora\nelse:\n    aguardar_agendamento() # ex.: semanal",
    explain: 'Não existe "retreinar depois desta transação" — o gatilho é sempre o desvio AGREGADO cruzando o limiar, ou um agendamento fixo. Uma transação isolada nunca dispara nada sozinha.',
  },
];

/**
 * Inspirado no Python Tutor (pythontutor.com): passo a passo, com um botão "próximo",
 * dissecando o que uma transação real passa dentro da API — cada passo usa dado real já
 * devolvido por POST /predict (pipeline_steps, top_factors), nada é recalculado aqui.
 */
export default function TransactionDebugger() {
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [sample, setSample] = useState(null);
  const [result, setResult] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [driftReport, setDriftReport] = useState(null);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    setStep(0);
    try {
      const sampleRes = await fetch(`${API_BASE}/generate-sample`);
      const s = await sampleRes.json();
      const predictRes = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(s.transaction),
      });
      const r = await predictRes.json();
      const [infoRes, driftRes] = await Promise.all([
        fetch(`${API_BASE}/model-info`),
        fetch(`${API_BASE}/drift-report`),
      ]);
      setModelInfo(infoRes.ok ? await infoRes.json() : null);
      setDriftReport(driftRes.ok ? await driftRes.json() : null);
      setSample(s);
      setResult(r);
    } catch {
      setError("não foi possível gerar/rodar a transação — a API está no ar?");
    } finally {
      setLoading(false);
    }
  };

  const ready = sample && result;
  const meta = STEP_META[step];
  const approved = result?.decision === "APROVAR";

  const driftStatus = (() => {
    if (!driftReport) return null;
    const rows = driftReport.real_drift_treino_vs_teste || [];
    const maxPsi = rows.reduce((m, r) => Math.max(m, r.psi ?? 0), 0);
    const thresholds = driftReport.thresholds || { moderado: 0.1, significativo: 0.25 };
    const status = maxPsi >= thresholds.significativo ? "significativo" : maxPsi >= thresholds.moderado ? "moderado" : "estável";
    return { maxPsi, status, thresholds };
  })();

  const goto = (i) => setStep(Math.max(0, Math.min(STEP_META.length - 1, i)));

  return (
    <div style={{ padding: 20, maxWidth: 780, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
        <div>
          <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
            passo a passo — uma transação por dentro
          </div>
          <div style={{ fontSize: 13, color: "var(--text-faint)", marginTop: 2 }}>
            inspirado em pythontutor.com — dissecando POST /predict, com dado real
          </div>
        </div>
        <button className="btn primary" onClick={load} disabled={loading}>
          {loading ? "gerando…" : ready ? "nova transação" : "gerar transação"}
        </button>
      </div>

      {error && <div style={{ color: "var(--danger)", fontSize: 13 }}>{error}</div>}

      {!ready && !loading && !error && (
        <div className="panel" style={{ textAlign: "center", color: "var(--text-faint)", padding: 40 }}>
          clique em "gerar transação" para começar
        </div>
      )}

      {ready && (
        <>
          {/* trilha de passos, clicável — como as setas do Python Tutor */}
          <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
            {STEP_META.map((s, i) => (
              <button
                key={s.key}
                onClick={() => goto(i)}
                className="mono"
                style={{
                  flex: 1, padding: "7px 4px", fontSize: 10, borderRadius: 6, cursor: "pointer",
                  border: `1px solid ${i === step ? "var(--accent)" : "var(--border)"}`,
                  background: i < step ? "var(--ok-soft)" : i === step ? "var(--accent-soft)" : "var(--surface)",
                  color: i === step ? "var(--accent-text)" : i < step ? "var(--ok)" : "var(--text-faint)",
                }}
              >
                {i + 1}
              </button>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* passo atual — "linha em destaque" (remonta a cada step via key, animando a entrada) */}
            <motion.div
              key={meta.key}
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
              className="panel"
            >
                <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: "var(--text-faint)", marginBottom: 4 }}>{meta.file}</div>
                <div style={{ fontSize: 14.5, fontWeight: 600, marginBottom: 10 }}>{meta.title}</div>
                <pre
                  className="mono"
                  style={{
                    background: "var(--canvas-bg)", color: "#cfe0d8", padding: 12, borderRadius: 8,
                    fontSize: 11.5, lineHeight: 1.6, whiteSpace: "pre-wrap", margin: 0, marginBottom: 10,
                  }}
                >
                  {meta.code}
                </pre>
                <p style={{ fontSize: 12.5, color: "var(--text-muted)", margin: 0 }}>{meta.explain}</p>

                {step === 3 && (
                  <div style={{ marginTop: 12, display: "flex", justifyContent: "center" }}>
                    <SigmoidChart
                      rawScore={Math.log(result.fraud_probability / (1 - result.fraud_probability))}
                      probability={result.fraud_probability}
                      threshold={result.threshold_used}
                      width={300}
                      height={160}
                    />
                  </div>
                )}

                {step === 4 && (
                  <div style={{ marginTop: 12 }}>
                    {result.top_factors.map((f) => (
                      <ShapBar key={f.feature} feature={f.feature} value={f.shap_value} />
                    ))}
                  </div>
                )}

                {step === 5 && (
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    style={{
                      marginTop: 12, textAlign: "center", padding: "14px 10px", borderRadius: 8,
                      background: approved ? "var(--ok-soft)" : "var(--danger-soft)",
                      color: approved ? "var(--ok)" : "var(--danger)",
                      fontWeight: 800, fontSize: 16, letterSpacing: "0.03em",
                    }}
                  >
                    {approved ? "✓ APROVAR" : "✕ BLOQUEAR"}
                    <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, fontWeight: 500, marginTop: 4 }}>
                      {fmt(result.fraud_probability, 3)} {approved ? "<" : "≥"} {fmt(result.threshold_used, 3)}
                    </div>
                  </motion.div>
                )}

                {step === 6 && modelInfo && (
                  <div className="mono" style={{ marginTop: 12, fontSize: 11.5, display: "flex", flexDirection: "column", gap: 5 }}>
                    <StateRow label="algoritmo" value={modelInfo.algoritmo} show />
                    <StateRow label="n_arvores" value={modelInfo.n_arvores} show />
                    <StateRow label="treinado_em" value={modelInfo.treinado_em} show />
                    <StateRow label="dataset" value={modelInfo.dataset} show />
                    <StateRow label="processo servindo" value={`127.0.0.1:8001 (${API_BASE.split(":").pop()})`} show highlight />
                  </div>
                )}

                {step === 7 && (
                  <div style={{ marginTop: 12 }}>
                    <div className="mono" style={{ fontSize: 11.5, marginBottom: 6 }}>
                      <StateRow label="publicado em" value="/events/live" show highlight />
                    </div>
                    <p style={{ fontSize: 11.5, color: "var(--text-faint)", margin: 0 }}>
                      o PSI real abaixo é calculado sobre TODO o conjunto de teste vs. treino — esta transação sozinha não muda esse número, ela só entra na fila de dados que um lote futuro vai comparar.
                    </p>
                  </div>
                )}

                {step === 8 && driftStatus && (
                  <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    style={{
                      marginTop: 12, padding: "12px 14px", borderRadius: 8, textAlign: "center",
                      background: driftStatus.status === "estável" ? "var(--ok-soft)" : "var(--danger-soft)",
                      color: driftStatus.status === "estável" ? "var(--ok)" : "var(--danger)",
                    }}
                  >
                    <div style={{ fontWeight: 800, fontSize: 14 }}>
                      {driftStatus.status === "estável" ? "gatilho calmo" : "gatilho ativo — PSI real cruzou o limiar"}
                    </div>
                    <div className="mono" style={{ fontSize: 11, marginTop: 4 }}>
                      PSI máx. real = {fmt(driftStatus.maxPsi, 3)} · limiar = {driftStatus.thresholds.significativo}
                    </div>
                  </motion.div>
                )}
            </motion.div>

            {/* estado acumulado — como o "frame de variáveis" do Python Tutor, crescendo a cada passo */}
            <div className="panel">
              <div className="panel-title">Estado da transação</div>
              <div className="mono" style={{ fontSize: 11.5, display: "flex", flexDirection: "column", gap: 6 }}>
                <StateRow label="Time" value={fmt(result.pipeline_steps["1_dado_bruto"].Time, 1)} show={step >= 0} />
                <StateRow label="Amount" value={`R$ ${fmt(result.pipeline_steps["1_dado_bruto"].Amount, 2)}`} show={step >= 0} />
                <StateRow label="hour_sin" value={fmt(result.pipeline_steps["2_feature_engineering"].hour_sin)} show={step >= 1} />
                <StateRow label="hour_cos" value={fmt(result.pipeline_steps["2_feature_engineering"].hour_cos)} show={step >= 1} />
                <StateRow label="amount_log" value={fmt(result.pipeline_steps["2_feature_engineering"].amount_log)} show={step >= 1} />
                <StateRow label="Amount_scaled" value={fmt(result.pipeline_steps["3_normalizacao"].Amount_scaled)} show={step >= 2} />
                <StateRow label="fraud_probability" value={fmt(result.fraud_probability)} show={step >= 3} highlight={step >= 3} />
                <StateRow label="top_factor" value={result.top_factors[0] ? `${result.top_factors[0].feature} (${fmt(result.top_factors[0].shap_value, 3)})` : "—"} show={step >= 4} />
                <StateRow label="decision" value={result.decision} show={step >= 5} highlight={step >= 5} />
                <StateRow label="modelo servindo" value={modelInfo ? `${modelInfo.n_arvores} árvores em memória` : "—"} show={step >= 6} />
                <StateRow label="evento ao vivo" value="publicado em /events/live" show={step >= 7} />
                <StateRow
                  label="gatilho de retreino"
                  value={driftStatus ? (driftStatus.status === "estável" ? "calmo" : "ativo") : "—"}
                  show={step >= 8}
                  highlight={step >= 8}
                />
              </div>
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "center", gap: 10, marginTop: 16 }}>
            <button className="btn" onClick={() => goto(step - 1)} disabled={step === 0}>◀ anterior</button>
            <span className="mono" style={{ fontSize: 12, color: "var(--text-faint)", alignSelf: "center" }}>
              passo {step + 1} / {STEP_META.length}
            </span>
            <button className="btn" onClick={() => goto(step + 1)} disabled={step === STEP_META.length - 1}>próximo ▶</button>
          </div>
        </>
      )}
    </div>
  );
}

function StateRow({ label, value, show, highlight }) {
  if (!show) return null;
  return (
    <motion.div
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      style={{
        display: "flex", justifyContent: "space-between", padding: "3px 6px", borderRadius: 4,
        background: highlight ? "var(--accent-soft)" : "transparent",
        color: highlight ? "var(--accent-text)" : "var(--text)",
      }}
    >
      <span style={{ color: highlight ? "var(--accent-text)" : "var(--text-faint)" }}>{label}</span>
      <span style={{ fontWeight: 600 }}>{value}</span>
    </motion.div>
  );
}

function ShapBar({ feature, value }) {
  const isFraudPush = value > 0;
  const width = Math.min(100, Math.abs(value) * 22);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, fontSize: 11 }}>
      <span className="mono" style={{ width: 42, color: "var(--text-faint)" }}>{feature}</span>
      <div style={{ flex: 1, height: 12, background: "var(--surface-alt)", borderRadius: 3, overflow: "hidden" }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${width}%` }}
          transition={{ duration: 0.4 }}
          style={{ height: "100%", background: isFraudPush ? "var(--danger)" : "var(--ok)" }}
        />
      </div>
      <span className="mono" style={{ width: 46, textAlign: "right", color: "var(--text-faint)" }}>{value.toFixed(3)}</span>
    </div>
  );
}
