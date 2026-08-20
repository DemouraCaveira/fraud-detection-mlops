import { useEffect, useMemo, useRef, useState } from "react";
import LifecycleFlow from "./flow/LifecycleFlow";
import DriftChart from "./charts/DriftChart";
import MetricsPanel from "./charts/MetricsPanel";
import LiveMonitorChart from "./charts/LiveMonitorChart";
import MonitoringDashboard from "./charts/MonitoringDashboard";
import ProbabilityHistogram from "./charts/ProbabilityHistogram";
import VersioningPanel from "./charts/VersioningPanel";
import CardMachine from "./pos/CardMachine";
import TransactionDebugger from "./debugger/TransactionDebugger";
import TreeEnsembleView from "./internals/TreeEnsembleView";
import { useReplay } from "./hooks/useReplay";
import { useLive } from "./hooks/useLive";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";

const STAGE_LABEL = {
  ingestao: "Ingestão",
  preprocessamento: "Pré-processamento",
  treino: "Treino",
  validacao: "Validação",
  explicabilidade: "Explicabilidade",
  drift: "Monitoramento de drift",
};

function fmtTime(date) {
  return date.toLocaleTimeString("pt-BR", { hour12: false });
}

export default function App() {
  const [mode, setMode] = useState("replay"); // "replay" | "live" | "debug" | "trees"
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(6);
  const [driftReport, setDriftReport] = useState(null);
  const [evalSummary, setEvalSummary] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [apiOnline, setApiOnline] = useState(null);

  const replay = useReplay(mode === "replay" && playing, speed);
  // o stream de /events/live fica sempre aberto — o monitoramento (nó "Monitoramento" e
  // o gráfico ao vivo na barra lateral) reage a QUALQUER predição real, não importa qual
  // aba está aberta: maquininha, curl, Swagger, a página de demo em /, etc.
  const live = useLive(true);
  const logRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
    fetch(`${API_BASE}/drift-report`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setDriftReport)
      .catch(() => setDriftReport(null));
    fetch(`${API_BASE}/evaluation-summary`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setEvalSummary)
      .catch(() => setEvalSummary(null));
    fetch(`${API_BASE}/model-info`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setModelInfo)
      .catch(() => setModelInfo(null));
  }, []);

  useEffect(() => {
    if (replay.finished) setPlaying(false);
  }, [replay.finished]);

  const logLines = useMemo(() => {
    const lines = [];
    if (mode === "replay") {
      for (const ev of replay.events) {
        if (ev.type === "start") lines.push({ cls: "", text: `${STAGE_LABEL[ev.stage] || ev.stage} — ${ev.label}` });
        if (ev.type === "end") lines.push({ cls: "drift", text: `${STAGE_LABEL[ev.stage] || ev.stage} concluída` });
      }
    } else {
      for (const ev of live.events) {
        lines.push({
          cls: ev.decision === "BLOQUEAR" ? "block" : "approve",
          text: `predição real — ${ev.decision}  (p=${ev.fraud_probability.toFixed(3)}, ${ev.latency_ms.toFixed(2)}ms)`,
        });
      }
    }
    return lines.slice(-80);
  }, [mode, replay.events, live.events]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);

  const handleReplayToggle = () => {
    if (playing) {
      setPlaying(false);
    } else {
      replay.clear();
      setPlaying(true);
    }
  };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <header
        style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "14px 20px", borderBottom: "1px solid var(--border)", gap: 16, flexWrap: "wrap",
        }}
      >
        <div>
          <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            fraud-detection-mlops — ciclo de vida
          </div>
          <div style={{ fontSize: 15, fontWeight: 600, marginTop: 2 }}>Visualizador do ciclo de vida do modelo</div>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <span className={`chip ${apiOnline ? "active" : ""}`}>{apiOnline == null ? "checando API…" : apiOnline ? "API online" : "API offline"}</span>

          <div style={{ display: "flex", border: "1px solid var(--border-strong)", borderRadius: 8, overflow: "hidden" }}>
            <button
              className="btn"
              style={{ border: "none", borderRadius: 0, background: mode === "replay" ? "var(--accent-soft)" : "var(--surface)" }}
              onClick={() => setMode("replay")}
            >
              replay do treino
            </button>
            <button
              className="btn"
              style={{ border: "none", borderRadius: 0, background: mode === "live" ? "var(--accent-soft)" : "var(--surface)" }}
              onClick={() => setMode("live")}
            >
              ao vivo (inferência)
            </button>
            <button
              className="btn"
              style={{ border: "none", borderRadius: 0, background: mode === "debug" ? "var(--accent-soft)" : "var(--surface)" }}
              onClick={() => setMode("debug")}
            >
              passo a passo
            </button>
            <button
              className="btn"
              style={{ border: "none", borderRadius: 0, background: mode === "trees" ? "var(--accent-soft)" : "var(--surface)" }}
              onClick={() => setMode("trees")}
            >
              202 árvores
            </button>
          </div>

          {mode === "replay" && (
            <>
              <button className="btn primary" onClick={handleReplayToggle}>
                {playing ? "pausar" : "▶ replay"}
              </button>
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: '"IBM Plex Mono"', fontSize: 11, color: "var(--text-muted)" }}>
                <span>velocidade</span>
                <input type="range" min="1" max="20" step="1" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
                <span>{speed}×</span>
              </div>
            </>
          )}
          <span className={`chip ${live.connected ? "active" : ""}`}>{live.connected ? "monitor ao vivo conectado" : "conectando monitor…"}</span>
        </div>
      </header>

      <div style={{ flex: 1, display: "grid", gridTemplateColumns: mode === "live" ? "340px 1fr 300px" : "1fr 300px", minHeight: 0 }}>
        {mode === "live" && (
          <div style={{ borderRight: "1px solid var(--border)", overflowY: "auto" }}>
            <div style={{ padding: "10px 16px 0", fontSize: 11, color: "var(--text-faint)" }}>
              gere um cliente aqui — o grafo ao lado reage em tempo real, é a mesma chamada
            </div>
            <CardMachine />
          </div>
        )}

        <div style={{ position: "relative", overflowY: "auto" }}>
          {mode === "debug" && <TransactionDebugger />}
          {mode === "trees" && <TreeEnsembleView />}
          {(mode === "replay" || mode === "live") && (
            <LifecycleFlow replayEvents={replay.events} liveEvents={live.events} driftReport={driftReport} mode={mode} />
          )}
        </div>

        <aside style={{ borderLeft: "1px solid var(--border)", padding: 16, display: "flex", flexDirection: "column", gap: 14, overflowY: "auto" }}>
          <div className="panel">
            <div className="panel-title">Monitoramento ao vivo — estilo Datadog</div>
            <MonitoringDashboard events={live.events} />
            <div style={{ marginTop: 10 }}>
              <LiveMonitorChart events={live.events} threshold={evalSummary?.test_metrics_main_model?.threshold} />
            </div>
            <div style={{ marginTop: 10 }}>
              <div style={{ fontSize: 10, color: "var(--text-faint)", marginBottom: 4 }}>distribuição das probabilidades (janela)</div>
              <ProbabilityHistogram events={live.events} threshold={evalSummary?.test_metrics_main_model?.threshold} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">Versionamento & gatilho de retreino</div>
            <VersioningPanel modelInfo={modelInfo} driftReport={driftReport} />
          </div>

          <div className="panel">
            <div className="panel-title">Métricas reais (validação)</div>
            <MetricsPanel summary={evalSummary} />
          </div>

          <div className="panel">
            <div className="panel-title">Drift real por atributo — PSI (treino vs. teste)</div>
            <DriftChart rows={driftReport?.real_drift_treino_vs_teste} thresholds={driftReport?.thresholds} />
          </div>

          <div className="panel" style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 180 }}>
            <div className="panel-title">Log de eventos</div>
            <div className="log" ref={logRef} style={{ flex: 1 }}>
              {logLines.length === 0 && <div className="row t">aguardando eventos…</div>}
              {logLines.map((l, i) => (
                <div className="row" key={i}>
                  <span className="t">{fmtTime(new Date())}</span> <span className={l.cls}>{l.text}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
