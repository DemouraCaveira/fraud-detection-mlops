import { useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, { Background, ConnectionLineType } from "reactflow";
import "reactflow/dist/style.css";
import StageNode from "./StageNode";
import PaymentNode from "./PaymentNode";
import FigureModal from "../components/FigureModal";

const nodeTypes = { stage: StageNode, payment: PaymentNode };

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";

const PIPELINE_STAGES = ["ingestao", "preprocessamento", "treino", "validacao", "explicabilidade"];

// gráficos reais já produzidos pelo pipeline (reports/figures/*.png, servidos em
// /figures) — associados à etapa a que mais se relacionam pedagogicamente. O rótulo
// "gerado em" mantém honesto de qual script cada um realmente vem, mesmo quando isso
// não é o mesmo nó do grafo (ex.: a curva de aprendizado é calculada em
// src/evaluate.py, mas ilustra o comportamento do treino).
const FIGURES = {
  ingestao: [{ file: "01_class_balance.png", caption: "Desbalanceamento real das classes", source: "src/eda.py" }],
  preprocessamento: [
    { file: "02_amount_distribution.png", caption: "Distribuição de Amount (justifica RobustScaler + log)", source: "src/eda.py" },
    { file: "03_time_distribution.png", caption: "Distribuição de Time (justifica a codificação cíclica de hora)", source: "src/eda.py" },
  ],
  treino: [
    { file: "07_learning_curve.png", caption: "Curva de aprendizado — overfitting/underfitting", source: "gerado em src/evaluate.py" },
    { file: "08_scale_benchmark.png", caption: "Tempo de treino vs. volume de dados", source: "gerado em src/evaluate.py" },
  ],
  validacao: [
    { file: "05_confusion_matrix.png", caption: "Matriz de confusão (threshold de custo real)", source: "src/evaluate.py" },
    { file: "06_cross_validation.png", caption: "Cross-validation — variância entre folds", source: "src/evaluate.py" },
  ],
  explicabilidade: [
    { file: "11_shap_summary.png", caption: "SHAP summary — impacto de cada atributo", source: "src/explainability.py" },
    { file: "12_shap_feature_importance.png", caption: "Importância média por atributo (SHAP)", source: "src/explainability.py" },
    { file: "13_shap_waterfall_fraud_case.png", caption: "Waterfall — um caso real de fraude explicado", source: "src/explainability.py" },
  ],
  deploy: [{ file: "09_complexity_comparison.png", caption: "Latência de inferência por complexidade de modelo", source: "gerado em src/evaluate.py" }],
  monitor: [{ file: "10_drift_monitor.png", caption: "PSI/KS — treino vs. teste e choque simulado", source: "monitoring/drift_monitor.py" }],
};

function figuresFor(id) {
  return (FIGURES[id] || []).map((f) => ({ src: `${API_BASE}/figures/${f.file}`, caption: f.caption, source: f.source }));
}

const NODE_META = {
  ingestao: { label: "Ingestão", kicker: "src/ingestion.py" },
  preprocessamento: { label: "Pré-processamento", kicker: "src/preprocessing.py" },
  treino: { label: "Treino", kicker: "src/train.py" },
  validacao: { label: "Validação", kicker: "src/evaluate.py" },
  explicabilidade: { label: "Explicabilidade", kicker: "src/explainability.py" },
  registry: { label: "Artefato / Registry", kicker: "models/*.joblib" },
  promocao: { label: "Promoção hom→prod", kicker: "referência — GitHub Actions" },
  backend: { label: "Aplicação Backend", kicker: "recebe o JSON da transação" },
  maquininha: { label: "Maquininha (Terminal POS)", kicker: "lê o cartão — chip, aproximação (NFC) ou tarja" },
  rede: { label: "Rede da Adquirente", kicker: "conexão do terminal (4G/5G/Wi-Fi) até o backend" },
  backend: { label: "Aplicação Backend", kicker: "recebe o JSON, monta o payload do modelo" },
  deploy: { label: "Deploy / API", kicker: "deploy/api.py — carrega models/lightgbm_fraud.joblib (202 árvores) 1x, em memória" },
  banco_dados: { label: "Banco de Dados Transacional", kicker: "camada OLTP — grava transação + decisão" },
  monitor: { label: "Monitoramento", kicker: "monitoring/drift_monitor.py" },
  retreino: { label: "Retreino", kicker: "gatilho: PSI real vs. limiar" },
};

// data de entrada/saída de cada nó do modo "ao vivo" — pra deixar concreto "o que chega
// e o que sai" em cada caixa, não só o nome da tecnologia
const NODE_IO = {
  maquininha: { in: "cartão físico (chip/NFC/tarja)", out: '{"cartao":"**** 4321","valor":"R$ 84,96"}' },
  rede: { in: "sinal de dados do terminal", out: "pacote roteado até a aplicação backend" },
  backend: { in: "JSON bruto da maquininha", out: '{"Time":..,"Amount":..,"V1..V28":[...]}' },
  deploy: { in: "Transaction {Time, Amount, V1..V28}", out: '{"decision":"...","fraud_probability":...}' },
  banco_dados: { in: "transação + decisão da API", out: "registro persistido (auditoria/consulta futura)" },
};

// Espaçamento generoso de propósito: com nós grudados as linhas somem umas nas outras
// numa projeção em sala. Cada modo tem seu próprio layout — o "ao vivo" é uma linha reta
// de pipeline, na ordem física real de uma transação (cliente → maquininha → rede →
// backend → API → banco de dados), sem nós sobrepostos ou fora de ordem; o laço de
// monitoramento/retreino fica numa segunda fileira, claramente abaixo, não misturado.
const LAYOUT = {
  replay: {
    ingestao: { x: 0, y: 260 },
    preprocessamento: { x: 260, y: 260 },
    treino: { x: 520, y: 260 },
    validacao: { x: 780, y: 260 },
    explicabilidade: { x: 1040, y: 260 },
    registry: { x: 1300, y: 260 },
    promocao: { x: 1300, y: 20 },
    deploy: { x: 1560, y: 260 },
    monitor: { x: 1560, y: 580 },
    retreino: { x: 1300, y: 580 },
  },
  live: {
    app_pagamento: { x: 0, y: 320 },
    maquininha: { x: 300, y: 320 },
    rede: { x: 600, y: 320 },
    backend: { x: 900, y: 320 },
    deploy: { x: 1200, y: 320 },
    banco_dados: { x: 1500, y: 320 },
    monitor: { x: 1200, y: 660 },
    retreino: { x: 900, y: 660 },
    registry: { x: 600, y: 660 },
  },
};

const MODE_NODES = { replay: Object.keys(LAYOUT.replay), live: Object.keys(LAYOUT.live) };

const EDGE_DEFS = {
  replay: [
    { id: "e1", source: "ingestao", target: "preprocessamento" },
    { id: "e2", source: "preprocessamento", target: "treino" },
    { id: "e3", source: "treino", target: "validacao" },
    { id: "e4", source: "validacao", target: "explicabilidade" },
    { id: "e5", source: "explicabilidade", target: "registry" },
    { id: "e6", source: "registry", target: "promocao", dashed: true },
    { id: "e7", source: "promocao", target: "deploy", dashed: true },
    { id: "e8", source: "deploy", target: "monitor" },
    { id: "e9", source: "monitor", target: "retreino", dashed: true },
    { id: "e10", source: "retreino", target: "registry", dashed: true },
  ],
  live: [
    { id: "e_card", source: "app_pagamento", target: "maquininha", label: "① cartão lido" },
    { id: "e_net", source: "maquininha", target: "rede", label: "② sinal do terminal" },
    { id: "e_route", source: "rede", target: "backend", label: "③ roteado à aplicação" },
    { id: "e_backend", source: "backend", target: "deploy", label: "④ POST /predict" },
    { id: "e_persist", source: "deploy", target: "banco_dados", label: "⑤ registra" },
    { id: "e8", source: "deploy", target: "monitor" },
    { id: "e9", source: "monitor", target: "retreino", dashed: true },
    { id: "e10", source: "retreino", target: "registry", dashed: true },
  ],
};

function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

export default function LifecycleFlow({ replayEvents, liveEvents, driftReport, mode }) {
  const [stageState, setStageState] = useState({});
  const [deployEcho, setDeployEcho] = useState(null);
  const [deployPulsing, setDeployPulsing] = useState(false);
  const [openFigure, setOpenFigure] = useState(null);
  const deployPulseTimer = useRef(null);

  // reconstrói o status de cada etapa a partir do histórico acumulado de eventos do
  // replay (start -> running, end -> done, com o payload real como legenda)
  useEffect(() => {
    if (mode !== "replay") return;
    const next = {};
    for (const ev of replayEvents) {
      if (!ev.stage) continue;
      next[ev.stage] = { status: ev.type === "start" ? "running" : "done", payload: ev.type === "end" ? ev.payload : next[ev.stage]?.payload };
    }
    setStageState(next);
  }, [replayEvents, mode]);

  useEffect(() => {
    if (mode !== "replay" || replayEvents.length === 0) {
      setStageState({});
    }
  }, [mode]);

  useEffect(() => {
    if (liveEvents.length === 0) return;
    const last = liveEvents[liveEvents.length - 1];
    if (last.type !== "prediction") return;
    setDeployEcho(last);
    setDeployPulsing(true);
    clearTimeout(deployPulseTimer.current);
    deployPulseTimer.current = setTimeout(() => setDeployPulsing(false), 900);
  }, [liveEvents]);

  useEffect(() => () => clearTimeout(deployPulseTimer.current), []);

  const driftStatus = useMemo(() => {
    if (!driftReport) return null;
    const rows = driftReport.real_drift_treino_vs_teste || [];
    const maxPsi = rows.reduce((m, r) => Math.max(m, r.psi ?? 0), 0);
    const thresholds = driftReport.thresholds || { moderado: 0.1, significativo: 0.25 };
    const status = maxPsi >= thresholds.significativo ? "significativo" : maxPsi >= thresholds.moderado ? "moderado" : "estavel";
    return { maxPsi, status, thresholds };
  }, [driftReport]);

  const { nodes, edges } = useMemo(() => {
    const layout = LAYOUT[mode] || LAYOUT.replay;

    const nodeList = Object.entries(layout).map(([id, pos]) => {
      if (id === "app_pagamento") {
        return { id, type: "payment", position: pos, data: { pulsing: deployPulsing } };
      }

      let status = "idle";
      let metric = null;
      let central = false;
      const meta = NODE_META[id];

      if (PIPELINE_STAGES.includes(id)) {
        const st = stageState[id];
        status = st?.status || "idle";
        if (st?.payload) metric = summarize(id, st.payload);
      } else if (id === "registry") {
        const treino = stageState["treino"];
        status = treino?.status === "done" ? "done" : mode === "live" ? "idle" : "idle";
        metric = treino?.payload
          ? `${treino.payload.n_arvores} árvores · ${fmt(treino.payload.main_train_s, 1)}s`
          : mode === "live" ? "mesmo artefato que o Deploy/API já tem carregado" : null;
      } else if (id === "maquininha" || id === "rede") {
        status = deployPulsing ? "running" : deployEcho ? "done" : "idle";
        metric = deployEcho ? "transação repassada adiante" : "aguardando cliente";
      } else if (id === "backend") {
        status = deployPulsing ? "running" : deployEcho ? "done" : "idle";
        metric = deployEcho ? "JSON recebido → repassado à API de inferência" : "aguardando requisição do cliente";
      } else if (id === "deploy") {
        central = true;
        status = deployPulsing ? "running" : deployEcho ? "done" : "idle";
        metric = deployEcho
          ? `última: ${deployEcho.decision} (p=${fmt(deployEcho.fraud_probability, 3)}, ${fmt(deployEcho.latency_ms, 2)}ms)`
          : "aguardando POST /predict…";
      } else if (id === "banco_dados") {
        status = deployPulsing ? "running" : deployEcho ? "done" : "idle";
        metric = deployEcho ? `registro salvo: transação + decisão "${deployEcho.decision}"` : "aguardando registro";
      } else if (id === "monitor") {
        // a etapa "drift" da captura não tem nó próprio — funde no nó de Monitoramento
        const st = stageState["drift"];
        status = st?.status || "idle";
        metric = driftStatus ? `PSI máx.: ${fmt(driftStatus.maxPsi, 3)} (${driftStatus.status})` : st?.payload ? summarize("drift", st.payload) : null;
      } else if (id === "retreino") {
        status = driftStatus && driftStatus.status !== "estavel" ? "armed" : "idle";
        metric = driftStatus ? `PSI ${fmt(driftStatus.maxPsi, 3)} ${driftStatus.status !== "estavel" ? "≥" : "<"} limiar ${driftStatus.thresholds.significativo}` : null;
      } else if (id === "promocao") {
        status = "idle";
        metric = "artefato validado em hom, baixado (não retreinado) em prod";
      }

      return {
        id,
        type: "stage",
        position: pos,
        data: {
          label: meta?.label, kicker: meta?.kicker, status, metric, central,
          figures: figuresFor(id), onOpenFigure: setOpenFigure,
          liveScores: id === "monitor" ? liveEvents : undefined,
          io: NODE_IO[id],
        },
      };
    });

    const isRunning = (id) => stageState[id]?.status === "running";
    const driftActive = driftStatus?.status && driftStatus.status !== "estavel";

    const EDGE_ANIM = {
      e1: isRunning("ingestao"), e2: isRunning("preprocessamento"), e3: isRunning("treino"),
      e4: isRunning("validacao"), e5: isRunning("explicabilidade"),
      e8: !!deployEcho, e9: driftActive, e10: driftActive,
      e_card: deployPulsing, e_net: deployPulsing, e_route: deployPulsing,
      e_backend: deployPulsing, e_persist: deployPulsing,
    };

    const edgeList = (EDGE_DEFS[mode] || EDGE_DEFS.replay).map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      type: "smoothstep",
      animated: !!EDGE_ANIM[e.id],
      label: e.label,
      labelStyle: { fill: "var(--text-faint)", fontFamily: '"IBM Plex Mono", monospace', fontSize: 10 },
      labelBgStyle: { fill: "var(--bg)", fillOpacity: 0.9 },
      style: {
        stroke: e.dashed ? "var(--accent)" : "var(--canvas-line)",
        strokeWidth: e.dashed ? 1.6 : 1.4,
        strokeDasharray: e.dashed ? "5 4" : undefined,
      },
    }));

    return { nodes: nodeList, edges: edgeList };
  }, [stageState, deployEcho, deployPulsing, driftStatus, liveEvents, mode]);

  return (
    <>
      <ReactFlow
        key={mode}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        connectionLineType={ConnectionLineType.SmoothStep}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
      >
        <Background color="var(--canvas-grid)" gap={28} />
      </ReactFlow>
      <FigureModal figure={openFigure} onClose={() => setOpenFigure(null)} />
    </>
  );
}

function summarize(stage, payload) {
  if (!payload) return null;
  switch (stage) {
    case "ingestao":
      return `${payload.n_linhas?.toLocaleString("pt-BR")} linhas · ${fmt(payload.taxa_fraude_pct, 3)}% fraude`;
    case "preprocessamento":
      return `treino ${payload.n_treino?.toLocaleString("pt-BR")} · val ${payload.n_validacao?.toLocaleString("pt-BR")} · teste ${payload.n_teste?.toLocaleString("pt-BR")}`;
    case "treino":
      return `${payload.n_arvores} árvores · ${fmt(payload.main_train_s, 1)}s`;
    case "validacao":
      return `PR-AUC teste ${fmt(payload.pr_auc_teste)} · CV ${fmt(payload.cv_pr_auc_mean)}`;
    case "explicabilidade":
      return payload.top_features?.[0] ? `top atributo: ${payload.top_features[0].feature}` : null;
    case "drift":
      return `PSI máx. ${fmt(payload.psi_maximo_real, 3)} (${payload.status_geral})`;
    default:
      return null;
  }
}
