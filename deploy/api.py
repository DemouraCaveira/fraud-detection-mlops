"""Deploy simulado: API de inferencia em tempo real (FastAPI) + página didática.

Representa o endpoint que, em producao, receberia uma transacao no momento da
autorizacao e devolveria a decisao antes da confirmacao do pagamento — o mesmo contrato
que o endpoint SageMaker real-time descrito em architecture/aws_architecture.md
implementaria na nuvem. A pagina em `/` (não `/docs`) é pensada para demonstração em
sala: mostra o pipeline passo a passo (dado bruto -> features -> modelo -> decisão), a
explicação SHAP ao vivo da predição especifica, e uma "ficha do modelo" com a linhagem
(dataset -> treino -> artefato -> esta API). Rodar localmente com:

    python main.py --serve

e abrir http://127.0.0.1:8000/ (a demo) ou http://127.0.0.1:8000/docs (Swagger).
"""

from __future__ import annotations

# ver nota de ordem de import em src/evaluate.py — lightgbm sempre antes de pandas/pyarrow
import lightgbm  # noqa: F401

import json
import os
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from deploy.demo_faker import _fit_class_jitter_stats, generate_synthetic_transaction
from src.preprocessing import apply_scaler, engineer_features, get_feature_columns
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)

_state: dict = {}


class Transaction(BaseModel):
    Time: float = Field(..., description="Segundos desde a primeira transacao do lote")
    Amount: float = Field(..., ge=0)
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    decision: str
    threshold_used: float
    latency_ms: float
    pipeline_steps: dict
    top_factors: list


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    processed_dir = resolve_path(config["data"]["processed_dir"])
    reports_dir = resolve_path(config["paths"]["reports_dir"])

    _state["config"] = config
    _state["models_dir"] = models_dir
    model_path = models_dir / "lightgbm_fraud.joblib"
    _state["model"] = joblib.load(model_path)
    _state["scaler"] = joblib.load(models_dir / "amount_scaler.joblib")
    with open(models_dir / "decision_threshold.json", "r", encoding="utf-8") as f:
        _state["threshold"] = json.load(f)["threshold"]

    # TreeExplainer e rapido o bastante (~0.1ms/amostra, ver reports/relatorio.md secao
    # 7) para calcular a explicacao SHAP de cada predicao em tempo real, nao so em lote.
    _state["explainer"] = shap.TreeExplainer(_state["model"])

    # usado so pelo endpoint /generate-sample (demo), nao pelo /predict real
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    _state["class_stats"] = _fit_class_jitter_stats(train_df, config)

    # metadados para a "ficha do modelo" / linhagem — le o que ja foi gerado pelo
    # pipeline de treino (src/train.py) e validacao (src/evaluate.py), nao recalcula nada
    model_info = {
        "algoritmo": "LightGBM (gradient boosting)",
        "n_arvores": int(getattr(_state["model"], "n_estimators_", getattr(_state["model"], "best_iteration_", 0))),
        "treinado_em": time.strftime("%d/%m/%Y %H:%M", time.localtime(model_path.stat().st_mtime)),
        "dataset": "OpenML data_id=1597 (Credit Card Fraud Detection, ULB)",
        "threshold_decisao": _state["threshold"],
    }
    best_params_path = models_dir / "lightgbm_best_params.json"
    if best_params_path.exists():
        with open(best_params_path, "r", encoding="utf-8") as f:
            model_info["hiperparametros"] = json.load(f)
    eval_summary_path = reports_dir / "evaluation_summary.json"
    if eval_summary_path.exists():
        with open(eval_summary_path, "r", encoding="utf-8") as f:
            eval_summary = json.load(f)
        m = eval_summary.get("test_metrics_main_model", {})
        model_info["metricas_teste"] = {
            "pr_auc": m.get("average_precision"),
            "roc_auc": m.get("roc_auc"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
        }
    _state["model_info"] = model_info

    logger.info(f"Modelo carregado ({model_info['n_arvores']} arvores). Threshold: {_state['threshold']:.4f}")
    yield
    _state.clear()


app = FastAPI(
    title="API de Deteccao de Fraude — Deploy Simulado",
    description="Endpoint de inferencia em tempo real para o modelo LightGBM de deteccao de fraude.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/model-info")
def model_info():
    """Ficha do modelo — a linhagem: dataset de origem, quando foi treinado, com quais
    hiperparametros, e o desempenho medido na validacao. O mesmo tipo de informacao que
    um Model Registry (SageMaker, MLflow) guardaria por versao de modelo."""
    return _state["model_info"]


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    config = _state["config"]
    raw = transaction.model_dump()
    row_df = pd.DataFrame([raw])
    row_df = engineer_features(row_df, config)
    row_df = apply_scaler(row_df, _state["scaler"], config)
    feature_cols = get_feature_columns(row_df, config)
    X = row_df[feature_cols]

    start = time.perf_counter()
    proba = float(_state["model"].predict_proba(X)[0, 1])
    shap_values = _state["explainer"](X)
    latency_ms = (time.perf_counter() - start) * 1000

    threshold = _state["threshold"]
    decision = "BLOQUEAR" if proba >= threshold else "APROVAR"

    # top atributos que mais empurraram ESTA predicao especifica (nao a media do
    # modelo) — a mesma tecnica de src/explainability.py, aplicada em tempo real
    contribs = list(zip(feature_cols, shap_values.values[0]))
    contribs.sort(key=lambda t: -abs(t[1]))
    top_factors = [{"feature": f, "shap_value": float(v)} for f, v in contribs[:5]]

    pipeline_steps = {
        "1_dado_bruto": {"Time": raw["Time"], "Amount": raw["Amount"]},
        "2_feature_engineering": {
            "hour_sin": float(row_df["hour_sin"].iloc[0]),
            "hour_cos": float(row_df["hour_cos"].iloc[0]),
            "amount_log": float(row_df["amount_log"].iloc[0]),
        },
        "3_normalizacao": {"Amount_scaled": float(row_df["Amount_scaled"].iloc[0])},
    }

    return PredictionResponse(
        fraud_probability=proba, decision=decision, threshold_used=threshold, latency_ms=latency_ms,
        pipeline_steps=pipeline_steps, top_factors=top_factors,
    )


@app.get("/generate-sample")
def generate_sample():
    """Gera uma transacao sintetica (Faker + jitter estatistico) para preencher a
    pagina de demonstracao — ver deploy/demo_faker.py para a explicacao completa da
    tecnica. Nao roda inferencia; so gera o payload para o usuario revisar/editar antes
    de enviar pra /predict, exatamente como um QA preencheria um formulario de teste."""
    txn = generate_synthetic_transaction(_state["class_stats"], _state["config"])
    metadata = {k: v for k, v in txn["metadata"].items() if k != "_classe_geradora"}
    transaction = {k: float(v) for k, v in txn["features"].items()}
    return {"metadata": metadata, "transaction": transaction}


_DEMO_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Detecção de Fraude — Demo ao Vivo</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #12161c; --muted: #5b6472;
    --border: #dbe0e6; --accent: #b45309; --accent-soft: #fff7ec; --accent2: #2563eb; --accent2-soft: #eef4ff;
    --good: #15803d; --good-soft: #ecfdf3; --bad: #b91c1c; --bad-soft: #fef2f2;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    padding: 32px 20px 80px;
  }
  .wrap { max-width: 820px; margin: 0 auto; }
  h1 { font-size: 24px; font-weight: 800; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 24px; font-size: 14px; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 22px; margin-bottom: 18px;
  }
  .card h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); margin: 0 0 16px; font-weight: 700; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
  label { display: block; font-size: 12.5px; color: var(--muted); margin-bottom: 4px; font-weight: 600; }
  input {
    width: 100%; padding: 9px 11px; border: 1px solid var(--border); border-radius: 7px;
    font-size: 14px; font-family: inherit; background: var(--bg); color: var(--ink);
  }
  input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  .meta-line { font-size: 13.5px; color: var(--muted); margin-bottom: 2px; }
  .meta-line b { color: var(--ink); }
  details { margin-top: 6px; }
  summary { cursor: pointer; font-size: 12.5px; color: var(--accent); font-weight: 600; }
  .v-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 12px; }
  .v-grid input { font-size: 12px; padding: 6px 7px; font-family: ui-monospace, Consolas, monospace; }
  .v-grid label { font-size: 10.5px; margin-bottom: 2px; }
  .btn {
    border: none; border-radius: 8px; padding: 12px 18px; font-size: 14.5px; font-weight: 700;
    cursor: pointer; width: 100%;
  }
  .btn-secondary { background: var(--bg); color: var(--ink); border: 1px solid var(--border); margin-bottom: 10px; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn:disabled { opacity: .5; cursor: default; }
  #result, #explainCard, #pipelineCard { display: none; }
  #result.show, #explainCard.show, #pipelineCard.show { display: block; }
  .badge {
    display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 800;
    font-size: 15px; letter-spacing: .02em; margin-bottom: 14px;
  }
  .badge.aprovar { background: var(--good-soft); color: var(--good); }
  .badge.bloquear { background: var(--bad-soft); color: var(--bad); }
  .bar-track { background: var(--bg); border: 1px solid var(--border); border-radius: 999px;
    height: 22px; overflow: hidden; margin-bottom: 6px; }
  .bar-fill { height: 100%; border-radius: 999px; transition: width .4s ease; }
  .stat-row { display: flex; justify-content: space-between; font-size: 13px; color: var(--muted); margin-top: 10px; }
  .stat-row b { color: var(--ink); font-family: ui-monospace, Consolas, monospace; }

  /* ficha do modelo */
  .model-card { display: flex; flex-wrap: wrap; gap: 22px; font-size: 13px; }
  .model-stat .l { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .03em; }
  .model-stat .v { font-weight: 700; font-family: ui-monospace, Consolas, monospace; font-size: 14px; }

  /* pipeline stepper */
  .pipeline { display: flex; align-items: stretch; gap: 6px; overflow-x: auto; padding-bottom: 4px; }
  .pstep {
    flex: 1; min-width: 150px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 9px; padding: 12px; font-size: 11.5px;
  }
  .pstep .ptitle { font-weight: 700; font-size: 11px; text-transform: uppercase;
    letter-spacing: .03em; color: var(--accent2); margin-bottom: 8px; }
  .pstep .pline { font-family: ui-monospace, Consolas, monospace; font-size: 11px;
    color: var(--ink); margin-bottom: 3px; word-break: break-all; }
  .pstep .pline .k { color: var(--muted); }
  .parrow { display: flex; align-items: center; color: var(--muted); font-size: 18px; padding: 0 2px; }

  /* shap factors */
  .factor-row { display: grid; grid-template-columns: 70px 1fr 60px; gap: 10px; align-items: center;
    font-size: 12.5px; margin-bottom: 8px; }
  .factor-row .fname { font-family: ui-monospace, Consolas, monospace; font-weight: 700; }
  .factor-track { background: var(--bg); border-radius: 4px; height: 16px; position: relative; overflow: hidden; }
  .factor-fill { height: 100%; position: absolute; top: 0; }
  .factor-fill.pos { background: var(--bad); right: 50%; }
  .factor-fill.neg { background: var(--good); left: 50%; }
  .factor-mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--border); }
  .fval { font-family: ui-monospace, Consolas, monospace; text-align: right; }
  .legend-shap { font-size: 11.5px; color: var(--muted); margin-top: 10px; }
  .legend-shap span.pos { color: var(--bad); font-weight: 700; }
  .legend-shap span.neg { color: var(--good); font-weight: 700; }

  /* linhagem */
  .lineage { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; font-size: 12px; }
  .lnode { background: var(--accent2-soft); color: var(--accent2); padding: 6px 12px; border-radius: 7px; font-weight: 700; }
  .larrow { color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <h1>🛡️ Detecção de Fraude — Demo ao Vivo</h1>
  <p class="sub">Gera uma transação sintética, envia pra API real (<code>/predict</code>) e mostra o pipeline, a decisão e o porquê — tudo calculado na hora, não simulado.</p>

  <div class="card">
    <h2>🧠 Ficha do modelo (linhagem)</h2>
    <div class="model-card" id="modelCard">Carregando…</div>
    <div class="lineage" id="lineageRow" style="margin-top:16px;"></div>
  </div>

  <div class="card">
    <h2>1. Transação</h2>
    <button class="btn btn-secondary" onclick="gerarAmostra()">🎲 Gerar transação sintética</button>
    <div id="metaBox"></div>
    <div class="row">
      <div><label>Valor (R$)</label><input id="Amount" type="number" step="0.01"></div>
      <div><label>Time (segundos desde t0)</label><input id="Time" type="number" step="1"></div>
    </div>
    <details>
      <summary>Ver atributos técnicos (V1–V28, anonimizados via PCA)</summary>
      <div class="v-grid" id="vGrid"></div>
    </details>
  </div>

  <button class="btn btn-primary" onclick="enviarPredicao()" id="sendBtn">🔍 Enviar para análise</button>

  <div class="card" id="pipelineCard">
    <h2>2. Pipeline — o que aconteceu com esta transação</h2>
    <div class="pipeline" id="pipelineFlow"></div>
  </div>

  <div class="card" id="result">
    <h2>3. Decisão do modelo</h2>
    <span id="badge" class="badge"></span>
    <div class="bar-track"><div class="bar-fill" id="barFill"></div></div>
    <div class="stat-row"><span>Probabilidade de fraude</span><b id="probaTxt"></b></div>
    <div class="stat-row"><span>Threshold de decisão</span><b id="threshTxt"></b></div>
    <div class="stat-row"><span>Latência da inferência</span><b id="latTxt"></b></div>
  </div>

  <div class="card" id="explainCard">
    <h2>4. Por que o modelo decidiu isso (SHAP, calculado nesta predição)</h2>
    <div id="factorsBox"></div>
    <div class="legend-shap">
      <span class="pos">■ vermelho</span> empurra para FRAUDE &nbsp;·&nbsp;
      <span class="neg">■ verde</span> empurra para LEGÍTIMA
    </div>
  </div>
</div>

<script>
const vFields = Array.from({length: 28}, (_, i) => "V" + (i + 1));

function buildVGrid() {
  const grid = document.getElementById("vGrid");
  grid.innerHTML = vFields.map(v =>
    `<div><label>${v}</label><input id="${v}" type="number" step="0.0001" value="0"></div>`
  ).join("");
}
buildVGrid();

async function carregarFichaModelo() {
  const res = await fetch("/model-info");
  const m = await res.json();
  const met = m.metricas_teste || {};
  const hp = m.hiperparametros || {};
  document.getElementById("modelCard").innerHTML = `
    <div class="model-stat"><div class="l">Algoritmo</div><div class="v">${m.algoritmo}</div></div>
    <div class="model-stat"><div class="l">Árvores</div><div class="v">${m.n_arvores}</div></div>
    <div class="model-stat"><div class="l">Treinado em</div><div class="v">${m.treinado_em}</div></div>
    <div class="model-stat"><div class="l">PR-AUC (teste)</div><div class="v">${(met.pr_auc*100).toFixed(2)}%</div></div>
    <div class="model-stat"><div class="l">ROC-AUC (teste)</div><div class="v">${(met.roc_auc*100).toFixed(2)}%</div></div>
    <div class="model-stat"><div class="l">num_leaves / max_depth</div><div class="v">${hp.num_leaves} / ${hp.max_depth}</div></div>
  `;
  document.getElementById("lineageRow").innerHTML = `
    <span class="lnode">📦 ${m.dataset}</span><span class="larrow">→</span>
    <span class="lnode">⚙️ Ingestão + Pré-processamento</span><span class="larrow">→</span>
    <span class="lnode">🎯 Treino (Optuna, hom)</span><span class="larrow">→</span>
    <span class="lnode">📋 Artefato promovido</span><span class="larrow">→</span>
    <span class="lnode">🚀 Esta API (prod)</span>
  `;
}
carregarFichaModelo();

async function gerarAmostra() {
  const res = await fetch("/generate-sample");
  const data = await res.json();
  document.getElementById("Amount").value = data.transaction.Amount.toFixed(2);
  document.getElementById("Time").value = Math.round(data.transaction.Time);
  vFields.forEach(v => { document.getElementById(v).value = data.transaction[v].toFixed(4); });
  const m = data.metadata;
  document.getElementById("metaBox").innerHTML =
    `<div class="meta-line"><b>${m.titular}</b> — cartão final ${m.cartao_final}</div>
     <div class="meta-line">${m.estabelecimento}, ${m.cidade}</div>`;
  document.getElementById("result").classList.remove("show");
  document.getElementById("pipelineCard").classList.remove("show");
  document.getElementById("explainCard").classList.remove("show");
}

function renderPipeline(steps) {
  const s1 = steps["1_dado_bruto"], s2 = steps["2_feature_engineering"], s3 = steps["3_normalizacao"];
  const stepHtml = (title, obj) => `
    <div class="pstep"><div class="ptitle">${title}</div>
      ${Object.entries(obj).map(([k,v]) => `<div class="pline"><span class="k">${k}:</span> ${typeof v === "number" ? v.toFixed(4) : v}</div>`).join("")}
    </div>`;
  document.getElementById("pipelineFlow").innerHTML =
    stepHtml("① Dado bruto recebido", s1) + '<div class="parrow">→</div>' +
    stepHtml("② Feature engineering", s2) + '<div class="parrow">→</div>' +
    stepHtml("③ Normalização (RobustScaler)", s3) + '<div class="parrow">→</div>' +
    `<div class="pstep"><div class="ptitle">④ Modelo</div><div class="pline">LightGBM.predict_proba()</div></div>`;
  document.getElementById("pipelineCard").classList.add("show");
}

function renderFactors(factors) {
  const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.001);
  document.getElementById("factorsBox").innerHTML = factors.map(f => {
    const pct = (Math.abs(f.shap_value) / maxAbs) * 50;
    const cls = f.shap_value >= 0 ? "pos" : "neg";
    return `<div class="factor-row">
      <span class="fname">${f.feature}</span>
      <div class="factor-track"><div class="factor-mid"></div><div class="factor-fill ${cls}" style="width:${pct}%"></div></div>
      <span class="fval">${f.shap_value >= 0 ? "+" : ""}${f.shap_value.toFixed(3)}</span>
    </div>`;
  }).join("");
  document.getElementById("explainCard").classList.add("show");
}

async function enviarPredicao() {
  const btn = document.getElementById("sendBtn");
  btn.disabled = true; btn.textContent = "Analisando...";

  const payload = { Amount: parseFloat(document.getElementById("Amount").value || 0),
                     Time: parseFloat(document.getElementById("Time").value || 0) };
  vFields.forEach(v => { payload[v] = parseFloat(document.getElementById(v).value || 0); });

  const res = await fetch("/predict", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
  });
  const data = await res.json();

  renderPipeline(data.pipeline_steps);

  const badge = document.getElementById("badge");
  const isBloquear = data.decision === "BLOQUEAR";
  badge.textContent = isBloquear ? "🚫 BLOQUEAR" : "✅ APROVAR";
  badge.className = "badge " + (isBloquear ? "bloquear" : "aprovar");

  const pct = (data.fraud_probability * 100);
  const fill = document.getElementById("barFill");
  fill.style.width = pct.toFixed(1) + "%";
  fill.style.background = isBloquear ? "var(--bad)" : "var(--good)";

  document.getElementById("probaTxt").textContent = pct.toFixed(2) + "%";
  document.getElementById("threshTxt").textContent = (data.threshold_used * 100).toFixed(2) + "%";
  document.getElementById("latTxt").textContent = data.latency_ms.toFixed(3) + " ms";
  document.getElementById("result").classList.add("show");

  renderFactors(data.top_factors);

  btn.disabled = false; btn.textContent = "🔍 Enviar para análise";
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return _DEMO_PAGE
