"""Deploy simulado: API de inferencia em tempo real (FastAPI).

Representa o endpoint que, em producao, receberia uma transacao no momento da
autorizacao e devolveria a decisao antes da confirmacao do pagamento — o mesmo contrato
que o endpoint SageMaker real-time descrito em architecture/aws_architecture.md
implementaria na nuvem. Rodar localmente com:

    uvicorn deploy.api:app --reload

e testar em http://127.0.0.1:8000/docs (Swagger gerado automaticamente pelo FastAPI).
"""

from __future__ import annotations

# ver nota de ordem de import em src/evaluate.py — lightgbm sempre antes de pandas/pyarrow
import lightgbm  # noqa: F401

import json
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])

    _state["config"] = config
    _state["model"] = joblib.load(models_dir / "lightgbm_fraud.joblib")
    _state["scaler"] = joblib.load(models_dir / "amount_scaler.joblib")
    with open(models_dir / "decision_threshold.json", "r", encoding="utf-8") as f:
        _state["threshold"] = json.load(f)["threshold"]
    _state["feature_cols"] = None  # calculado no primeiro request (depende das colunas geradas)

    logger.info(f"Modelo carregado. Threshold de decisao: {_state['threshold']:.4f}")
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


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    config = _state["config"]
    row_df = pd.DataFrame([transaction.model_dump()])
    row_df = engineer_features(row_df, config)
    row_df = apply_scaler(row_df, _state["scaler"], config)
    feature_cols = get_feature_columns(row_df, config)

    start = time.perf_counter()
    proba = float(_state["model"].predict_proba(row_df[feature_cols])[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000

    threshold = _state["threshold"]
    decision = "BLOQUEAR" if proba >= threshold else "APROVAR"

    return PredictionResponse(
        fraud_probability=proba, decision=decision, threshold_used=threshold, latency_ms=latency_ms
    )
