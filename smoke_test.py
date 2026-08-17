"""Teste de fumaca: carrega o modelo treinado e roda uma predicao real sobre uma
transacao real do conjunto de teste — a demonstracao funcional minima do projeto.
Deve rodar sem erro apos `python run_pipeline.py`.

Uso:
    python smoke_test.py
"""

from __future__ import annotations

# ver nota de ordem de import em src/evaluate.py — lightgbm sempre antes de pandas/pyarrow
import lightgbm  # noqa: F401

import json
import sys
import time

import joblib
import pandas as pd

from src.preprocessing import get_feature_columns
from src.utils import load_config, resolve_path


def main() -> int:
    config = load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    processed_dir = resolve_path(config["data"]["processed_dir"])

    model_path = models_dir / "lightgbm_fraud.joblib"
    test_path = processed_dir / "test.parquet"
    if not model_path.exists() or not test_path.exists():
        print(f"ERRO: artefatos nao encontrados ({model_path} / {test_path}). Rode `python run_pipeline.py` primeiro.")
        return 1

    model = joblib.load(model_path)
    test_df = pd.read_parquet(test_path)
    feature_cols = get_feature_columns(test_df, config)

    sample_row = test_df.sample(n=1, random_state=7)
    target_col = config["features"]["target_col"]

    start = time.perf_counter()
    proba = float(model.predict_proba(sample_row[feature_cols])[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000

    threshold_path = models_dir / "decision_threshold.json"
    threshold = 0.5
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold = json.load(f)["threshold"]

    decision = "BLOQUEAR" if proba >= threshold else "APROVAR"
    real_label = "FRAUDE" if int(sample_row[target_col].iloc[0]) == 1 else "LEGITIMA"

    print("=" * 60)
    print("SMOKE TEST — predicao real sobre transacao do conjunto de teste")
    print("=" * 60)
    print(f"Rotulo real da transacao amostrada : {real_label}")
    print(f"Probabilidade de fraude prevista    : {proba:.6f}")
    print(f"Threshold de decisao                : {threshold:.6f}")
    print(f"Decisao do modelo                   : {decision}")
    print(f"Latencia da inferencia              : {latency_ms:.3f} ms")
    print("=" * 60)
    print("OK — modelo carregado e predicao executada com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
