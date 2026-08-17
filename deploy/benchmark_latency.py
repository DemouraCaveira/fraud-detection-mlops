"""Benchmark de latencia de inferencia — a prova numerica de que 'toda a arquitetura
end-to-end responde em milissegundos'.

Mede dois cenarios, nos dois modelos treinados (baseline e principal):
  1. Latencia por transacao unica (o caso real de producao: uma transacao chega, o
     modelo decide aprovar/bloquear antes da confirmacao do pagamento).
  2. Throughput em lote (quantas transacoes por segundo o modelo processa).

Os numeros aqui alimentam diretamente a secao de performance do relatorio e a escolha
de instancia/endpoint na arquitetura AWS (architecture/aws_architecture.md).
"""

from __future__ import annotations

# ver nota de ordem de import em src/evaluate.py — lightgbm sempre antes de pandas/pyarrow
import lightgbm  # noqa: F401

import json
import time

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import get_feature_columns
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def measure_single_row_latency(model, X: pd.DataFrame, n_repeats: int = 2000) -> dict:
    """Mede a latencia de predict_proba para UMA transacao por vez (o cenario real de
    producao), repetindo em linhas amostradas aleatoriamente para estabilizar a medida."""
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(X), size=n_repeats)

    latencies_ms = []
    for i in idx:
        row = X.iloc[[i]]
        start = time.perf_counter()
        model.predict_proba(row)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms = np.array(latencies_ms)
    return {
        "n_repeats": n_repeats,
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(latencies_ms.mean()),
        "max_ms": float(latencies_ms.max()),
    }


def measure_batch_throughput(model, X: pd.DataFrame, batch_size: int = 10_000) -> dict:
    """Mede quantas transacoes por segundo o modelo processa em lote (cenario de
    reprocessamento/backfill, nao de decisao em tempo real)."""
    batch = X.sample(n=min(batch_size, len(X)), random_state=42, replace=len(X) < batch_size)
    start = time.perf_counter()
    model.predict_proba(batch)
    elapsed = time.perf_counter() - start
    throughput = len(batch) / elapsed
    return {"batch_size": len(batch), "elapsed_s": elapsed, "throughput_per_s": float(throughput)}


def run_benchmark(config: dict | None = None) -> dict:
    config = config or load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    processed_dir = resolve_path(config["data"]["processed_dir"])

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    feature_cols = get_feature_columns(test_df, config)
    X_test = test_df[feature_cols]

    results = {}
    for label, filename in [
        ("baseline_logreg", "baseline_logreg.joblib"),
        ("lightgbm_fraud", "lightgbm_fraud.joblib"),
    ]:
        model_path = models_dir / filename
        if not model_path.exists():
            logger.warning(f"Modelo {model_path} nao encontrado, pulando.")
            continue
        model = joblib.load(model_path)

        logger.info(f"Medindo latencia de {label}...")
        single = measure_single_row_latency(model, X_test)
        batch = measure_batch_throughput(model, X_test)
        results[label] = {"single_row": single, "batch": batch}

        logger.info(
            f"{label}: p50={single['p50_ms']:.3f}ms p95={single['p95_ms']:.3f}ms "
            f"p99={single['p99_ms']:.3f}ms | throughput={batch['throughput_per_s']:,.0f} transacoes/s"
        )

    out_path = resolve_path(config["paths"]["reports_dir"]) / "latency_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Resultados de latencia salvos em {out_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
