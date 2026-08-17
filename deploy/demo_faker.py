"""Demonstracao funcional do modelo com dados sinteticos gerados via Faker.

Os atributos V1..V28 do dataset sao componentes de PCA anonimizados — nao tem
significado de negocio direto, entao nao da para "fakear" um V14 com Faker do jeito
que se fakeia um nome ou uma data. A estrategia aqui e a mesma usada por times de dados
para gerar dados sinteticos realistas sem vazar dados reais: amostra-se um vetor de
atributos real (por classe, legitima ou fraude) e aplica-se um pequeno jitter gaussiano
proporcional ao desvio-padrao de cada atributo — a transacao final e sintetica (nao
existe nos dados originais), mas estatisticamente plausivel para o modelo.

O Faker entra na camada de apresentacao: gera nome do titular do cartao, bandeira,
final do cartao e nome do estabelecimento — o "invólucro" legivel por humanos que
acompanha a transacao sintetica na demonstracao ao vivo, exatamente como um payload
real de autorizacao chegaria a uma API de fraude em producao.
"""

from __future__ import annotations

# ver nota de ordem de import em src/evaluate.py — lightgbm sempre antes de pandas/pyarrow
import lightgbm  # noqa: F401

import json
import time

import joblib
import numpy as np
import pandas as pd
from faker import Faker

from src.preprocessing import apply_scaler, engineer_features, get_feature_columns
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)
fake = Faker("pt_BR")


def _fit_class_jitter_stats(train_df: pd.DataFrame, config: dict) -> dict:
    target_col = config["features"]["target_col"]
    v_cols = [c for c in train_df.columns if c.startswith("V")]
    stats = {}
    for label in (0, 1):
        subset = train_df.loc[train_df[target_col] == label, v_cols]
        stats[label] = {"rows": subset, "std": subset.std()}
    return stats


def generate_synthetic_transaction(
    class_stats: dict, config: dict, force_class: int | None = None, jitter_scale: float = 0.05
) -> dict:
    """Gera uma transacao sintetica: amostra um vetor real da classe alvo e aplica jitter."""
    rng = np.random.default_rng()
    target_class = force_class if force_class is not None else rng.choice([0, 1], p=[0.97, 0.03])

    pool = class_stats[target_class]["rows"]
    std = class_stats[target_class]["std"]
    base_row = pool.sample(n=1).iloc[0]
    jitter = rng.normal(loc=0, scale=std.values * jitter_scale)
    synthetic_v = base_row.values + jitter

    amount = float(np.round(np.abs(rng.normal(loc=88, scale=120)), 2)) if target_class == 0 else float(
        np.round(np.abs(rng.normal(loc=340, scale=250)), 2)
    )
    seconds_now = float(rng.uniform(0, 172_800))  # dentro da janela de 2 dias do dataset original

    row = {v: synthetic_v[i] for i, v in enumerate(pool.columns)}
    row[config["features"]["amount_col"]] = amount
    row[config["features"]["time_col"]] = seconds_now

    metadata = {
        "titular": fake.name(),
        "cartao_final": fake.credit_card_number(card_type="visa")[-4:],
        "estabelecimento": fake.company(),
        "cidade": fake.city(),
        "_classe_geradora": target_class,  # apenas para o log da demo, nao entra no modelo
    }
    return {"features": row, "metadata": metadata}


def run_demo(n_transactions: int = 6):
    config = load_config()
    processed_dir = resolve_path(config["data"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    class_stats = _fit_class_jitter_stats(train_df, config)

    model = joblib.load(models_dir / "lightgbm_fraud.joblib")
    scaler = joblib.load(models_dir / "amount_scaler.joblib")
    threshold_path = models_dir / "decision_threshold.json"
    threshold = 0.5
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold = json.load(f)["threshold"]

    feature_cols = get_feature_columns(train_df, config)

    # forca pelo menos uma transacao claramente fraudulenta na demo, o resto e aleatorio
    forced_classes = [1] + [None] * (n_transactions - 1)

    print("\n" + "=" * 78)
    print("DEMONSTRACAO FUNCIONAL — transacoes sinteticas (Faker) + inferencia real")
    print("=" * 78)

    for i, forced in enumerate(forced_classes, start=1):
        txn = generate_synthetic_transaction(class_stats, config, force_class=forced)
        row_df = pd.DataFrame([txn["features"]])
        row_df = engineer_features(row_df, config)
        row_df = apply_scaler(row_df, scaler, config)

        start = time.perf_counter()
        proba = model.predict_proba(row_df[feature_cols])[0, 1]
        latency_ms = (time.perf_counter() - start) * 1000
        decision = "BLOQUEAR" if proba >= threshold else "APROVAR"

        meta = txn["metadata"]
        print(f"\n[{i}] {meta['titular']} — cartao final {meta['cartao_final']} — {meta['estabelecimento']} ({meta['cidade']})")
        print(f"    Valor: R$ {txn['features']['Amount']:.2f}")
        print(f"    Probabilidade de fraude: {proba:.4f} -> decisao: {decision}  (latencia: {latency_ms:.3f} ms)")

    print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    run_demo()
