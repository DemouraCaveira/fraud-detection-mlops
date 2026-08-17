"""Etapa 2 do pipeline: pre-processamento e engenharia de atributos.

Decisoes de design (documentadas aqui porque sao pontos de discussao em aula):

1. Split TEMPORAL, nao aleatorio. A coluna `Time` e o numero de segundos desde a
   primeira transacao do dataset (~2 dias corridos). Um split aleatorio (train_test_split
   padrao) misturaria transacoes futuras no treino e passadas no teste, o que nunca
   acontece em producao (o modelo so ve o passado) e infla artificialmente a metrica.
   Aqui ordenamos por Time e cortamos em fatias — treino=passado, teste=futuro.

2. Nao usamos SMOTE/oversampling. Em ~200k+ linhas de treino, gerar amostras sinteticas
   da classe minoritaria via SMOTE tem custo (busca de k-vizinhos) que cresce com o
   tamanho do dataset e nao escala bem para o cenario de "big data" que este projeto
   quer demonstrar. Em vez disso, o desbalanceamento e tratado via peso de classe
   (`class_weight`/`scale_pos_weight`), que e O(1) extra sobre o treino normal.

3. `Amount` recebe RobustScaler (usa mediana/IQR, robusto a outliers extremos de valor
   de transacao) em vez de StandardScaler. `V1..V28` ja vem padronizado do PCA original
   do dataset, entao nao sao re-escalados.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.utils import get_logger, load_config, resolve_path, timed

logger = get_logger(__name__)

SECONDS_IN_DAY = 24 * 60 * 60


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    time_col = config["features"]["time_col"]
    amount_col = config["features"]["amount_col"]

    df = df.copy()

    # hora do dia (ciclica) a partir dos segundos corridos - transacoes de fraude tendem
    # a se concentrar em horarios de menor vigilancia (madrugada), entao isso costuma ter
    # sinal preditivo real, nao so decorativo
    seconds_of_day = df[time_col] % SECONDS_IN_DAY
    hour_of_day = seconds_of_day / 3600.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)

    # amount em log1p reduz a assimetria (cauda longa de valores altos de transacao)
    df["amount_log"] = np.log1p(df[amount_col])

    return df


def temporal_split(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide o dataset por ordem cronologica (coluna Time), nao aleatoriamente."""
    time_col = config["features"]["time_col"]
    df_sorted = df.sort_values(time_col).reset_index(drop=True)

    n = len(df_sorted)
    train_frac = config["data"]["split"]["train_frac"]
    val_frac = config["data"]["split"]["val_frac"]

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    logger.info(
        f"Split temporal -> treino: {len(train_df)} ({train_df[config['features']['target_col']].mean():.4%} fraude), "
        f"val: {len(val_df)} ({val_df[config['features']['target_col']].mean():.4%} fraude), "
        f"teste: {len(test_df)} ({test_df[config['features']['target_col']].mean():.4%} fraude)"
    )
    return train_df, val_df, test_df


def fit_scaler(train_df: pd.DataFrame, config: dict) -> RobustScaler:
    amount_col = config["features"]["amount_col"]
    scaler = RobustScaler()
    scaler.fit(train_df[[amount_col]])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: RobustScaler, config: dict) -> pd.DataFrame:
    amount_col = config["features"]["amount_col"]
    df = df.copy()
    df[f"{amount_col}_scaled"] = scaler.transform(df[[amount_col]])
    return df


def get_feature_columns(df: pd.DataFrame, config: dict) -> list[str]:
    target_col = config["features"]["target_col"]
    time_col = config["features"]["time_col"]
    amount_col = config["features"]["amount_col"]
    exclude = {target_col, time_col, amount_col}
    return [c for c in df.columns if c not in exclude]


def run_preprocessing(raw_df: pd.DataFrame, config: dict | None = None):
    config = config or load_config()

    with timed(logger, "Engenharia de atributos"):
        df = engineer_features(raw_df, config)

    train_df, val_df, test_df = temporal_split(df, config)

    scaler = fit_scaler(train_df, config)
    train_df = apply_scaler(train_df, scaler, config)
    val_df = apply_scaler(val_df, scaler, config)
    test_df = apply_scaler(test_df, scaler, config)

    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    logger.info(f"Conjuntos processados salvos em {processed_dir}")

    models_dir = resolve_path(config["paths"]["models_dir"])
    joblib.dump(scaler, models_dir / "amount_scaler.joblib")

    return train_df, val_df, test_df, scaler


if __name__ == "__main__":
    from src.ingestion import ingest

    cfg = load_config()
    raw = ingest(cfg)
    run_preprocessing(raw, cfg)
