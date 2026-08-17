"""Etapa 1 do pipeline: ingestao dos dados brutos.

Fonte: OpenML dataset id 1597 ("creditcard"), identico ao Kaggle Credit Card Fraud
Detection (284.807 transacoes de cartao de credito de titulares europeus, set/2013;
492 fraudes = 0.172%). Usamos OpenML em vez do Kaggle diretamente porque nao exige
autenticacao/API key — requisito de reprodutibilidade do projeto (qualquer pessoa
consegue rodar, sem credencial e sem caminho local).

Baixamos o parquet diretamente da API publica do OpenML (`parquet_url`), em vez de usar
`sklearn.datasets.fetch_openml`: o OpenML marca a coluna `Time` como `row_id_attribute`
(identificador de linha, nao atributo), entao `fetch_openml` a descarta do DataFrame por
padrao — e `Time` e essencial aqui (split temporal, feature de hora do dia). O download
direto preserva todas as 31 colunas originais.

Referencia: https://www.openml.org/d/1597
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from src.utils import get_logger, load_config, resolve_path, timed

logger = get_logger(__name__)

OPENML_PARQUET_URL_TEMPLATE = "https://data.openml.org/datasets/0000/{data_id:04d}/dataset_{data_id}.pq"


def ingest(config: dict | None = None) -> pd.DataFrame:
    """Baixa (ou le do cache local) o dataset bruto de transacoes e retorna um DataFrame."""
    config = config or load_config()
    cache_path = resolve_path(config["data"]["raw_cache_path"])

    if cache_path.exists():
        logger.info(f"Cache encontrado em {cache_path}, lendo direto (sem novo download).")
        return pd.read_parquet(cache_path)

    data_id = config["data"]["openml_id"]
    url = OPENML_PARQUET_URL_TEMPLATE.format(data_id=data_id)
    with timed(logger, f"Download do dataset via OpenML (data_id={data_id})"):
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        df = pd.read_parquet(io.BytesIO(response.content))

    # normaliza nome/tipo da coluna alvo: OpenML traz Class como categorica ('0'/'1')
    target_col = config["features"]["target_col"]
    df[target_col] = df[target_col].astype(int)

    df.to_parquet(cache_path, index=False)
    logger.info(f"Dataset cacheado em {cache_path} — {df.shape[0]} linhas, {df.shape[1]} colunas.")
    return df


def validate_schema(df: pd.DataFrame, config: dict) -> None:
    """Checagem minima de sanidade do dado bruto — falha cedo se o schema mudou."""
    target_col = config["features"]["target_col"]
    time_col = config["features"]["time_col"]
    amount_col = config["features"]["amount_col"]

    expected_cols = {target_col, time_col, amount_col, *[f"V{i}" for i in range(1, 29)]}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas esperadas ausentes no dataset bruto: {missing}")
    if df[target_col].nunique() != 2:
        raise ValueError(f"Coluna alvo '{target_col}' deveria ser binaria, encontrado {df[target_col].unique()}")
    if df.isnull().any().any():
        raise ValueError("Dataset bruto contem valores ausentes inesperados.")

    logger.info("Schema do dataset bruto validado com sucesso.")


if __name__ == "__main__":
    cfg = load_config()
    data = ingest(cfg)
    validate_schema(data, cfg)
    fraud_rate = data[cfg["features"]["target_col"]].mean()
    logger.info(f"Taxa de fraude: {fraud_rate:.4%} ({data[cfg['features']['target_col']].sum()} casos)")
