"""Prepara os dados (ingestão + pré-processamento) SEM treinar nada.

Usado pela branch `prod` da esteira: produção nunca re-treina o modelo — ela reaproveita
o artefato exato que já foi validado em `hom` (ver .github/workflows/ci.yml e
BRANCHING.md). Preparar os dados de novo em prod é aceitável (é um passo determinístico
e idempotente, não é o modelo em si), mas o modelo tem que ser o mesmo binário promovido,
nunca um recriado do zero.

Uso:
    python prepare_data.py
"""

from __future__ import annotations

# IMPORTANTE (Windows): ver nota de ordem de import em src/evaluate.py.
import lightgbm  # noqa: F401

from src.ingestion import ingest, validate_schema
from src.preprocessing import run_preprocessing
from src.utils import get_logger, load_config, timed

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    with timed(logger, "Preparacao de dados (ingestao + pre-processamento, sem treino)"):
        raw_df = ingest(config)
        validate_schema(raw_df, config)
        run_preprocessing(raw_df, config)


if __name__ == "__main__":
    main()
