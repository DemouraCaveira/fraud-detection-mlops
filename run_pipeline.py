"""Entry point unico do pipeline produtivo — o que rodaria agendado/orquestrado em
producao (ex.: disparado por um job de retreino no EventBridge/Step Functions da
arquitetura AWS, ver architecture/aws_architecture.md).

Uso:
    python run_pipeline.py

Nao inclui EDA (src/eda.py) nem geracao dos graficos de validacao/monitoramento
(src/evaluate.py e monitoring/drift_monitor.py tambem tem seus proprios `__main__`) —
isso e deliberado: EDA e um passo exploratorio de uma vez so, nao um passo do pipeline
de retreino automatizado; e os relatorios de validacao/monitoramento sao consumidos por
humanos, nao pelo proprio pipeline. Rode-os separadamente quando quiser regenerar os
graficos/relatorios do projeto.
"""

from __future__ import annotations

# IMPORTANTE (Windows): ver nota de ordem de import em src/evaluate.py — lightgbm sempre
# precisa ser importado antes de pandas/pyarrow no mesmo processo.
import lightgbm  # noqa: F401

from src.ingestion import ingest, validate_schema
from src.preprocessing import run_preprocessing
from src.train import run_training
from src.utils import get_logger, load_config, timed

logger = get_logger(__name__)


def main():
    config = load_config()

    with timed(logger, "Pipeline completo"):
        with timed(logger, "1/3 Ingestao"):
            raw_df = ingest(config)
            validate_schema(raw_df, config)

        with timed(logger, "2/3 Pre-processamento"):
            train_df, val_df, test_df, _ = run_preprocessing(raw_df, config)

        with timed(logger, "3/3 Treinamento"):
            run_training(train_df, val_df, config)

    logger.info(
        "Pipeline concluido. Rode `python -m src.evaluate`, `python -m src.explainability` "
        "e `python -m monitoring.drift_monitor` para gerar metricas, explicabilidade e "
        "relatorio de drift."
    )


if __name__ == "__main__":
    main()
