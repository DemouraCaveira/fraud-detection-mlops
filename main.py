"""Ponto de entrada de demonstração/produção do modelo de detecção de fraude.

Este é o arquivo para rodar em aula e mostrar o modelo "em operação" — não treina nada
(isso é `run_pipeline.py`), só carrega o modelo já treinado e o coloca para funcionar,
das duas formas que um modelo roda em produção:

  1. Modo demonstração (padrão): gera transações sintéticas (Faker) e roda inferência
     real sobre elas, imprimindo a decisão no terminal — sem precisar de navegador nem
     de ferramenta externa (Postman etc.), bom para uma demo rápida em sala.
  2. Modo servidor (--serve): sobe a API de inferência de verdade (FastAPI/uvicorn),
     o mesmo tipo de processo que rodaria atrás de um load balancer em produção —
     acessível em http://127.0.0.1:8000/docs, com Swagger interativo.

Uso:
    python main.py                    # demonstracao funcional (Faker + inferencia real)
    python main.py --n 10             # gera 10 transacoes em vez de 6
    python main.py --serve            # sobe a API de inferencia real (Ctrl+C para parar)
    python main.py --serve --port 8080

Pré-requisito: modelo treinado. Se faltar, este script avisa e para — rode
`python run_pipeline.py` primeiro (treina em ~1 minuto, ver README.md).
"""

from __future__ import annotations

# IMPORTANTE (Windows): lightgbm precisa ser importado antes de pandas/pyarrow no mesmo
# processo — ver nota completa em src/evaluate.py. Mantido aqui por seguranca, mesmo
# que este arquivo em si nao use lightgbm diretamente.
import lightgbm  # noqa: F401

import argparse
import sys

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def _ensure_model_trained(config: dict) -> None:
    models_dir = resolve_path(config["paths"]["models_dir"])
    model_path = models_dir / "lightgbm_fraud.joblib"
    if not model_path.exists():
        print(f"ERRO: nenhum modelo treinado encontrado em '{model_path}'.")
        print("Rode primeiro:  python run_pipeline.py   (treina em ~1 minuto)")
        sys.exit(1)


def run_serve(port: int) -> None:
    import uvicorn

    print(f"Subindo a API de inferencia em http://127.0.0.1:{port}/docs (Ctrl+C para parar)")
    uvicorn.run("deploy.api:app", host="127.0.0.1", port=port, reload=False)


def run_demo(n_transactions: int) -> None:
    from deploy.demo_faker import run_demo as run_faker_demo

    run_faker_demo(n_transactions=n_transactions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecção de Fraude em Cartão de Crédito — modelo em operação"
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Sobe a API de inferência real (FastAPI/uvicorn) em vez da demo de terminal",
    )
    parser.add_argument("--port", type=int, default=8000, help="Porta da API (usado com --serve)")
    parser.add_argument(
        "--n", type=int, default=6, help="Número de transações sintéticas na demo de terminal",
    )
    args = parser.parse_args()

    config = load_config()
    _ensure_model_trained(config)

    if args.serve:
        run_serve(args.port)
    else:
        run_demo(args.n)


if __name__ == "__main__":
    main()
