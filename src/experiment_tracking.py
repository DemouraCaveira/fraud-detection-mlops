"""Etapa de registro de experimentos: MLflow local (tracking URI = pasta `mlruns/`, sem
servidor externo — mlflow ui pode ser aberto depois localmente com `mlflow ui`).

Roda DEPOIS de src/train.py e src/evaluate.py: le os artefatos e metricas ja calculados
(nao retreina nada) e registra duas runs — baseline e modelo principal — para que
qualquer pessoa consiga comparar hiperparametros, metricas e artefatos entre execucoes
do pipeline ao longo do tempo, o proposito central de um model registry/experiment
tracker em um time de ML de verdade.
"""

from __future__ import annotations

# ver nota de ordem de import em src/evaluate.py — lightgbm sempre antes de pandas/pyarrow
import lightgbm  # noqa: F401

import json

import mlflow

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def log_experiments(config: dict | None = None):
    config = config or load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    tracking_uri = resolve_path(config["paths"]["mlflow_tracking_uri"])

    mlflow.set_tracking_uri(f"file:///{tracking_uri.as_posix()}")
    mlflow.set_experiment(config["project"]["name"])

    with open(reports_dir / "evaluation_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(models_dir / "lightgbm_best_params.json", "r", encoding="utf-8") as f:
        best_params = json.load(f)

    # run 1: baseline
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("class_weight", config["training"]["logistic_regression"]["class_weight"])
        mlflow.log_metrics({
            f"test_{k}": v for k, v in summary["test_metrics_baseline"].items() if isinstance(v, (int, float))
        })
        mlflow.log_artifact(str(models_dir / "baseline_logreg.joblib"))
        logger.info("Run 'baseline_logistic_regression' registrada no MLflow.")

    # run 2: modelo principal (tunado)
    with mlflow.start_run(run_name="lightgbm_tuned"):
        mlflow.log_params(best_params)
        mlflow.log_param("hpo_method", config["training"]["hpo"]["method"])
        mlflow.log_param("hpo_n_trials", config["training"]["hpo"]["n_trials"])
        mlflow.log_metrics({
            f"test_{k}": v for k, v in summary["test_metrics_main_model"].items() if isinstance(v, (int, float))
        })
        mlflow.log_metric("cv_pr_auc_mean", summary["cv_pr_auc_mean_lightgbm"])
        mlflow.log_metric("cv_pr_auc_std", summary["cv_pr_auc_std_lightgbm"])
        mlflow.log_metric("temporal_generalization_gap", summary["temporal_generalization_gap_lightgbm"])
        mlflow.log_artifact(str(models_dir / "lightgbm_fraud.joblib"))
        mlflow.log_artifact(str(reports_dir / "evaluation_summary.json"))
        logger.info("Run 'lightgbm_tuned' registrada no MLflow.")

    logger.info(f"Tracking local em: {tracking_uri} — abrir com `mlflow ui --backend-store-uri {tracking_uri}`")


if __name__ == "__main__":
    log_experiments()
