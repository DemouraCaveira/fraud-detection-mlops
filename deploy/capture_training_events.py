"""Captura uma execucao real e completa da pipeline (ingestao -> pre-processamento ->
treino -> validacao -> explicabilidade -> drift), registrando o tempo real decorrido
entre cada etapa em `reports/training_events.json`.

Isso alimenta o modo "replay" do visualizador do ciclo de vida (frontend/): a UI nao
inventa duracoes nem numeros, ela reproduz — em velocidade ajustavel — uma execucao que
de fato aconteceu. Reaproveita exatamente as mesmas funcoes que `run_pipeline.py`,
`src/evaluate.py`, `src/explainability.py` e `monitoring/drift_monitor.py` ja usam; este
script so adiciona os timestamps e grava o resultado.

Uso (como modulo, para o import de `monitoring`/`src` resolver a partir da raiz do
projeto — mesmo padrao de `python -m src.evaluate`):
    python -m deploy.capture_training_events

Demora aproximadamente o mesmo tempo que rodar run_pipeline.py + evaluate +
explainability + drift em sequencia (alguns minutos, a maior parte no benchmark de
validacao) -- rode uma vez antes da aula, nao durante.
"""

from __future__ import annotations

# IMPORTANTE (Windows): lightgbm precisa ser importado antes de pandas/pyarrow no mesmo
# processo — ver nota completa em src/evaluate.py.
import lightgbm  # noqa: F401

import json
import time

from monitoring.drift_monitor import run_drift_monitoring
from src.evaluate import run_evaluation
from src.explainability import run_explainability
from src.ingestion import ingest, validate_schema
from src.preprocessing import run_preprocessing
from src.train import run_training
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)

_t0: float = 0.0
_events: list[dict] = []


def _now() -> float:
    return time.perf_counter() - _t0


def _emit(stage: str, type_: str, label: str, payload: dict | None = None) -> None:
    event = {"stage": stage, "type": type_, "label": label, "payload": payload or {}, "ts_offset_s": round(_now(), 3)}
    _events.append(event)
    logger.info(f"[{event['ts_offset_s']:7.2f}s] {stage}/{type_} — {label}")


def main() -> None:
    global _t0
    config = load_config()
    _t0 = time.perf_counter()
    target_col = config["features"]["target_col"]

    _emit("ingestao", "start", "Baixando/lendo dataset bruto (OpenML data_id=1597)")
    raw_df = ingest(config)
    validate_schema(raw_df, config)
    _emit("ingestao", "end", "Dataset validado", {
        "n_linhas": int(raw_df.shape[0]),
        "n_colunas": int(raw_df.shape[1]),
        "taxa_fraude_pct": float(raw_df[target_col].mean() * 100),
    })

    _emit("preprocessamento", "start", "Feature engineering + split temporal")
    train_df, val_df, test_df, _scaler = run_preprocessing(raw_df, config)
    _emit("preprocessamento", "end", "Splits prontos", {
        "n_treino": int(len(train_df)), "n_validacao": int(len(val_df)), "n_teste": int(len(test_df)),
    })

    _emit("treino", "start", "Baseline (LogReg) + tuning Optuna + LightGBM final")
    train_result = run_training(train_df, val_df, config)
    _emit("treino", "end", "Modelo treinado", {
        "n_arvores": int(train_result["main_model"].best_iteration_),
        "baseline_train_s": round(train_result["timings"]["baseline_train_s"], 3),
        "main_train_s": round(train_result["timings"]["main_train_s"], 3),
        "best_params": train_result["best_params"],
    })

    _emit("validacao", "start", "Cross-validation, threshold de custo, benchmarks de escala/complexidade")
    eval_summary = run_evaluation(config)
    _emit("validacao", "end", "Validacao concluida", {
        "pr_auc_teste": eval_summary["test_metrics_main_model"]["average_precision"],
        "roc_auc_teste": eval_summary["test_metrics_main_model"]["roc_auc"],
        "cv_pr_auc_mean": eval_summary["cv_pr_auc_mean_lightgbm"],
        "gap_generalizacao_temporal": eval_summary["temporal_generalization_gap_lightgbm"],
    })

    _emit("explicabilidade", "start", "SHAP TreeExplainer sobre o conjunto de teste")
    explain_summary = run_explainability(config)
    _emit("explicabilidade", "end", "Top atributos calculados", {
        "top_features": explain_summary["top_features_by_shap"][:5],
    })

    _emit("drift", "start", "PSI/KS — treino vs. teste e simulacao de choque em Amount")
    drift_report = run_drift_monitoring(config)
    psi_values = [r["psi"] for r in drift_report["real_drift_treino_vs_teste"]]
    max_psi_real = max(psi_values) if psi_values else 0.0
    thresholds = drift_report["thresholds"]
    status = (
        "significativo" if max_psi_real >= thresholds["significativo"]
        else "moderado" if max_psi_real >= thresholds["moderado"]
        else "estavel"
    )
    _emit("drift", "end", "Relatorio de drift gerado", {
        "psi_maximo_real": max_psi_real, "status_geral": status, "thresholds": thresholds,
    })

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    out_path = reports_dir / "training_events.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_events, f, indent=2, ensure_ascii=False)
    logger.info(f"{len(_events)} eventos capturados em {out_path} (duracao total: {_now():.1f}s)")


if __name__ == "__main__":
    main()
