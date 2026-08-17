"""Etapa 5 do pipeline: explicabilidade via SHAP.

Usamos `shap.TreeExplainer`, que explora a propria estrutura de arvore do LightGBM para
calcular valores de Shapley de forma exata e rapida (poly-time na profundidade da arvore,
em vez do custo exponencial do calculo generico de Shapley) — mais um lugar onde a escolha
de um modelo baseado em arvore paga dividendo em performance, inclusive na etapa de
interpretacao.
"""

from __future__ import annotations

# ver nota de ordem de import em src/evaluate.py — lightgbm sempre antes de pandas/pyarrow
import lightgbm  # noqa: F401

import numpy as np
import pandas as pd
import shap

from src.preprocessing import get_feature_columns
from src.utils import get_logger, load_config, timed

logger = get_logger(__name__)


def compute_shap_values(model, df: pd.DataFrame, config: dict, sample_size: int = 5000):
    feature_cols = get_feature_columns(df, config)
    seed = config["project"]["random_seed"]

    sample = df.sample(n=min(sample_size, len(df)), random_state=seed) if len(df) > sample_size else df
    X = sample[feature_cols]

    with timed(logger, f"Calculo de SHAP values ({len(X)} amostras)"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

    return explainer, shap_values, X


def top_features_by_importance(shap_values, feature_cols: list[str], top_n: int = 10) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return importance_df


def explain_single_case(explainer, model, df_row: pd.DataFrame, feature_cols: list[str]):
    """Explica uma predicao individual (ex.: um caso de fraude real do conjunto de teste)."""
    X = df_row[feature_cols]
    shap_values = explainer(X)
    proba = model.predict_proba(X)[:, 1]
    return shap_values, proba


def _make_plots(shap_values, importance_df, single_case_shap, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils import resolve_path

    out_dir = resolve_path(config["paths"]["figures_dir"])

    fig = plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, show=False)
    plt.title("SHAP summary — impacto de cada atributo nas predicoes")
    plt.tight_layout()
    plt.savefig(out_dir / "11_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1], color="#4C72B0")
    ax.set_xlabel("|SHAP value| medio")
    ax.set_title("Top atributos por importancia (SHAP)")
    fig.tight_layout()
    fig.savefig(out_dir / "12_shap_feature_importance.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    shap.plots.waterfall(single_case_shap[0], show=False)
    plt.title("Explicacao de um caso individual de fraude (conjunto de teste)")
    plt.tight_layout()
    plt.savefig(out_dir / "13_shap_waterfall_fraud_case.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Figuras de explicabilidade salvas em {out_dir}")


def run_explainability(config: dict | None = None):
    import json

    import joblib

    from src.utils import resolve_path

    config = config or load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])
    processed_dir = resolve_path(config["data"]["processed_dir"])
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    target_col = config["features"]["target_col"]

    model = joblib.load(models_dir / "lightgbm_fraud.joblib")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    feature_cols = get_feature_columns(test_df, config)

    explainer, shap_values, X_sample = compute_shap_values(model, test_df, config)
    importance_df = top_features_by_importance(shap_values, feature_cols, top_n=10)
    logger.info(f"Top 10 atributos por SHAP:\n{importance_df.to_string(index=False)}")

    # explica um caso real de fraude do teste (o de maior probabilidade prevista, para
    # garantir que seja um caso "confiante" e didatico de mostrar em aula)
    fraud_cases = test_df[test_df[target_col] == 1].copy()
    fraud_cases["proba"] = model.predict_proba(fraud_cases[feature_cols])[:, 1]
    top_fraud_case = fraud_cases.sort_values("proba", ascending=False).head(1)
    single_case_shap, single_proba = explain_single_case(explainer, model, top_fraud_case, feature_cols)

    _make_plots(shap_values, importance_df, single_case_shap, config)

    summary = {
        "top_features_by_shap": importance_df.to_dict(orient="records"),
        "example_fraud_case": {
            "predicted_probability": float(single_proba[0]),
            "top_contributing_features": {
                feature_cols[i]: float(single_case_shap.values[0][i])
                for i in np.argsort(-np.abs(single_case_shap.values[0]))[:5]
            },
        },
    }
    with open(reports_dir / "explainability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Resumo de explicabilidade salvo em {reports_dir / 'explainability_summary.json'}")

    return summary


if __name__ == "__main__":
    run_explainability()
