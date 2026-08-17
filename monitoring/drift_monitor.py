"""Monitoramento de drift — data drift (PSI/KS) e uma nota de performance drift.

Duas comparacoes, deliberadamente diferentes em natureza:

1. Drift REAL (nao simulado): treino vs. teste. Como o split e temporal (Fase 2), o
   conjunto de teste e literalmente "o futuro" em relacao ao treino — entao PSI/KS entre
   os dois e uma medida genuina de como a distribuicao de transacoes mudou ao longo das
   ~48h do dataset, nao um exercicio artificial.

2. Drift SIMULADO (rotulado como tal): perturbamos deliberadamente `Amount` num lote
   sintetico para mostrar o que o monitor detectaria em um cenario real de mudanca de
   comportamento (ex.: inflacao, mudanca de mix de produtos) — serve para demonstrar a
   sensibilidade do monitor, nao e reportado como um numero real do dataset.

PSI (Population Stability Index) é o padrão de mercado em risco/fraude para monitorar
mudanca de distribuicao de uma feature entre dois periodos:
  PSI = sum( (pct_atual - pct_referencia) * ln(pct_atual / pct_referencia) )
  ao longo dos bins de um histograma. Regra pratica: PSI < 0.1 sem mudanca relevante,
  0.1-0.25 mudanca moderada (investigar), > 0.25 mudanca significativa (alerta).
"""

from __future__ import annotations

# ver nota de ordem de import em src/evaluate.py — lightgbm sempre antes de pandas/pyarrow
import lightgbm  # noqa: F401

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.preprocessing import get_feature_columns
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25


def population_stability_index(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    bin_edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0  # feature quase constante, PSI nao se aplica

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(current), 1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def classify_psi(psi: float) -> str:
    if psi < PSI_MODERATE_THRESHOLD:
        return "estavel"
    if psi < PSI_SIGNIFICANT_THRESHOLD:
        return "moderado"
    return "significativo"


def compute_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, config: dict, label: str) -> pd.DataFrame:
    feature_cols = get_feature_columns(reference_df, config)
    rows = []
    for col in feature_cols:
        psi = population_stability_index(reference_df[col], current_df[col])
        ks_stat, ks_pvalue = ks_2samp(reference_df[col], current_df[col])
        rows.append({
            "feature": col,
            "psi": psi,
            "psi_status": classify_psi(psi),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
        })
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    n_significant = (df["psi_status"] == "significativo").sum()
    logger.info(f"[{label}] {n_significant}/{len(df)} features com PSI significativo (>{PSI_SIGNIFICANT_THRESHOLD})")
    return df


def simulate_drifted_batch(df: pd.DataFrame, config: dict, amount_shift_pct: float = 0.60) -> pd.DataFrame:
    """Gera um lote sintetico com drift deliberado em Amount (ex.: +60% em media), para
    demonstrar a sensibilidade do monitor — NAO representa dado real do dataset."""
    amount_col = config["features"]["amount_col"]
    drifted = df.copy()
    drifted[amount_col] = drifted[amount_col] * (1 + amount_shift_pct)
    drifted["amount_log"] = np.log1p(drifted[amount_col])
    return drifted


def _plot_drift(real_drift_df: pd.DataFrame, simulated_drift_df: pd.DataFrame, config: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = resolve_path(config["paths"]["figures_dir"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, title in [
        (axes[0], real_drift_df.head(15), "Drift real: treino vs. teste (top 15 features por PSI)"),
        (axes[1], simulated_drift_df.head(15), "Drift simulado: teste vs. lote com +60% em Amount"),
    ]:
        colors = ["#C44E52" if s == "significativo" else "#DD8452" if s == "moderado" else "#4C72B0"
                  for s in df["psi_status"]]
        ax.barh(df["feature"], df["psi"], color=colors)
        ax.axvline(PSI_MODERATE_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        ax.axvline(PSI_SIGNIFICANT_THRESHOLD, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("PSI")
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_dir / "10_drift_monitor.png", dpi=150)
    plt.close(fig)
    logger.info(f"Figura de drift salva em {out_dir / '10_drift_monitor.png'}")


def run_drift_monitoring(config: dict | None = None):
    config = config or load_config()
    processed_dir = resolve_path(config["data"]["processed_dir"])
    reports_dir = resolve_path(config["paths"]["reports_dir"])

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    real_drift_df = compute_drift_report(train_df, test_df, config, label="treino_vs_teste (real)")

    # referencia = TESTE (nao treino) aqui, de proposito: isola o efeito do choque
    # artificial em Amount, sem misturar com o drift real de hour_sin/cos ja existente
    # entre treino e teste (que dominaria a escala do grafico e esconderia o sinal
    # injetado) — a comparacao "de mercado" de um monitor de producao e sempre contra o
    # lote de referencia mais recente, nao contra o treino original
    drifted_batch = simulate_drifted_batch(test_df, config)
    simulated_drift_df = compute_drift_report(test_df, drifted_batch, config, label="teste_vs_lote_simulado")

    _plot_drift(real_drift_df, simulated_drift_df, config)

    report = {
        "real_drift_treino_vs_teste": real_drift_df.to_dict(orient="records"),
        "simulado_drift_amount_shift_60pct": simulated_drift_df.to_dict(orient="records"),
        "thresholds": {"moderado": PSI_MODERATE_THRESHOLD, "significativo": PSI_SIGNIFICANT_THRESHOLD},
    }
    out_path = reports_dir / "drift_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Relatorio de drift salvo em {out_path}")

    return report


if __name__ == "__main__":
    run_drift_monitoring()
