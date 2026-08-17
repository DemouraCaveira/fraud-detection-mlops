"""Analise exploratoria (EDA) — script exploratorio, NAO faz parte do pipeline produtivo
(run_pipeline.py). E rodado uma vez, manualmente, para gerar as figuras usadas no
relatorio e no notebook; um pipeline de retreino automatizado nao precisa re-gerar EDA
a cada execucao. Essa separacao (exploratorio vs. pipeline repetivel) e deliberada.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.ingestion import ingest
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def plot_class_balance(df: pd.DataFrame, config: dict, out_dir):
    target_col = config["features"]["target_col"]
    counts = df[target_col].value_counts().sort_index()
    fraud_rate = df[target_col].mean()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legitima (0)", "Fraude (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("Numero de transacoes (escala log)")
    ax.set_title(f"Desbalanceamento de classes — fraude = {fraud_rate:.4%} do total")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "01_class_balance.png", dpi=150)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame, config: dict, out_dir):
    target_col = config["features"]["target_col"]
    amount_col = config["features"]["amount_col"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, name, color in [(0, "Legitima", "#4C72B0"), (1, "Fraude", "#C44E52")]:
        subset = df.loc[df[target_col] == label, amount_col]
        axes[0].hist(subset, bins=50, alpha=0.6, label=name, color=color, density=True)
    axes[0].set_xlim(0, 500)
    axes[0].set_xlabel("Amount (valor da transacao)")
    axes[0].set_title("Distribuicao de valor (recortado em $500)")
    axes[0].legend()

    sns.boxplot(data=df, x=target_col, y=amount_col, ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Legitima", "Fraude"])
    axes[1].set_title("Amount por classe (escala log)")

    fig.tight_layout()
    fig.savefig(out_dir / "02_amount_distribution.png", dpi=150)
    plt.close(fig)


def plot_time_distribution(df: pd.DataFrame, config: dict, out_dir):
    target_col = config["features"]["target_col"]
    time_col = config["features"]["time_col"]

    hour_of_day = (df[time_col] % (24 * 3600)) / 3600.0
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, name, color in [(0, "Legitima", "#4C72B0"), (1, "Fraude", "#C44E52")]:
        mask = df[target_col] == label
        ax.hist(hour_of_day[mask], bins=48, alpha=0.6, label=name, color=color, density=True)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuicao de transacoes por hora do dia, por classe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "03_time_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_with_target(df: pd.DataFrame, config: dict, out_dir):
    target_col = config["features"]["target_col"]
    v_cols = [c for c in df.columns if c.startswith("V")]
    corr = df[v_cols + [target_col]].corr()[target_col].drop(target_col).sort_values()

    fig, ax = plt.subplots(figsize=(6, 8))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set_xlabel("Correlacao de Pearson com a classe (fraude=1)")
    ax.set_title("Correlacao dos componentes PCA (V1..V28) com fraude")
    fig.tight_layout()
    fig.savefig(out_dir / "04_correlation_target.png", dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame, config: dict):
    out_dir = resolve_path(config["paths"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_class_balance(df, config, out_dir)
    plot_amount_distribution(df, config, out_dir)
    plot_time_distribution(df, config, out_dir)
    plot_correlation_with_target(df, config, out_dir)

    logger.info(f"Figuras de EDA salvas em {out_dir}")


if __name__ == "__main__":
    cfg = load_config()
    raw_df = ingest(cfg)
    run_eda(raw_df, cfg)
