"""Etapa 4 do pipeline: validacao, metricas e benchmark de performance/escala.

Por que PR-AUC (Average Precision) e a metrica principal, nao accuracy: com 0.17% de
fraude, um modelo que sempre prediz "nao fraude" acerta 99.83% (accuracy altissima e
inutil). PR-AUC resume precisao x recall ao longo de todos os thresholds e e o padrao
de mercado para deteccao de fraude/anomalias.

Threshold de decisao: nao usamos 0.5 por padrao. Sweepamos thresholds sobre o conjunto
de validacao e escolhemos o que minimiza um custo assimetrico explicito (deixar passar
uma fraude custa muito mais que bloquear por engano uma transacao legitima) — e assim
que threshold de fraude e definido na pratica.

Benchmark de escala: para provar que o treino nao vira gargalo em "big data", reamostramos
(bootstrap) o conjunto de treino em multiplos do tamanho original e medimos o tempo de
treino do LightGBM em cada multiplo — evidenciando crescimento sub-linear (nao O(n^2)).
"""

from __future__ import annotations

# IMPORTANTE (Windows): lightgbm precisa ser importado ANTES de pandas/pyarrow no mesmo
# processo. Nesta plataforma, se pandas (via pyarrow, usado no read_parquet) carrega sua
# DLL nativa primeiro, a DLL nativa do LightGBM entra em conflito de simbolos e qualquer
# chamada ao booster (fit/predict) falha com access violation — nao acontece em
# Linux/Colab, mas o import lightgbm-primeiro e inofensivo em qualquer SO, entao mantemos
# a ordem sempre assim por seguranca/portabilidade.
import lightgbm as lgb

import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier

from src.preprocessing import get_feature_columns
from src.utils import get_logger, load_config, timed

logger = get_logger(__name__)


def find_cost_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, config: dict) -> dict:
    cost_cfg = config["evaluation"]["cost_matrix"]
    fn_cost = cost_cfg["false_negative_cost"]
    fp_cost = cost_cfg["false_positive_cost"]

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best = {"threshold": 0.5, "total_cost": float("inf")}

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = fn * fn_cost + fp * fp_cost
        if total_cost < best["total_cost"]:
            best = {"threshold": float(t), "total_cost": float(total_cost), "fn": int(fn), "fp": int(fp)}

    logger.info(
        f"Threshold otimo por custo: {best['threshold']:.4f} "
        f"(FN={best['fn']}, FP={best['fp']}, custo_total={best['total_cost']:.0f})"
    )
    return best


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    preds = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1_score": float(f1_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(threshold),
    }


def cross_validate(model_ctor, train_df: pd.DataFrame, config: dict) -> list[float]:
    """K-Fold estratificado (nao temporal, so para checar variancia do modelo) sobre o
    conjunto de treino — reporta PR-AUC por fold para discutir overfitting/underfitting."""
    target_col = config["features"]["target_col"]
    feature_cols = get_feature_columns(train_df, config)
    k = config["evaluation"]["cv_folds"]
    seed = config["project"]["random_seed"]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    X, y = train_df[feature_cols], train_df[target_col]
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = model_ctor()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        score = average_precision_score(y.iloc[val_idx], proba)
        scores.append(score)
        logger.info(f"CV fold {fold}/{k}: PR-AUC={score:.4f}")

    logger.info(f"CV PR-AUC: media={np.mean(scores):.4f} desvio={np.std(scores):.4f}")
    return scores


def scale_benchmark(train_df: pd.DataFrame, config: dict, model_ctor) -> pd.DataFrame:
    """Reamostra o treino em multiplos crescentes de tamanho e mede o tempo de treino,
    provando que o pipeline escala para volumes tipo 'big data' sem crescimento explosivo."""
    target_col = config["features"]["target_col"]
    feature_cols = get_feature_columns(train_df, config)
    # SCALE_BENCHMARK_MULTIPLIERS permite encolher o benchmark (ex.: "1,5" em vez de
    # "1,2,5,10,20") sem editar config.yaml — usado pela esteira de CI, onde o runner
    # compartilhado e mais lento que uma maquina dedicada e o benchmark completo (que
    # sobe ate 20x/~4M linhas) sozinho ja passa de tempo de sobra do limite do job.
    env_multipliers = os.environ.get("SCALE_BENCHMARK_MULTIPLIERS", "").strip()
    if env_multipliers:
        multipliers = [int(x) for x in env_multipliers.split(",") if x.strip()]
    else:
        multipliers = config["evaluation"]["scale_benchmark"]["row_multipliers"]
    seed = config["project"]["random_seed"]
    rng = np.random.default_rng(seed)

    n_base = len(train_df)
    results = []
    for m in multipliers:
        n_target = n_base * m
        if m == 1:
            sample = train_df
        else:
            idx = rng.integers(0, n_base, size=n_target)
            sample = train_df.iloc[idx]

        model = model_ctor()
        start = time.perf_counter()
        model.fit(sample[feature_cols], sample[target_col])
        elapsed = time.perf_counter() - start

        results.append({"multiplier": m, "n_rows": n_target, "train_seconds": elapsed})
        logger.info(f"Benchmark de escala: {n_target:,} linhas ({m}x) -> treino em {elapsed:.2f}s")

    return pd.DataFrame(results)


def complexity_comparison_benchmark(train_df: pd.DataFrame, config: dict, lgbm_ctor) -> pd.DataFrame:
    """Mede a latencia de UMA predicao (o cenario real de producao) em funcao do numero
    de linhas de treino, para tres familias de modelo com complexidade assintotica de
    inferencia diferente:

    - KNN (busca exata por vizinhos, algorithm='brute'): O(n) por predicao — precisa
      comparar a query com todo o conjunto de treino.
    - Regressao Logistica: O(d) por predicao — independe do tamanho do treino.
    - LightGBM (arvores): O(k * profundidade) por predicao — cresce, na pratica, muito
      mais devagar que o tamanho do treino (profundidade e limitada, ~log2 do numero de
      folhas).

    Este e o experimento que sustenta, com numeros medidos e nao so teoria, a decisao de
    modelo do projeto: o mesmo motivo que da o desempenho preditivo (arvores de
    decisao/boosting) tambem da a performance de inferencia em producao.
    """
    target_col = config["features"]["target_col"]
    feature_cols = get_feature_columns(train_df, config)
    cfg = config["evaluation"]["complexity_comparison"]
    row_sizes = cfg["row_sizes"]
    n_repeats = cfg["n_query_repeats"]
    seed = config["project"]["random_seed"]
    rng = np.random.default_rng(seed)

    n_base = len(train_df)
    results = []
    for n in row_sizes:
        if n <= n_base:
            sample = train_df.sample(n=n, random_state=seed)
        else:
            idx = rng.integers(0, n_base, size=n)
            sample = train_df.iloc[idx]
        X, y = sample[feature_cols], sample[target_col]

        models = {
            "KNN (O(n))": KNeighborsClassifier(n_neighbors=5, algorithm="brute", n_jobs=-1),
            "Regressao Logistica (O(d))": LogisticRegression(max_iter=500, class_weight="balanced"),
            "LightGBM (O(k*log folhas))": lgbm_ctor(),
        }
        for model_name, model in models.items():
            model.fit(X, y)
            query_idx = rng.integers(0, n, size=n_repeats)
            latencies = []
            for i in query_idx:
                row = X.iloc[[i]]
                start = time.perf_counter()
                model.predict_proba(row)
                latencies.append((time.perf_counter() - start) * 1000)
            p50 = float(np.percentile(latencies, 50))
            results.append({"n_rows": n, "model": model_name, "latency_ms_p50": p50})
            logger.info(f"Complexidade | {model_name} | n={n:,} | latencia p50={p50:.4f}ms")

    return pd.DataFrame(results)


def learning_curve_data(model_ctor, train_df: pd.DataFrame, config: dict) -> dict:
    target_col = config["features"]["target_col"]
    feature_cols = get_feature_columns(train_df, config)
    seed = config["project"]["random_seed"]

    train_sizes, train_scores, val_scores = learning_curve(
        model_ctor(),
        train_df[feature_cols],
        train_df[target_col],
        cv=3,
        scoring="average_precision",
        train_sizes=np.linspace(0.1, 1.0, 6),
        random_state=seed,
        n_jobs=-1,
    )
    return {
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist(),
    }


# ---------------------------------------------------------------------------------
# Plots + orquestracao (roda tudo, salva metricas em reports/ e figuras em reports/figures)
# ---------------------------------------------------------------------------------


def _make_plots(test_metrics, cv_scores, scale_df, complexity_df, lc_data, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.utils import resolve_path

    sns.set_theme(style="whitegrid")
    out_dir = resolve_path(config["paths"]["figures_dir"])

    # matriz de confusao
    cm = test_metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred: Legitima", "Pred: Fraude"], yticklabels=["Real: Legitima", "Real: Fraude"])
    ax.set_title(f"Matriz de confusao (teste, threshold={test_metrics['threshold']:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "05_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # CV scores (overfitting/underfitting)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(1, len(cv_scores) + 1), cv_scores, color="#4C72B0")
    ax.axhline(np.mean(cv_scores), color="#C44E52", linestyle="--", label=f"media={np.mean(cv_scores):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("PR-AUC (validacao)")
    ax.set_title(f"Cross-validation ({len(cv_scores)}-fold) — desvio={np.std(cv_scores):.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_cross_validation.png", dpi=150)
    plt.close(fig)

    # curva de aprendizado
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lc_data["train_sizes"], lc_data["train_scores_mean"], "o-", label="Treino", color="#4C72B0")
    ax.plot(lc_data["train_sizes"], lc_data["val_scores_mean"], "o-", label="Validacao (CV)", color="#C44E52")
    ax.set_xlabel("Numero de amostras de treino")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Curva de aprendizado — LightGBM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "07_learning_curve.png", dpi=150)
    plt.close(fig)

    # benchmark de escala (linhas x tempo de treino) — log-log: escala log nos DOIS eixos
    # e o jeito correto de visualizar comportamento assintotico. Uma relacao linear
    # (O(n)) aparece como reta de inclinacao ~1; log-x com y linear (erro comum) faz
    # ate uma reta O(n) parecer uma explosao exponencial, distorcendo a leitura.
    n_rows = scale_df["n_rows"].values
    train_seconds = scale_df["train_seconds"].values
    reference_linear = train_seconds[0] * (n_rows / n_rows[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(n_rows, train_seconds, "o-", color="#55A868", label="Tempo medido (LightGBM)")
    ax.plot(n_rows, reference_linear, "--", color="gray", alpha=0.7, label="Referencia O(n) linear")
    ax.set_xlabel("Numero de linhas de treino")
    ax.set_ylabel("Tempo de treino (s)")
    ax.set_title("Benchmark de escala — tempo de treino vs. volume de dados (log-log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "08_scale_benchmark.png", dpi=150)
    plt.close(fig)

    # comparacao de complexidade de inferencia
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for model_name, group in complexity_df.groupby("model"):
        ax.plot(group["n_rows"], group["latency_ms_p50"], "o-", label=model_name)
    ax.set_xlabel("Numero de linhas de treino")
    ax.set_ylabel("Latencia por predicao (ms, mediana)")
    ax.set_title("Latencia de inferencia vs. volume de treino, por familia de modelo")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "09_complexity_comparison.png", dpi=150)
    plt.close(fig)

    logger.info(f"Figuras de validacao/performance salvas em {out_dir}")


def run_evaluation(config: dict | None = None):
    import json

    import joblib

    from src.utils import resolve_path

    config = config or load_config()
    processed_dir = resolve_path(config["data"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir = resolve_path(config["paths"]["reports_dir"])

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    feature_cols = get_feature_columns(train_df, config)
    target_col = config["features"]["target_col"]

    main_model = joblib.load(models_dir / "lightgbm_fraud.joblib")
    baseline_model = joblib.load(models_dir / "baseline_logreg.joblib")
    with open(models_dir / "lightgbm_best_params.json", "r", encoding="utf-8") as f:
        best_params = json.load(f)

    def lgbm_ctor():
        return lgb.LGBMClassifier(
            objective="binary", verbosity=-1, seed=config["project"]["random_seed"], n_jobs=-1, **best_params
        )

    # 1) threshold otimo por custo, calculado no conjunto de VALIDACAO (nao no teste, para
    #    nao vazar informacao do teste na escolha do ponto de operacao)
    val_proba = main_model.predict_proba(val_df[feature_cols])[:, 1]
    threshold_info = find_cost_optimal_threshold(val_df[target_col].values, val_proba, config)
    with open(models_dir / "decision_threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)

    # 2) metricas finais no conjunto de TESTE (nunca visto durante tuning/threshold)
    test_proba = main_model.predict_proba(test_df[feature_cols])[:, 1]
    test_metrics = compute_metrics(test_df[target_col].values, test_proba, threshold_info["threshold"])
    logger.info(f"Metricas (teste): {test_metrics}")

    # comparacao justa: o baseline tambem recebe seu proprio threshold otimizado por custo
    # na validacao (nao 0.5 fixo) — caso contrario a comparacao favorece artificialmente
    # um dos dois modelos
    baseline_val_proba = baseline_model.predict_proba(val_df[feature_cols])[:, 1]
    baseline_threshold_info = find_cost_optimal_threshold(val_df[target_col].values, baseline_val_proba, config)

    baseline_proba = baseline_model.predict_proba(test_df[feature_cols])[:, 1]
    baseline_metrics = compute_metrics(test_df[target_col].values, baseline_proba, baseline_threshold_info["threshold"])
    logger.info(f"Metricas baseline (teste, threshold otimizado={baseline_threshold_info['threshold']:.4f}): {baseline_metrics}")

    def baseline_ctor():
        params = config["training"]["logistic_regression"]
        return LogisticRegression(
            max_iter=params["max_iter"], class_weight=params["class_weight"],
            random_state=config["project"]["random_seed"],
        )

    # 3) cross-validation (overfitting/underfitting) com os hiperparametros tunados —
    #    K-Fold ESTRATIFICADO E EMBARALHADO (nao respeita ordem temporal), para servir de
    #    diagnostico de variancia do modelo. Comparar isso com a metrica no teste TEMPORAL
    #    (acima) e o proprio experimento que evidencia por que o split temporal importa:
    #    se o desempenho no teste temporal for bem mais baixo que na CV embaralhada, e
    #    sinal de que o padrao de fraude mudou ao longo do tempo (o modelo aprendeu algo
    #    que nao generaliza igualmente para o futuro) — o mesmo fenomeno de concept drift
    #    discutido na secao de monitoramento.
    with timed(logger, "Cross-validation (LightGBM)"):
        cv_scores = cross_validate(lgbm_ctor, train_df, config)
    with timed(logger, "Cross-validation (baseline)"):
        cv_scores_baseline = cross_validate(baseline_ctor, train_df, config)

    temporal_generalization_gap = float(np.mean(cv_scores) - test_metrics["average_precision"])
    logger.info(
        f"Gap de generalizacao temporal (LightGBM): CV embaralhada={np.mean(cv_scores):.4f} vs. "
        f"teste temporal={test_metrics['average_precision']:.4f} (gap={temporal_generalization_gap:.4f})"
    )

    # 4) curva de aprendizado
    with timed(logger, "Curva de aprendizado"):
        lc_data = learning_curve_data(lgbm_ctor, train_df, config)

    # 5) benchmark de escala (treino) e de complexidade de inferencia
    with timed(logger, "Benchmark de escala"):
        scale_df = scale_benchmark(train_df, config, lgbm_ctor)
    with timed(logger, "Benchmark de complexidade de inferencia"):
        complexity_df = complexity_comparison_benchmark(train_df, config, lgbm_ctor)

    _make_plots(test_metrics, cv_scores, scale_df, complexity_df, lc_data, config)

    summary = {
        "test_metrics_main_model": test_metrics,
        "test_metrics_baseline": baseline_metrics,
        "baseline_threshold_info": baseline_threshold_info,
        "cv_pr_auc_scores_lightgbm": cv_scores,
        "cv_pr_auc_mean_lightgbm": float(np.mean(cv_scores)),
        "cv_pr_auc_std_lightgbm": float(np.std(cv_scores)),
        "cv_pr_auc_scores_baseline": cv_scores_baseline,
        "cv_pr_auc_mean_baseline": float(np.mean(cv_scores_baseline)),
        "cv_pr_auc_std_baseline": float(np.std(cv_scores_baseline)),
        "temporal_generalization_gap_lightgbm": temporal_generalization_gap,
        "scale_benchmark": scale_df.to_dict(orient="records"),
        "complexity_comparison": complexity_df.to_dict(orient="records"),
    }
    with open(reports_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Resumo de avaliacao salvo em {reports_dir / 'evaluation_summary.json'}")

    return summary


if __name__ == "__main__":
    run_evaluation()
