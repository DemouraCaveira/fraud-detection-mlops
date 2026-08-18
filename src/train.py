"""Etapa 3 do pipeline: treinamento.

Dois modelos, deliberadamente escolhidos para contrastar complexidade computacional
(o eixo central de performance deste projeto):

- Baseline: Regressao Logistica. Inferencia O(d) no numero de atributos — muito rapida,
  mas fronteira de decisao linear, teto de desempenho mais baixo neste problema.
- Modelo principal: LightGBM (gradient boosting histograma-based). Cada arvore custa
  O(log n_folhas) por predicao (profundidade da arvore), e o ensemble e K arvores em
  paralelo/sequencia — ainda assim ordens de magnitude mais rapido que um modelo cuja
  inferencia escala com o tamanho do dataset de treino (ex.: KNN, O(n)). O treino em si
  usa o algoritmo histogram-based do LightGBM, que discretiza os atributos em bins e
  escala aproximadamente O(n_amostras x n_atributos) por iteracao, mas com uma constante
  muito menor que boosting exato — e o que permite treinar em segundos mesmo com milhoes
  de linhas (ver benchmark de escala em evaluate.py).

Tuning de hiperparametros via Optuna (otimizacao bayesiana / TPE): em vez de varrer
exaustivamente uma grade (Grid Search, custo exponencial no numero de hiperparametros),
o Optuna usa os resultados de tentativas anteriores para guiar a proxima, convergindo
para uma boa regiao do espaco de busca em muito menos avaliacoes.
"""

from __future__ import annotations

import json
import os
import time

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.preprocessing import get_feature_columns
from src.utils import get_logger, load_config, resolve_path, timed

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_baseline(train_df: pd.DataFrame, config: dict) -> tuple[LogisticRegression, float]:
    target_col = config["features"]["target_col"]
    feature_cols = get_feature_columns(train_df, config)
    params = config["training"]["logistic_regression"]

    model = LogisticRegression(
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
        random_state=config["project"]["random_seed"],
    )
    start = time.perf_counter()
    model.fit(train_df[feature_cols], train_df[target_col])
    elapsed = time.perf_counter() - start
    logger.info(f"Baseline (Regressao Logistica) treinado em {elapsed:.3f}s")
    return model, elapsed


def tune_lightgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict) -> dict:
    """Busca bayesiana (Optuna/TPE) com early stopping DENTRO de cada trial: o boosting
    para assim que a metrica de validacao para de melhorar, em vez de sempre treinar as
    ate `n_estimators` arvores. Isso e o que torna 25-30 trials pratico em minutos, e nao
    em dezenas de minutos — a maior parte dos trials converge (ou se mostra ruim) bem
    antes do teto de arvores configurado."""
    target_col = config["features"]["target_col"]
    feature_cols = get_feature_columns(train_df, config)
    hpo_cfg = config["training"]["hpo"]
    space = hpo_cfg["search_space"]
    seed = config["project"]["random_seed"]
    early_stopping_rounds = hpo_cfg.get("early_stopping_rounds", 30)

    # HPO_N_TRIALS permite sobrescrever o numero de trials sem editar config.yaml — usado
    # pela esteira de CI para rodar um tuning "leve" na branch dev (feedback rapido) e o
    # tuning completo em hom/prod (mesmo espaco de busca, validado de verdade). O workflow
    # do GitHub Actions sempre DEFINE a env var (mesmo vazia em hom/prod), entao checamos
    # string vazia explicitamente em vez de confiar so no default do os.environ.get.
    env_n_trials = os.environ.get("HPO_N_TRIALS", "").strip()
    n_trials = int(env_n_trials) if env_n_trials else hpo_cfg["n_trials"]

    n_pos = train_df[target_col].sum()
    n_neg = len(train_df) - n_pos
    scale_pos_weight = n_neg / n_pos

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "verbosity": -1,
            "seed": seed,
            "scale_pos_weight": scale_pos_weight,
            "num_leaves": trial.suggest_int("num_leaves", *space["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", *space["min_child_samples"]),
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", *space["feature_fraction"]),
            "bagging_fraction": trial.suggest_float("bagging_fraction", *space["bagging_fraction"]),
            "bagging_freq": trial.suggest_int("bagging_freq", *space["bagging_freq"]),
        }
        model = lgb.LGBMClassifier(**params, n_jobs=-1)
        model.fit(
            train_df[feature_cols],
            train_df[target_col],
            eval_X=val_df[feature_cols],
            eval_y=val_df[target_col],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
        )
        val_proba = model.predict_proba(val_df[feature_cols])[:, 1]
        return average_precision_score(val_df[target_col], val_proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=hpo_cfg["timeout_seconds"])
    elapsed = time.perf_counter() - start

    logger.info(
        f"Optuna: {len(study.trials)} trials em {elapsed:.1f}s — "
        f"melhor PR-AUC (val)={study.best_value:.4f}, params={study.best_params}"
    )
    best_params = dict(study.best_params)
    best_params["scale_pos_weight"] = scale_pos_weight
    return best_params, study


def train_lightgbm(
    train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict, best_params: dict
) -> tuple[lgb.LGBMClassifier, float]:
    target_col = config["features"]["target_col"]
    feature_cols = get_feature_columns(train_df, config)
    seed = config["project"]["random_seed"]
    early_stopping_rounds = config["training"]["hpo"].get("early_stopping_rounds", 30)

    model = lgb.LGBMClassifier(
        objective="binary", verbosity=-1, seed=seed, n_jobs=-1, **best_params
    )
    start = time.perf_counter()
    model.fit(
        train_df[feature_cols],
        train_df[target_col],
        eval_X=val_df[feature_cols],
        eval_y=val_df[target_col],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    elapsed = time.perf_counter() - start
    logger.info(
        f"LightGBM final treinado em {elapsed:.3f}s ({len(train_df)} linhas, "
        f"{model.best_iteration_} arvores apos early stopping)"
    )
    return model, elapsed


def run_training(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict | None = None):
    config = config or load_config()
    models_dir = resolve_path(config["paths"]["models_dir"])

    with timed(logger, "Treino do baseline"):
        baseline_model, baseline_time = train_baseline(train_df, config)
    joblib.dump(baseline_model, models_dir / "baseline_logreg.joblib")

    with timed(logger, "Tuning de hiperparametros (Optuna)"):
        best_params, study = tune_lightgbm(train_df, val_df, config)

    with timed(logger, "Treino do modelo principal (LightGBM)"):
        main_model, main_train_time = train_lightgbm(train_df, val_df, config, best_params)
    joblib.dump(main_model, models_dir / "lightgbm_fraud.joblib")

    # persistido para reconstruir o mesmo modelo (sem early stopping) em CV/benchmarks
    # de evaluate.py, e para documentar no relatorio qual tuning foi de fato realizado
    best_params_with_fixed_estimators = dict(best_params)
    best_params_with_fixed_estimators["n_estimators"] = main_model.best_iteration_
    with open(models_dir / "lightgbm_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_with_fixed_estimators, f, indent=2)

    return {
        "baseline_model": baseline_model,
        "main_model": main_model,
        "best_params": best_params,
        "optuna_study": study,
        "timings": {"baseline_train_s": baseline_time, "main_train_s": main_train_time},
    }


if __name__ == "__main__":
    import pandas as pd

    cfg = load_config()
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    run_training(train_df, val_df, cfg)
