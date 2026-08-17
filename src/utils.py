"""Utilitarios compartilhados: config, logging e caminhos. Ponto unico para nao repetir
boilerplate de carregamento de config em cada script do pipeline."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve um caminho do config.yaml relativo a raiz do projeto (nunca hardcoded)."""
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timed(logger: logging.Logger, label: str):
    """Context manager para medir e logar o tempo de uma etapa do pipeline em segundos."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{label} concluido em {elapsed:.3f}s")
