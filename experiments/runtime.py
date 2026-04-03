from __future__ import annotations

import logging
from pathlib import Path

from .io_utils import ExperimentStore


def configure_logging(store: ExperimentStore, level: str = 'INFO', quiet: bool = False) -> Path:
    log_path = store.logs_dir / 'run.log'
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if not quiet:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return log_path
