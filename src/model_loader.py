"""Model loading and runtime configuration helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the project YAML configuration."""
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration file: {config_path}")
    return config


def load_model(model_path: str | Path) -> tf.keras.Model:
    """Load a trained Keras model and fail with a useful message if missing."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run the training pipeline first."
        )
    return tf.keras.models.load_model(path, compile=False)
