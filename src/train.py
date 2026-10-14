"""Train a custom CNN or EfficientNetB0 classifier."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tensorflow as tf
import yaml

from data_loader import build_datasets
from models import build_custom_cnn, build_transfer_model, compile_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train(config_path: str = "config.yaml", model_name: str | None = None) -> tf.keras.callbacks.History:
    config = load_config(config_path)
    tf.keras.utils.set_random_seed(config["seed"])
    data_cfg, train_cfg = config["data"], config["training"]
    train_ds, val_ds, _, class_weights = build_datasets(**{
        "root": data_cfg["root"], "image_size": data_cfg["image_size"],
        "batch_size": data_cfg["batch_size"], "validation_split": data_cfg["validation_split"],
        "seed": config["seed"],
    })
    selected = model_name or train_cfg["model"]
    model = build_custom_cnn((data_cfg["image_size"], data_cfg["image_size"], 3)) if selected == "custom" else build_transfer_model((data_cfg["image_size"], data_cfg["image_size"], 3))
    compile_model(model, train_cfg["learning_rate"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=train_cfg["patience"], restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(train_cfg["model_path"], monitor="val_roc_auc", mode="max", save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=train_cfg["epochs"], class_weight=class_weights, callbacks=callbacks)
    if selected != "custom" and train_cfg["fine_tune_epochs"] > 0:
        model = build_transfer_model((data_cfg["image_size"], data_cfg["image_size"], 3), fine_tune=True)
        model.set_weights(tf.keras.models.load_model(train_cfg["model_path"]).get_weights())
        compile_model(model, train_cfg["fine_tune_learning_rate"])
        model.fit(train_ds, validation_data=val_ds, epochs=train_cfg["fine_tune_epochs"], class_weight=class_weights, callbacks=callbacks)
    logging.info("Best model saved to %s", train_cfg["model_path"])
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=("custom", "efficientnet"))
    args = parser.parse_args()
    train(args.config, args.model)
