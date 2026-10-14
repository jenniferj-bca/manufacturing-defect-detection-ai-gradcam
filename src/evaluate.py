"""Evaluate a saved model and export metrics and confusion matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from data_loader import build_datasets


def evaluate(config_path: str = "config.yaml", model_path: str | None = None) -> dict:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_cfg, train_cfg, eval_cfg = config["data"], config["training"], config["evaluation"]
    _, _, dataset, _ = build_datasets(data_cfg["root"], data_cfg["image_size"], data_cfg["batch_size"], data_cfg["validation_split"], config["seed"])
    path = model_path or train_cfg["model_path"]
    model = tf.keras.models.load_model(path)
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    probabilities = model.predict(dataset, verbose=0).ravel()
    y_pred = (probabilities >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, target_names=["OK", "Defective"], digits=4)
    auc = roc_auc_score(y_true, probabilities)
    output = Path(eval_cfg["report_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{report}\nROC-AUC: {auc:.4f}\n", encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["OK", "Defective"], yticklabels=["OK", "Defective"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(eval_cfg["confusion_matrix_path"], dpi=160); plt.close()
    return {"classification_report": report, "roc_auc": float(auc), "confusion_matrix": matrix.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model")
    args = parser.parse_args()
    print(evaluate(args.config, args.model))
