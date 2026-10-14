"""Single-image prediction with Grad-CAM explanation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from data_loader import preprocess_image
from gradcam import make_gradcam, render_gradcam


def predict_image(image_path: str, model_path: str = "models/best_model.keras", image_size: int = 224, output_path: str | None = None) -> dict:
    model = tf.keras.models.load_model(model_path)
    batch = preprocess_image(image_path, image_size)
    probability = float(model.predict(batch, verbose=0)[0, 0])
    label = "Defective" if probability >= 0.5 else "OK"
    confidence = probability if label == "Defective" else 1.0 - probability
    heatmap = make_gradcam(model, batch, class_index=1 if label == "Defective" else 0)
    original = tf.image.rgb_to_grayscale(batch[0]).numpy()
    destination = output_path or str(Path("reports/gradcam_examples") / f"{Path(image_path).stem}_gradcam.png")
    render_gradcam(original, heatmap, destination)
    return {"class": label, "confidence": confidence, "defect_probability": probability, "gradcam_path": destination}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default="models/best_model.keras")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        size = yaml.safe_load(handle)["data"]["image_size"]
    print(predict_image(args.image, args.model, size, args.output))
