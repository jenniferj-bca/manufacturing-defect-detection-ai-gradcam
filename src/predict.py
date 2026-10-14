"""Image preprocessing and prediction helpers for the inspection app."""
from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_pil_image(image: Image.Image, image_size: int) -> tf.Tensor:
    """Convert an uploaded image to the same grayscale-to-RGB tensor used in training."""
    grayscale = image.convert("L")
    array = np.asarray(grayscale, dtype=np.float32)[..., None]
    tensor = tf.convert_to_tensor(array)
    tensor = tf.image.resize(tensor, [image_size, image_size])
    tensor = tf.image.grayscale_to_rgb(tensor)
    return tensor[None, ...]


def predict(model: tf.keras.Model, batch: tf.Tensor) -> dict[str, Any]:
    """Return binary defect probability, label, confidence, and both probabilities."""
    raw_output = np.asarray(model(batch, training=False)).reshape(-1)
    if raw_output.size != 1:
        raise ValueError("The loaded model must produce one binary output per image.")
    defect_probability = float(np.clip(raw_output[0], 0.0, 1.0))
    defective = defect_probability >= 0.5
    label = "Defective" if defective else "OK"
    confidence = defect_probability if defective else 1.0 - defect_probability
    return {
        "label": label,
        "confidence": confidence,
        "defect_probability": defect_probability,
        "ok_probability": 1.0 - defect_probability,
        "class_index": 1 if defective else 0,
    }
