"""Model-agnostic Grad-CAM generation and image rendering utilities."""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the last convolutional feature output, including nested backbones."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            if any(isinstance(nested, tf.keras.layers.Conv2D) for nested in layer.layers):
                return layer.name
    raise ValueError("No convolutional feature layer was found for Grad-CAM.")


def make_gradcam(
    model: tf.keras.Model,
    image: tf.Tensor,
    layer_name: str | None = None,
    class_index: int = 1,
) -> np.ndarray:
    """Return a normalized heatmap for a single preprocessed image batch."""
    if image.shape.rank != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects a batch containing exactly one image.")
    target_name = layer_name or find_last_conv_layer(model)
    target_layer = model.get_layer(target_name)
    grad_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])
    with tf.GradientTape() as tape:
        feature_maps, prediction = grad_model(image, training=False)
        score = prediction[:, 0] if class_index == 1 else 1.0 - prediction[:, 0]
    gradients = tape.gradient(score, feature_maps)
    if gradients is None:
        raise RuntimeError(f"Gradients are unavailable for layer '{target_name}'.")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(feature_maps * weights[:, None, None, :], axis=-1)
    cam = tf.maximum(cam[0], 0.0)
    maximum = tf.reduce_max(cam)
    return (cam / (maximum + tf.keras.backend.epsilon())).numpy()


def make_visuals(
    original: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize the heatmap and return RGB jet heatmap and RGB uint8 overlay."""
    base = np.asarray(original)
    if base.ndim == 3:
        base = base[..., 0] if base.shape[-1] == 1 else base.mean(axis=-1)
    base = np.clip(base, 0, 255).astype(np.uint8)
    resized = tf.image.resize(heatmap[..., None], base.shape[:2]).numpy()[..., 0]
    color = (plt.colormaps["jet"](np.clip(resized, 0, 1))[..., :3] * 255).astype(np.uint8)
    base_rgb = np.repeat(base[..., None], 3, axis=-1).astype(np.float32)
    overlay = np.clip((1.0 - alpha) * base_rgb + alpha * color, 0, 255).astype(np.uint8)
    return color, overlay


def render_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a reusable Original | Heatmap | Overlay figure."""
    heatmap_rgb, overlay = make_visuals(image, heatmap)
    base = np.asarray(image)
    if base.ndim == 3:
        base = np.squeeze(base)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(base, cmap="gray")
    axes[1].imshow(heatmap_rgb)
    axes[2].imshow(overlay)
    for axis, title in zip(axes, ("Original", "Grad-CAM heatmap", "Overlay")):
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return figure, overlay
