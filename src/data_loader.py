"""Dataset discovery and tf.data input pipelines for casting images."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

CLASS_NAMES = ("ok_front", "def_front")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_class_root(root: str | Path) -> Path:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    candidates = [root_path] + [p for p in root_path.rglob("*") if p.is_dir()]
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in CLASS_NAMES):
            return candidate
    raise FileNotFoundError(
        f"Could not find both {CLASS_NAMES} below {root_path}. "
        "See data/README.md for the expected layout."
    )


def collect_paths(root: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    class_root = resolve_class_root(root)
    paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        files = sorted(
            p for p in (class_root / class_name).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No images found in {class_root / class_name}")
        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))
    if not paths:
        raise ValueError(f"No supported images found below {class_root}")
    return np.asarray(paths), np.asarray(labels, dtype=np.int32)


def _load_image(path: tf.Tensor, label: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.grayscale_to_rgb(tf.cast(image, tf.float32))
    return image, tf.cast(label, tf.float32)


def build_datasets(
    root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, float]]:
    """Create stratified train/validation datasets and a deterministic test-like dataset.

    The Kaggle release has no explicit validation folder. The returned third dataset
    is the full discovered set and should be replaced with a separate test root when
    one is available.
    """
    paths, labels = collect_paths(root)
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        split = max(1, int(len(indices) * validation_split))
        val_indices.extend(indices[:split])
        train_indices.extend(indices[split:])

    def make(indices: list[int] | np.ndarray, shuffle: bool) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((paths[indices], labels[indices]))
        dataset = dataset.map(
            lambda path, label: _load_image(path, label, image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if shuffle:
            dataset = dataset.shuffle(min(len(indices), 2048), seed=seed, reshuffle_each_iteration=True)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    counts = np.bincount(labels, minlength=2)
    total = counts.sum()
    class_weights = {i: float(total / (2 * count)) for i, count in enumerate(counts) if count}
    return make(np.asarray(train_indices), True), make(np.asarray(val_indices), False), make(np.arange(len(paths)), False), class_weights


def preprocess_image(image_path: str | Path, image_size: int = 224) -> tf.Tensor:
    """Load one grayscale image and return a model-ready batch tensor."""
    image, _ = _load_image(tf.convert_to_tensor(str(image_path)), tf.constant(0), image_size)
    return tf.expand_dims(image, axis=0)
