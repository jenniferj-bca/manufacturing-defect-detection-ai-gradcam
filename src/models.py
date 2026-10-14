"""Custom and transfer-learning Keras models."""
from __future__ import annotations

import tensorflow as tf


def build_custom_cnn(input_shape=(224, 224, 3)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="defect_probability")(x)
    return tf.keras.Model(inputs, outputs, name="casting_custom_cnn")


def build_transfer_model(input_shape=(224, 224, 3), fine_tune: bool = False) -> tf.keras.Model:
    """Build EfficientNetB0 with a binary sigmoid head.

    ImageNet weights are optional at runtime: if downloading weights is unavailable,
    the model is still constructible with random initialization.
    """
    try:
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    except Exception:
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=input_shape
        )
    backbone.trainable = fine_tune
    if fine_tune:
        for layer in backbone.layers[:-30]:
            layer.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="defect_probability")(x)
    return tf.keras.Model(inputs, outputs, name="casting_efficientnetb0")


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model
