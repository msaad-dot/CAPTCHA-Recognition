# ================================
# model.py — CRNN (CNN + BiLSTM) with CTC-ready outputs
# ================================
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models, Input
# pyright: reportMissingImports=false

# ==============================
# Global configuration
# ==============================
IMG_WIDTH: int = 200   # Input image width
IMG_HEIGHT: int = 50   # Input image height

# Total number of classes = 62 (A–Z + a–z + 0–9) + 1 extra for CTC blank
NUM_CLASSES: int = 62 + 1

# Model capacity settings
CNN_DIMS: tuple[int, int, int] = (64, 128, 256)   # Number of filters per CNN block
RNN_UNITS: int = 256                             # LSTM hidden units per direction

# Regularization
DROPOUT_CNN: float = 0.25
DROPOUT_RNN: float = 0.15

# Mixed precision training option
USE_MIXED_PRECISION: bool = False  # Set True if GPU supports mixed_float16

if USE_MIXED_PRECISION:
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("⚙️ Mixed precision enabled: mixed_float16")
    except Exception as e:
        print(f"⚠️ Could not enable mixed precision: {e}")

# ==============================
# Building blocks
# ==============================
def conv_block(x: tf.Tensor, filters: int, pool_size: tuple[int, int]) -> tf.Tensor:
    """
    Convolutional block:
      - Two 3x3 convolutions + BatchNorm + ReLU
      - MaxPooling (controls width/height reduction)
      - Dropout
    """
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size)(x)
    if DROPOUT_CNN > 0:
        x = layers.Dropout(DROPOUT_CNN)(x)
    return x

# ==============================
# Model builders
# ==============================
def build_crnn(return_logits: bool = False):
    """
    Build the CRNN (Convolutional Recurrent Neural Network) inference model.
    Input:
      (B, 50, 200, 1) grayscale CAPTCHA images
    Output:
      (B, T, NUM_CLASSES) sequence of per-timestep probability distributions

    Returns:
      inference_model : tf.keras.Model
      time_reduction  : int (horizontal downsampling factor, usually 4)
    """
    image = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

    # CNN feature extractor
    x = image
    x = conv_block(x, CNN_DIMS[0], pool_size=(2, 2))  # Reduce width → W/2
    x = conv_block(x, CNN_DIMS[1], pool_size=(2, 2))  # Reduce width → W/4
    x = conv_block(x, CNN_DIMS[2], pool_size=(2, 1))  # Keep width (no further reduction)
    x = layers.Conv2D(CNN_DIMS[2], 3, padding='same', activation='relu')(x)

    # Collapse height dimension (average pooling across height)
    # (B, H', W', C) → (B, W', C)
    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1), name="collapse_height")(x)

    # Bidirectional LSTM encoder over time axis (width dimension)
    x = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True, dropout=DROPOUT_RNN),
                             merge_mode='concat', name='bilstm_1')(x)
    x = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True, dropout=DROPOUT_RNN),
                             merge_mode='concat', name='bilstm_2')(x)

    # Per-timestep classification
    logits = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', name='softmax')(x)

    inference_model = models.Model(inputs=image, outputs=logits, name="crnn_ctc_infer")

    # With two (2,2) pooling → width reduced by factor 4
    time_reduction = 4

    if return_logits:
        return inference_model, time_reduction
    return inference_model, time_reduction


def build_ctc_train_model():
    """
    Build a training model that computes CTC loss directly in the graph.

    Inputs:
      - image: (B, 50, 200, 1) preprocessed images
      - labels: (B, max_chars) int32, padded with -1
      - input_length: (B, 1) number of timesteps per sample (after CNN reduction)
      - label_length: (B, 1) true length of each label sequence

    Output:
      - ctc_loss: (B, 1) scalar loss value for each sample
    """
    infer, time_reduction = build_crnn(return_logits=True)

    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_length = Input(name='input_length', shape=(1,), dtype='int32')
    label_length = Input(name='label_length', shape=(1,), dtype='int32')

    y_pred = infer.get_layer('softmax').output  # (B, T, NUM_CLASSES)

    def ctc_lambda(args):
        y_true, y_pred_, in_len, lab_len = args
        return tf.keras.backend.ctc_batch_cost(y_true, y_pred_, in_len, lab_len)

    ctc_loss = layers.Lambda(ctc_lambda, name='ctc_loss')([labels, y_pred, input_length, label_length])

    train_model = models.Model(
        inputs=[infer.input, labels, input_length, label_length],
        outputs=ctc_loss,
        name="crnn_ctc_train",
    )

    return train_model, infer, time_reduction, NUM_CLASSES

# ==============================
# Utilities
# ==============================
def compute_time_steps(img_width: int = IMG_WIDTH, time_reduction: int = 4) -> int:
    """Return number of time steps T given input width and reduction factor."""
    return img_width // time_reduction

# ==============================
# Self-test (when run standalone)
# ==============================
if __name__ == "__main__":
    import numpy as np
    print("\n🔧 Building models…")
    train_model, infer_model, tr, num_classes = build_ctc_train_model()
    print(f"Model: {infer_model.name} | time_reduction={tr} | NUM_CLASSES={num_classes}")

    infer_model.summary(line_length=120)

    # Dry run with dummy images to confirm shapes
    dummy = np.zeros((2, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    y = infer_model.predict(dummy, verbose=0)
    print(f"\n√ Inference output shape: {y.shape}  (B, T, C)")
    print(f"   → T should be IMG_WIDTH/{tr} = {IMG_WIDTH//tr}")

    # Show training model summary
    train_model.summary(line_length=120)
    print("\nTip: compile as follows inside training script:")
    print("  train_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),")
    print("         loss={'ctc_loss': lambda y_true, y_pred: y_pred})")
