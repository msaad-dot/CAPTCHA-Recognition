# ================================================
# export_safe.py — Convert a trained CRNN model into a "safe" format
# ================================================
# Why?
# - The trained CRNN model (crnn_ctc_final.keras) may contain Lambda layers
#   that can cause compatibility issues when loading the model in different
#   TensorFlow/Keras versions.
# - This script rebuilds the model using a registered custom layer (CollapseHeight)
#   and reloads the trained weights, producing a portable "safe" model.
#
# Usage:
#   python export_safe.py --in_keras ./saved_model/crnn_ctc_final.keras \
#                         --out_keras ./saved_model/crnn_ctc_final_safe.keras
# ================================================

import os, zipfile, tempfile, argparse, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, Input
# pyright: reportMissingImports=false

# ==============================
# Global settings (match training model)
# ==============================
IMG_WIDTH, IMG_HEIGHT = 200, 50
NUM_CLASSES = 62 + 1        # 62 characters (A–Z, a–z, 0–9) + 1 CTC blank
CNN_DIMS = (64, 128, 256)   # CNN filters per block
RNN_UNITS = 256             # LSTM hidden units per direction
DROPOUT_CNN = 0.25
DROPOUT_RNN = 0.15

# ==============================
# CNN block (same as training model)
# ==============================
def conv_block(x, filters, pool):
    """Two 3x3 conv layers + BatchNorm + ReLU → MaxPool + optional Dropout."""
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool)(x)
    if DROPOUT_CNN > 0:
        x = layers.Dropout(DROPOUT_CNN)(x)
    return x

# ==============================
# Custom layer: CollapseHeight
# ==============================
# - Replaces the Lambda layer used during training
# - Averages over the height dimension to collapse (B, H, W, C) → (B, W, C)
class CollapseHeight(layers.Layer):
    def call(self, t):
        return tf.reduce_mean(t, axis=1)
    def get_config(self):
        return super().get_config()

# ==============================
# Safe CRNN model builder
# ==============================
def build_crnn_safe():
    """Rebuild CRNN architecture with custom CollapseHeight layer (no Lambdas)."""
    image = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

    # CNN feature extractor
    x = conv_block(image, CNN_DIMS[0], (2,2))   # Width → W/2
    x = conv_block(x,     CNN_DIMS[1], (2,2))   # Width → W/4
    x = conv_block(x,     CNN_DIMS[2], (2,1))   # Keep width
    x = layers.Conv2D(CNN_DIMS[2], 3, padding='same', activation='relu')(x)

    # Collapse height dimension
    x = CollapseHeight(name="collapse_height")(x)

    # Bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True, dropout=DROPOUT_RNN),
                             merge_mode='concat', name='bilstm_1')(x)
    x = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True, dropout=DROPOUT_RNN),
                             merge_mode='concat', name='bilstm_2')(x)

    # Per-timestep classification
    logits = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', name='softmax')(x)

    return models.Model(image, logits, name="crnn_ctc_infer_safe")

# ==============================
# Extract weights from .keras archive
# ==============================
def extract_weights_from_keras_archive(keras_path: str) -> str | None:
    """
    Extract .h5 weights file from a .keras zip archive.
    Returns path to the extracted .h5 file (temporary).
    """
    if not os.path.exists(keras_path):
        raise FileNotFoundError(keras_path)

    with zipfile.ZipFile(keras_path, 'r') as z:
        names = z.namelist()
        h5s = [n for n in names if n.endswith(".h5")]
        if not h5s:
            return None
        # Prefer file containing "weights", fallback to first .h5 file
        cand = next((n for n in h5s if "weights" in n), h5s[0])
        tmpdir = tempfile.mkdtemp()
        z.extract(cand, tmpdir)
        return os.path.join(tmpdir, cand)

# ==============================
# Main script
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_keras", required=True, help="Input .keras model (with Lambda)")
    ap.add_argument("--out_keras", required=True, help="Output safe .keras model")
    args = ap.parse_args()

    # Build safe architecture
    safe = build_crnn_safe()

    # Extract weights from original .keras
    wpath = extract_weights_from_keras_archive(args.in_keras)
    if not wpath:
        raise RuntimeError("No .h5 weights found inside the .keras archive.")

    # Load weights into safe model
    safe.load_weights(wpath)

    # Save safe model
    safe.save(args.out_keras)
    print("✅ Exported safe model:", args.out_keras)

if __name__ == "__main__":
    main()
