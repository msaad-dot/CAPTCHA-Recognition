# export_safe.py
# Usage:
#   python export_safe.py --in_keras ./saved_model/crnn_ctc_final.keras \
#                         --out_keras ./saved_model/crnn_ctc_final_safe.keras
import os, zipfile, tempfile, argparse, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, Input
# pyright: reportMissingImports=false

IMG_WIDTH, IMG_HEIGHT = 200, 50
NUM_CLASSES = 62 + 1
CNN_DIMS = (64, 128, 256)
RNN_UNITS = 256
DROPOUT_CNN = 0.25
DROPOUT_RNN = 0.15

def conv_block(x, filters, pool):
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

class CollapseHeight(layers.Layer):
    def call(self, t):
        return tf.reduce_mean(t, axis=1)
    def get_config(self):
        return super().get_config()

def build_crnn_safe():
    image = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    x = conv_block(image, CNN_DIMS[0], (2,2))
    x = conv_block(x,     CNN_DIMS[1], (2,2))
    x = conv_block(x,     CNN_DIMS[2], (2,1))
    x = layers.Conv2D(CNN_DIMS[2], 3, padding='same', activation='relu')(x)
    x = CollapseHeight(name="collapse_height")(x)
    x = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True, dropout=DROPOUT_RNN),
                             merge_mode='concat', name='bilstm_1')(x)
    x = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True, dropout=DROPOUT_RNN),
                             merge_mode='concat', name='bilstm_2')(x)
    logits = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', name='softmax')(x)
    return models.Model(image, logits, name="crnn_ctc_infer_safe")

def extract_weights_from_keras_archive(keras_path: str) -> str | None:
    if not os.path.exists(keras_path):
        raise FileNotFoundError(keras_path)
    with zipfile.ZipFile(keras_path, 'r') as z:
        names = z.namelist()
        h5s = [n for n in names if n.endswith(".h5")]
        if not h5s:
            return None
        cand = next((n for n in h5s if "weights" in n), h5s[0])
        tmpdir = tempfile.mkdtemp()
        z.extract(cand, tmpdir)
        return os.path.join(tmpdir, cand)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_keras", required=True)
    ap.add_argument("--out_keras", required=True)
    args = ap.parse_args()

    safe = build_crnn_safe()
    wpath = extract_weights_from_keras_archive(args.in_keras)
    if not wpath:
        raise RuntimeError("No .h5 weights found inside the .keras archive.")
    safe.load_weights(wpath)
    safe.save(args.out_keras)
    print("✅ Exported safe model:", args.out_keras)

if __name__ == "__main__":
    main()
