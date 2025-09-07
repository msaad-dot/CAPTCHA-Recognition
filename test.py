# ==========================================
# test.py — robust inference for CRNN+CTC
# Safe/Lambda compatible + smart auto-crop + (variants for single image)
# ==========================================
import os, argparse, numpy as np, tensorflow as tf, zipfile, tempfile, cv2
from tensorflow.keras import layers, models
# pyright: reportMissingImports=false

# ---- from your project
from preprocess import _preprocess_img, indices_to_text, CHAR_SET_LEN
BLANK_INDEX = CHAR_SET_LEN  # 62
IMG_WIDTH, IMG_HEIGHT = 200, 50
NUM_CLASSES = 62 + 1
CNN_DIMS = (64, 128, 256)
RNN_UNITS = 256
DROPOUT_CNN = 0.25
DROPOUT_RNN = 0.15

# ---- register custom layer for safe models
try:
    from keras.saving import register_keras_serializable  # Keras 3
except Exception:
    from tensorflow.keras.utils import register_keras_serializable  # TF/Keras 2

@register_keras_serializable(package="custom", name="CollapseHeight")
class CollapseHeight(layers.Layer):
    def call(self, t):  # (B,H,W,C) -> (B,W,C)
        return tf.reduce_mean(t, axis=1)
    def get_config(self):
        return super().get_config()

# ---- same architecture (fallback)
def conv_block(x, filters, pool):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool)(x)
    if DROPOUT_CNN > 0:
        x = layers.Dropout(DROPOUT_CNN)(x)
    return x

def build_crnn_same():
    image = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    x = conv_block(image, CNN_DIMS[0], (2,2))   # W/2
    x = conv_block(x,     CNN_DIMS[1], (2,2))   # W/4
    x = conv_block(x,     CNN_DIMS[2], (2,1))   # keep W
    x = layers.Conv2D(CNN_DIMS[2], 3, padding='same', activation='relu')(x)
    x = CollapseHeight(name="collapse_height")(x)
    x = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True, dropout=DROPOUT_RNN),
                             merge_mode='concat', name='bilstm_1')(x)
    x = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True, dropout=DROPOUT_RNN),
                             merge_mode='concat', name='bilstm_2')(x)
    logits = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', name='softmax')(x)
    return models.Model(image, logits, name="crnn_ctc_infer_safe")

def extract_weights_from_keras_archive(keras_path: str):
    with zipfile.ZipFile(keras_path, 'r') as z:
        names = z.namelist()
        h5s = [n for n in names if n.endswith(".h5")]
        if not h5s:
            return None
        cand = next((n for n in h5s if "weights" in n), h5s[0])
        tmpdir = tempfile.mkdtemp()
        z.extract(cand, tmpdir)
        return os.path.join(tmpdir, cand)

def load_model_robust(path):
    # 1) Safe model
    try:
        return tf.keras.models.load_model(
            path, compile=False, custom_objects={"CollapseHeight": CollapseHeight}
        )
    except Exception:
        pass
    # 2) Legacy (Lambda)
    try:
        return tf.keras.models.load_model(
            path, compile=False, safe_mode=False, custom_objects={"CollapseHeight": CollapseHeight}
        )
    except Exception:
        pass
    # 3) Fallback: rebuild & load weights
    wpath = extract_weights_from_keras_archive(path)
    if not wpath:
        raise RuntimeError("No weights found inside .keras — run export_safe.py first.")
    m = build_crnn_same()
    m.load_weights(wpath)
    return m

# ---- greedy CTC decode
def ctc_greedy_decode(y_pred: np.ndarray):
    B, T, C = y_pred.shape
    in_len = np.full((B,), T, dtype=np.int32)
    decoded, _ = tf.keras.backend.ctc_decode(y_pred, in_len, greedy=True)
    seq = decoded[0].numpy()
    outs = []
    for i in range(B):
        row = seq[i]
        outs.append(row[row != -1].tolist())
    return outs

def iter_images(folder, limit=None):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    cnt = 0
    for r, _, fs in os.walk(folder):
        fs.sort()
        for f in fs:
            if f.lower().endswith(exts):
                yield os.path.join(r, f)
                cnt += 1
                if limit and cnt >= limit:
                    return

# =========================
# Smart auto-crop (threshold + edges) + strong preprocess
# =========================
def smart_crop(gray: np.ndarray) -> np.ndarray:
    """يحاول يطلع الـROI للنص: أوتسو (نغمتين) + كاني + dilate ثم أكبر مستطيل صالح."""
    if gray is None:
        return None
    H, W = gray.shape[:2]

    # 1) Otsu (polarity both)
    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, th     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)

    # 2) Canny edges
    v = np.median(gray)
    edges = cv2.Canny(gray, 0.66*v, 1.33*v)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cands = [th_inv, th, edges]
    boxes = []
    for m in cands:
        ys, xs = np.where(m > 0)
        if len(xs) < 30 or len(ys) < 30:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        # padding صغير
        pad = max(2, int(0.02 * max(H, W)))
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(W-1, x1 + pad); y1 = min(H-1, y1 + pad)
        boxes.append((x0, y0, x1, y1))

    if not boxes:
        return gray
    # اختار أكبر منطقة ذات نسبة معقولة
    def score(b):
        x0,y0,x1,y1 = b
        area = (x1-x0+1)*(y1-y0+1)
        ar = (x1-x0+1)/(y1-y0+1 + 1e-6)
        ok = 0.8 <= ar <= 8.0  # نص أفقي غالبًا
        return (area if ok else 0)
    best = max(boxes, key=score)
    x0,y0,x1,y1 = best
    if (x1-x0)<10 or (y1-y0)<10:
        return gray
    return gray[y0:y1+1, x0:x1+1]

def strong_enhance(gray: np.ndarray) -> np.ndarray:
    """تعزيز قوي للحروف الرفيعة (يُفعل مع --strong)."""
    g = gray.copy()
    # normalize contrast
    g = cv2.convertScaleAbs(g, alpha=1.4, beta=0)
    # اختَر قطبية تلقائيًا: لو الخلفية أغمق كتير → invert
    _, thO = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
    _, thI = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inv = (np.count_nonzero(thI) > np.count_nonzero(thO))
    if inv: g = 255 - g
    # سُمك بسيط
    g = cv2.dilate(g, np.ones((2,2), np.uint8), iterations=1)
    # unsharp mask خفيف
    blur = cv2.GaussianBlur(g, (0,0), 1.0)
    g = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
    return g

def to_model_tensor(img_gray: np.ndarray) -> np.ndarray:
    """resize keep-aspect + CLAHE + normalize → (50,200,1)"""
    h, w = img_gray.shape[:2]
    target_w, target_h = IMG_WIDTH, IMG_HEIGHT
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    rz = cv2.resize(img_gray, (nw, nh), interpolation=interp)
    pad_w, pad_h = target_w - nw, target_h - nh
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    mean_val = int(np.mean(rz))
    canv = cv2.copyMakeBorder(rz, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=mean_val)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    canv = clahe.apply(canv)
    canv = canv.astype(np.float32) / 255.0
    return canv.reshape(IMG_HEIGHT, IMG_WIDTH, 1)

def make_variants(gray: np.ndarray):
    k = np.ones((2,2), np.uint8)
    v0 = to_model_tensor(gray)
    v1 = to_model_tensor(255 - gray)
    v2 = to_model_tensor(cv2.dilate(gray, k, iterations=1))
    v3 = to_model_tensor(cv2.dilate(255 - gray, k, iterations=1))
    return [v0, v1, v2, v3]

def score_logits(logits: np.ndarray) -> float:
    if logits.ndim == 3:
        logits = logits[0]
    return float(logits.max(axis=1).mean())

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image")
    g.add_argument("--folder")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--no_auto_crop", action="store_true")
    ap.add_argument("--force_invert", action="store_true")
    ap.add_argument("--strong", action="store_true", help="enable strong enhancement for thin strokes")
    ap.add_argument("--save_debug", type=str, default=None)
    args = ap.parse_args()

    print("📦 Loading model:", args.model)
    infer = load_model_robust(args.model)
    print("✅ Model ready.")

    # -------- single image (variants) --------
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(args.image)
        raw = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if raw is None: raise RuntimeError("Failed to read image.")

        gray = raw if args.no_auto_crop else smart_crop(raw)
        if args.strong:
            gray = strong_enhance(gray)
        if args.force_invert:
            gray = 255 - gray

        if args.save_debug:
            os.makedirs(args.save_debug, exist_ok=True)
            cv2.imwrite(os.path.join(args.save_debug, "00_raw.png"), raw)
            cv2.imwrite(os.path.join(args.save_debug, "01_cropped.png"), gray)

        Xs = make_variants(gray)
        if args.save_debug:
            for i, t in enumerate(Xs):
                vis = (t.squeeze() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(args.save_debug, f"v{i}.png"), vis)

        X = np.stack(Xs, axis=0)
        y_all = infer.predict(X, batch_size=len(Xs), verbose=0)
        scores = [score_logits(y_all[i]) for i in range(len(Xs))]
        best_i = int(np.argmax(scores))
        best_logits = y_all[best_i:best_i+1]

        decoded, _ = tf.keras.backend.ctc_decode(
            best_logits, np.array([best_logits.shape[1]]), greedy=True
        )
        seq = decoded[0].numpy()[0]
        seq = seq[seq != -1].tolist()
        pred = indices_to_text(seq)
        print(f"{args.image} → {pred}  (variant={best_i}, score={scores[best_i]:.3f})")
        return

    # -------- folder (smart-crop per image) --------
    if not os.path.isdir(args.folder):
        raise FileNotFoundError(args.folder)
    paths = list(iter_images(args.folder, args.limit))
    if not paths:
        print("✗ No images found."); return

    imgs, keep = [], []
    for p in paths:
        gimg = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gimg is None: continue
        gimg = gimg if args.no_auto_crop else smart_crop(gimg)
        if args.strong:
            gimg = strong_enhance(gimg)
        if args.force_invert:
            gimg = 255 - gimg
        imgs.append(to_model_tensor(gimg)); keep.append(p)

    X = np.stack(imgs, axis=0)
    y = infer.predict(X, batch_size=args.batch_size, verbose=0)
    seqs = ctc_greedy_decode(y)
    preds = [indices_to_text(s) for s in seqs]
    for p, txt in zip(keep, preds):
        print(f"{p} → {txt}")

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        for g in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
    main()
