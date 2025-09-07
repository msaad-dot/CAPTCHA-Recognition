# ================================================
# train_crnn_ctc.py — Train CRNN with CTC loss, evaluate, and save
# ================================================
import os, numpy as np, tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# pyright: reportMissingImports=false

from preprocess import load_data, CHAR_SET_LEN  # load dataset and alphabet size
from model import build_ctc_train_model, compute_time_steps

# ==============================
# Training Settings
# ==============================
EPOCHS = 30
BATCH_SIZE = 128
SAVE_DIR = "./saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_PATH  = os.path.join(SAVE_DIR, "crnn_ctc_best.keras")   # best checkpoint (val_loss)
FINAL_PATH = os.path.join(SAVE_DIR, "crnn_ctc_final.keras")  # final trained model

# Reproducibility: fix seeds (optional but recommended)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42); tf.random.set_seed(42)

# (Optional) Enable dynamic GPU memory allocation
try:
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# ==============================
# Metric Helpers
# ==============================
def ctc_greedy_decode(y_pred: np.ndarray, input_len: np.ndarray):
    """
    Decode predictions using greedy CTC decoding.
    Args:
      y_pred: model output (B, T, C)
      input_len: sequence lengths (B,1)
    Returns:
      list of predicted sequences (as lists of indices)
    """
    decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_len[:, 0], greedy=True)
    seq = decoded[0].numpy()  # shape (B, max_dec_len), padded with -1
    return [row[row != -1].tolist() for row in seq]

def exact_match_rate(true_dense: np.ndarray, pred_seqs: list[list[int]]) -> float:
    """Percentage of predictions that exactly match ground-truth labels."""
    ok = 0
    for i, seq in enumerate(pred_seqs):
        true = true_dense[i]
        true = true[true != -1].tolist()
        if seq == true:
            ok += 1
    return ok / len(pred_seqs) if pred_seqs else 0.0

def cer(true_dense: np.ndarray, pred_seqs: list[list[int]]) -> float:
    """
    Compute Character Error Rate (CER) using Levenshtein distance.
    Lower is better.
    """
    def lev(a, b):
        dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            dp[i][0] = i
        for j in range(1, len(b) + 1):
            dp[0][j] = j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[-1][-1]

    total_ed, total_len = 0, 0
    for i, seq in enumerate(pred_seqs):
        true = true_dense[i]
        true = true[true != -1].tolist()
        total_ed += lev(true, seq)
        total_len += max(1, len(true))
    return total_ed / total_len

# ==============================
# Load dataset
# ==============================
print("📥 Loading data…")
Xtr, Xte, Ytr_dense, Yte_dense, Ltr, Lte = load_data()
print("✅ Data ready!")

# Validate label indices are within range
vals_tr = Ytr_dense[Ytr_dense >= 0]
vals_te = Yte_dense[Yte_dense >= 0]
max_allowed = CHAR_SET_LEN - 1  # maximum valid index
print(f"[Sanity] label range train: min={vals_tr.min()} max={vals_tr.max()} | expected max <= {max_allowed}")
print(f"[Sanity] label range test : min={vals_te.min()} max={vals_te.max()} | expected max <= {max_allowed}")
assert vals_tr.max() <= max_allowed and vals_te.max() <= max_allowed, \
    f"Found label index > {max_allowed}. Fix CHARS in preprocess.py!"

# ==============================
# Build model
# ==============================
train_model, infer_model, time_reduction, NUM_CLASSES = build_ctc_train_model()

# Compute number of timesteps T after CNN downsampling
IMG_WIDTH = Xtr.shape[2]
time_steps = compute_time_steps(IMG_WIDTH, time_reduction)

# Sanity check: ensure T >= max label length (required by CTC)
max_lab = max(int(Ltr.max()), int(Lte.max()))
assert time_steps >= max_lab, f"CTC requires T ≥ max label length, got T={time_steps} < {max_lab}"

# ==============================
# Compile model
# ==============================
train_model.compile(
    optimizer=Adam(1e-3),
    loss=lambda y_true, y_pred: y_pred  # identity loss (CTC loss is already computed in graph)
)

# Prepare input lengths for CTC
in_len_tr = np.full((len(Xtr), 1), time_steps, dtype=np.int32)
in_len_te = np.full((len(Xte), 1), time_steps, dtype=np.int32)

# Dummy targets (since loss is identity)
dummy_tr = np.zeros((len(Xtr), 1), dtype=np.float32)
dummy_te = np.zeros((len(Xte), 1), dtype=np.float32)

# ==============================
# Callbacks
# ==============================
checkpoint = ModelCheckpoint(BEST_PATH, monitor="val_loss", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5, verbose=1)

# ==============================
# Train
# ==============================
print("🚀 Training…")
history = train_model.fit(
    x=[Xtr, Ytr_dense, in_len_tr, Ltr],
    y=dummy_tr,
    validation_data=([Xte, Yte_dense, in_len_te, Lte], dummy_te),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ==============================
# Evaluate model
# ==============================
print("\n🔎 Evaluating on validation set…")
y_pred = infer_model.predict(Xte, batch_size=max(128, BATCH_SIZE), verbose=1)
decoded = ctc_greedy_decode(y_pred, in_len_te)

em = exact_match_rate(Yte_dense, decoded)
cer_val = cer(Yte_dense, decoded)
print(f"\n🎯 Exact-Match: {em:.4f} | ✂️ CER: {cer_val:.4f}")

# Print a few decoded examples
for i in range(min(5, len(decoded))):
    true = Yte_dense[i]
    true = true[true != -1].tolist()
    print(f"GT idx: {true} | Pred idx: {decoded[i]}")

# ==============================
# Save trained model
# ==============================
infer_model.save(FINAL_PATH)
print(f"\n💾 Inference model saved: {FINAL_PATH}")
print(f"💾 Best checkpoint (by val_loss): {BEST_PATH}")
