# ================================
# preprocess.py (CRNN + CTC) — data preprocessing utilities
# ================================
import os, re, cv2, numpy as np
from sklearn.model_selection import train_test_split

# ==============================
# Global Settings
# ==============================
DATA_DIR   = "/content/data/data"  # Path to your dataset folder
IMG_WIDTH  = 200                   # Target image width after preprocessing
IMG_HEIGHT = 50                    # Target image height after preprocessing
MAX_CHARS  = 5                     # Maximum number of characters in a CAPTCHA (used for label padding)

# Character set: Uppercase A–Z + Lowercase a–z + Digits 0–9
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
CHAR_SET_LEN = len(CHARS)
assert CHAR_SET_LEN == 62, f"Expected 62 symbols, got {CHAR_SET_LEN}"
BLANK_INDEX  = CHAR_SET_LEN        # Extra index reserved for CTC blank

# Lookup tables for converting characters ↔ indices
char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for i, c in enumerate(CHARS)}

# Regex pattern: filenames must match exactly 5 alphanumeric chars (e.g., "aB3xQ.png")
valid_name_re = re.compile(r"^[A-Za-z0-9]{5}$")


# ==============================
# Image Preprocessing Helpers
# ==============================

def _resize_keep_aspect(img, target_w=IMG_WIDTH, target_h=IMG_HEIGHT):
    """
    Resize an image while keeping aspect ratio, then pad to (target_w, target_h).
    
    - Scaling:
        Uses INTER_CUBIC if scaling up (for smoother enlargement).
        Uses INTER_AREA if scaling down (for sharper shrinking).
    - Padding:
        Pads with the mean intensity of the image, so borders blend with background.
    
    Returns:
        A padded image with shape (target_h, target_w).
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)             # scaling factor to fit within target size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))  # new width/height

    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    pad_w, pad_h = target_w - nw, target_h - nh
    left, right  = pad_w // 2, pad_w - pad_w // 2
    top,  bottom = pad_h // 2, pad_h - pad_h // 2

    mean_val = int(np.mean(resized))  # padding color close to average background
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=mean_val
    )
    return padded


def _preprocess_img(path: str):
    """
    Complete preprocessing pipeline for a single image:
      1) Load in grayscale.
      2) Resize with aspect ratio and pad to (200x50).
      3) Apply CLAHE (adaptive histogram equalization) for better local contrast.
      4) Normalize pixel values to [0,1].
      5) Reshape to (H, W, 1) for CNN input.
    
    Returns:
        Preprocessed image (50, 200, 1) or None if load failed.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = _resize_keep_aspect(img, IMG_WIDTH, IMG_HEIGHT)

    # Optional denoising step (disabled by default).
    # Use this if your dataset has random impulse noise/dots.
    # img = cv2.medianBlur(img, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    return img.reshape(IMG_HEIGHT, IMG_WIDTH, 1)


def text_to_indices(text: str) -> list[int]:
    """Convert a text string into a list of integer indices based on CHARS."""
    return [char_to_idx[c] for c in text]


def indices_to_text(indices: list[int]) -> str:
    """Convert a list of indices back into a text string, ignoring invalid indices."""
    return "".join(idx_to_char[i] for i in indices if 0 <= i < CHAR_SET_LEN)


# ==============================
# Dataset Loader
# ==============================

def load_data(limit: int | None = None, test_size: float = 0.1, seed: int = 42):
    """
    Load and preprocess CAPTCHA dataset.

    Args:
        limit (int|None): Limit number of files loaded (for debugging). Default None = load all.
        test_size (float): Proportion of dataset for testing. Default 0.1 (10%).
        seed (int): Random seed for reproducibility.

    Returns:
        X_train, X_test: numpy arrays of images (N, 50, 200, 1)
        y_train, y_test: dense label arrays padded with -1 (N, MAX_CHARS)
        len_train, len_test: actual label lengths (N, 1), required for CTC
    """
    images, labels, label_lens = [], [], []
    skipped = {"bad_len": 0, "bad_chars": 0, "read_fail": 0}

    files = [f for f in os.listdir(DATA_DIR)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    if limit:
        files = files[:limit]

    for f in files:
        name = os.path.splitext(f)[0]

        # Validate filename format against regex
        if not valid_name_re.match(name):
            if len(name) != MAX_CHARS:
                skipped["bad_len"] += 1  # wrong length
            else:
                skipped["bad_chars"] += 1  # contains unsupported chars
            continue

        path = os.path.join(DATA_DIR, f)
        img = _preprocess_img(path)
        if img is None:
            skipped["read_fail"] += 1
            continue

        images.append(img)
        seq = text_to_indices(name)
        labels.append(seq)
        label_lens.append(len(seq))

    X = np.array(images, dtype=np.float32)

    # Create dense labels: shape (N, MAX_CHARS), fill with -1
    y_dense = -np.ones((len(labels), MAX_CHARS), dtype=np.int32)
    for i, seq in enumerate(labels):
        y_dense[i, :len(seq)] = np.array(seq, dtype=np.int32)

    # Store label lengths as (N,1)
    y_len = np.array(label_lens, dtype=np.int32).reshape(-1, 1)

    # Split dataset into train and test
    X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
        X, y_dense, y_len, test_size=test_size, random_state=seed, shuffle=True
    )

    print(f"✅ Loaded {len(X)} images | ❌ Skipped: {skipped}")
    print(f"Split → Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, len_train, len_test
