Project: CAPTCHA-Recognition (CRNN + CTC)

Overview
- A CRNN (CNN + BiLSTM) trained with CTC loss to recognize 5-character CAPTCHAs.
- Current character set: Uppercase A–Z, Lowercase a–z, Digits 0–9 (total 62).
- Input is preprocessed to 200×50 grayscale with CLAHE and normalization.

Repository Structure
.
├─ model.py         # CRNN model (inference + training graph with CTC)
├─ preprocess.py    # Data loader, image preprocessing, text↔index mapping
├─ train.py         # Training + evaluation + saving the inference model
├─ test.py          # Robust single-image / folder inference (greedy CTC)
├─ requirements.txt # Dependencies (minimal)
└─ saved_model/     # Checkpoints and exported model (.keras) will be saved here

Environment & Install
1) Create a virtual environment (recommended).
2) Install dependencies:
   pip install -r requirements.txt

Dataset
- Images must be named with the ground-truth label (e.g., `aB3xQ.png` → "aB3xQ").
- Fixed label length is 5 (MAX_CHARS=5).
- Update the dataset path in `preprocess.py`:
   DATA_DIR = "/content/data/data"

Training
Run:
   python train.py
- The script will:
  - Load & split data (train/test).
  - Build the CRNN+CTC model.
  - Train with callbacks (checkpointing, early stop, ReduceLROnPlateau).
  - Evaluate Exact-Match Accuracy and Character Error Rate (CER).
  - Save the inference model to: `saved_model/crnn_ctc_final.keras`
  - Save the best checkpoint by validation loss to: `saved_model/crnn_ctc_best.keras`

Inference
Single image:
   python test.py --model saved_model/crnn_ctc_final.keras --image path/to/captcha.png

Folder of images:
   python test.py --model saved_model/crnn_ctc_final.keras --folder path/to/images --limit 100

Helpful flags (test.py):
   --no_auto_crop     # disable smart auto-crop
   --force_invert     # force color inversion if background/foreground are flipped
   --strong           # stronger enhancement for thin strokes
   --save_debug out/  # write intermediate visuals

Character Set Notes
- Current setup matches 62 symbols:
    CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
- If you later restrict to lowercase+digits only:
  - Change CHARS in `preprocess.py`.
  - Update NUM_CLASSES in `model.py` to `len(CHARS) + 1` (for CTC blank).
  - Retrain.

Troubleshooting
- CUDA OOM: reduce batch size in `train.py` (BATCH_SIZE).
- Bad labels error: ensure filenames are 5 alphanumeric chars exactly (regex is in `preprocess.py`).
- Windows path issues: use raw strings (e.g., `r"C:\data\captcha"`).
