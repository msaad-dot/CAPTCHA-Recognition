Project: CAPTCHA Recognition (CRNN + CTC)

Overview
--------
This project implements a deep learning model for recognizing text-based CAPTCHAs using a Convolutional Recurrent Neural Network (CRNN) trained with Connectionist Temporal Classification (CTC) loss.

- Character set: Uppercase A–Z, Lowercase a–z, Digits 0–9 (62 symbols total).
- Input images are resized to 200×50 grayscale, enhanced with CLAHE, normalized, and passed into the CRNN.

Repository Structure
--------------------

├─ model.py         # CRNN definition (inference + training with CTC)

├─ preprocess.py    # Dataset loader and preprocessing

├─ train.py         # Training and evaluation script

├─ test.py          # Inference (single image or folder)

├─ requirements.txt # Dependencies

└─ saved_model/     # Saved checkpoints (.keras)

Dataset
-------
- Source: Kaggle — Captcha Dataset (123k images)
  https://www.kaggle.com/datasets/fournierp/captcha-version-2

- Each file name encodes the ground-truth text (e.g., `aB3xQ.png` → "aB3xQ").
- Labels are fixed length = 5 (`MAX_CHARS=5` in preprocess.py).
- Update the dataset path in preprocess.py:
  DATA_DIR = "/path/to/dataset"

Training
--------
Run:
   python train.py

The script will:
- Preprocess dataset and split into train/test.
- Train CRNN with CTC loss (30 epochs by default).
- Save checkpoints to `saved_model/crnn_ctc_best.keras`.
- Export the final model to `saved_model/crnn_ctc_final.keras`.

Evaluation
----------
Metrics:
- Exact Match Accuracy (all 5 characters correct).
- Character Error Rate (CER).

Example results (with Kaggle dataset, 62-class setup):
- Exact Match Accuracy: ~0.76
- CER: ~0.09

Inference
---------
Single image:
   python test.py --model saved_model/crnn_ctc_final.keras --image path/to/captcha.png

Folder of images:
   python test.py --model saved_model/crnn_ctc_final.keras --folder path/to/images --limit 50

Optional flags (test.py):
- --no_auto_crop   disable smart cropping
- --force_invert   force invert colors
- --strong         stronger preprocessing for thin fonts
- --save_debug out/ save intermediate debug images

Examples
--------
Here are some samples from the dataset (add them under a `docs/` folder):

![sample1](docs/sample1.jpg)
![sample2](docs/sample2.jpg)

Notes
-----
- To switch character sets (e.g., lowercase + digits only):
  * Edit `CHARS` in preprocess.py.
  * Update `NUM_CLASSES = len(CHARS) + 1` in model.py.
  * Retrain the model.
- Make sure CNN time steps ≥ label length (checked in train.py).

Author
------
Mohamed Saad
