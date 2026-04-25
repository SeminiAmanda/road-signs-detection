"""
GTSRB - German Traffic Sign Recognition Benchmark
Complete Preprocessing Pipeline
================================================
Covers:
  1. Load & resize images (Train + Test)
  2. Normalize pixel values
  3. Handle class imbalance (class weights + optional oversampling)
  4. Data augmentation (ImageDataGenerator)
  5. Train/validation split
  6. Save preprocessed arrays for reuse
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE       = 32        # resize all images to 32×32
NUM_CLASSES    = 43
VAL_SPLIT      = 0.2       # 20% of training data used for validation
RANDOM_SEED    = 42
BATCH_SIZE     = 32
BASE_PATH      = "./data/archive/"    # root folder containing Train/, Test/, CSVs
SAVE_ARRAYS    = True      # set True to save .npy files for fast reloading


# ─────────────────────────────────────────────
# STEP 1 — LOAD & RESIZE TRAINING IMAGES
# ─────────────────────────────────────────────
def load_train_data(base_path: str):
    """
    Reads all images from Train/0/ … Train/42/
    Returns:
        images : np.ndarray  shape (N, IMG_SIZE, IMG_SIZE, 3)  uint8
        labels : np.ndarray  shape (N,)                        int
    """
    images, labels = [], []
    train_path = os.path.join(base_path, "Train")

    for class_id in range(NUM_CLASSES):
        class_folder = os.path.join(train_path, str(class_id))
        if not os.path.isdir(class_folder):
            print(f"  [warn] missing folder: {class_folder}")
            continue
        for fname in sorted(os.listdir(class_folder)):
            if not fname.lower().endswith(".png"):
                continue
            img_path = os.path.join(class_folder, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                images.append(np.array(img))
                labels.append(class_id)
            except Exception as e:
                print(f"  [error] could not load {img_path}: {e}")

    print(f"Loaded {len(images)} training images across {NUM_CLASSES} classes.")
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


# ─────────────────────────────────────────────
# STEP 1b — LOAD & RESIZE TEST IMAGES
# ─────────────────────────────────────────────
def load_test_data(base_path: str):
    """
    Reads Test.csv and loads images from Test/ folder.
    Returns:
        images : np.ndarray  shape (N, IMG_SIZE, IMG_SIZE, 3)  uint8
        labels : np.ndarray  shape (N,)                        int
    """
    csv_path = os.path.join(base_path, "Test.csv")
    df = pd.read_csv(csv_path)

    images, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(base_path, row["Path"])
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            images.append(np.array(img))
            labels.append(int(row["ClassId"]))
        except Exception as e:
            print(f"  [error] could not load {img_path}: {e}")

    print(f"Loaded {len(images)} test images.")
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


# ─────────────────────────────────────────────
# STEP 2 — NORMALIZE PIXEL VALUES
# ─────────────────────────────────────────────
def normalize(images: np.ndarray) -> np.ndarray:
    """Scale uint8 [0, 255] → float32 [0.0, 1.0]"""
    return images.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# STEP 3 — CLASS IMBALANCE: COMPUTE CLASS WEIGHTS
# ─────────────────────────────────────────────
def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Returns a dict {class_id: weight} where minority classes
    receive higher weights so the model pays more attention to them.
    Pass this dict to model.fit(class_weight=class_weights).
    """
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=labels,
    )
    cw = dict(enumerate(weights))
    print("Class weights computed (sample):",
          {k: round(v, 3) for k, v in list(cw.items())[:5]}, "...")
    return cw


# ─────────────────────────────────────────────
# STEP 4 — DATA AUGMENTATION GENERATOR
# ─────────────────────────────────────────────
def build_augmentation_generator() -> ImageDataGenerator:
    """
    Returns an ImageDataGenerator with traffic-sign-safe augmentations.

    Key decisions:
      - horizontal_flip=False  → flipped signs are different sign classes
      - rotation_range=10      → mild tilt only; real signs don't tilt sharply
      - brightness_range       → simulate different lighting / weather
    """
    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20.0,   # subtle colour jitter
        horizontal_flip=False,      # NEVER flip traffic signs
        vertical_flip=False,
        fill_mode="nearest",
    )


# ─────────────────────────────────────────────
# STEP 5 — TRAIN / VALIDATION SPLIT
# ─────────────────────────────────────────────
def split_data(images: np.ndarray, labels: np.ndarray):
    """
    Stratified split so every class keeps its ratio in both sets.
    Returns X_train, X_val, y_train, y_val  (float32 images, int labels)
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        images, labels,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    print(f"Train: {X_tr.shape[0]} samples | Val: {X_val.shape[0]} samples")
    return X_tr, X_val, y_tr, y_val


# ─────────────────────────────────────────────
# OPTIONAL — SAVE / LOAD PREPROCESSED ARRAYS
# ─────────────────────────────────────────────
def save_arrays(save_dir: str, **arrays):
    """Save preprocessed numpy arrays to disk for fast reloading."""
    os.makedirs(save_dir, exist_ok=True)
    for name, arr in arrays.items():
        path = os.path.join(save_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"  Saved {path}  {arr.shape}  {arr.dtype}")


def load_arrays(save_dir: str, *names):
    """Load previously saved numpy arrays."""
    return [np.load(os.path.join(save_dir, f"{n}.npy")) for n in names]


# ─────────────────────────────────────────────
# MAIN — RUN THE FULL PIPELINE
# ─────────────────────────────────────────────
def preprocess_gtsrb(base_path: str = BASE_PATH):
    print("\n=== GTSRB Preprocessing Pipeline ===\n")

    # 1. Load raw images
    print("--- Step 1: Loading images ---")
    X_raw, y_raw       = load_train_data(base_path)
    X_test_raw, y_test = load_test_data(base_path)

    # 2. Normalize
    print("\n--- Step 2: Normalizing ---")
    X_all  = normalize(X_raw)
    X_test = normalize(X_test_raw)
    print(f"Pixel range after norm: [{X_all.min():.2f}, {X_all.max():.2f}]")

    # 3. Train / val split
    print("\n--- Step 3: Train/val split ---")
    X_train, X_val, y_train, y_val = split_data(X_all, y_raw)

    # 4. Class weights (for imbalance)
    print("\n--- Step 4: Class weights ---")
    class_weights = compute_class_weights(y_train)

    # 5. One-hot encode labels
    print("\n--- Step 5: One-hot encoding ---")
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat   = to_categorical(y_val,   NUM_CLASSES)
    y_test_cat  = to_categorical(y_test,  NUM_CLASSES)
    print(f"y_train_cat shape: {y_train_cat.shape}")

    # 6. Build augmentation generator
    print("\n--- Step 6: Augmentation generator ready ---")
    aug = build_augmentation_generator()
    train_gen = aug.flow(X_train, y_train_cat, batch_size=BATCH_SIZE, seed=RANDOM_SEED)

    # 7. (Optional) Save to disk
    if SAVE_ARRAYS:
        print("\n--- Step 7: Saving arrays ---")
        save_arrays(
            os.path.join(base_path, "preprocessed"),
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
        )

    print("\n=== Preprocessing complete ===")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Classes : {NUM_CLASSES}")

    return {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train_cat,
        "y_val":   y_val_cat,
        "y_test":  y_test_cat,
        "train_gen":     train_gen,       # augmented generator → pass to model.fit
        "class_weights": class_weights,  # pass to model.fit
    }

if __name__ == "__main__":
    preprocess_gtsrb()