"""
GTSRB - CNN Training (Simplified & Fixed)
==========================================
Usage:
    python scripts/train_model.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PREPROCESSED  = "./data/archive/preprocessed"
MODEL_PATH    = "./scripts/best_model.keras"
NUM_CLASSES   = 43
IMG_SIZE      = 32
BATCH_SIZE    = 64
EPOCHS        = 30


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("Loading data...")
    X_train = np.load(os.path.join(PREPROCESSED, "X_train.npy"))
    X_val   = np.load(os.path.join(PREPROCESSED, "X_val.npy"))
    X_test  = np.load(os.path.join(PREPROCESSED, "X_test.npy"))
    y_train = np.load(os.path.join(PREPROCESSED, "y_train.npy"))
    y_val   = np.load(os.path.join(PREPROCESSED, "y_val.npy"))
    y_test  = np.load(os.path.join(PREPROCESSED, "y_test.npy"))

    print(f"  y_train values: {y_train[:5]}  (min={y_train.min()}, max={y_train.max()})")

    # Only one-hot encode if labels are integers (not already encoded)
    if y_train.ndim == 1:
        print("  One-hot encoding labels...")
        y_train = to_categorical(y_train, NUM_CLASSES)
        y_val   = to_categorical(y_val,   NUM_CLASSES)
        y_test  = to_categorical(y_test,  NUM_CLASSES)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# BUILD MODEL (no BatchNorm — simpler & stable)
# ─────────────────────────────────────────────
def build_model():
    model = Sequential([
        # Block 1
        Conv2D(32, (3,3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        # Block 2
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        # Block 3
        Conv2D(128, (3,3), activation='relu', padding='same'),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        # Head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# TRAIN (no ImageDataGenerator — feed raw arrays)
# ─────────────────────────────────────────────
def train(model, X_train, X_val, y_train, y_val):
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    print("\n--- Test Set Evaluation ---")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")


# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.legend()

    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.legend()

    plt.tight_layout()
    plt.savefig('./scripts/training_curves.png', dpi=150)
    print("Saved → scripts/training_curves.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== GTSRB Training (Fixed) ===\n")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    model = build_model()
    history = train(model, X_train, X_val, y_train, y_val)
    evaluate(model, X_test, y_test)
    plot_history(history)
    print(f"\n✅ Done! Model saved → {MODEL_PATH}")