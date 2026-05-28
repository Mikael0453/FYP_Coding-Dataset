import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ─────────────────────────────────────────────
#  NORMALIZATION  (must match gestura_app.py)
# ─────────────────────────────────────────────
def normalize_landmarks(landmarks):
    """
    Translate wrist to origin, then scale by hand size.
    Input : 1-D array of 63 values  [x0,y0,z0, x1,y1,z1, ...]
    Output: 1-D array of 63 normalised values
    """
    lm = np.array(landmarks, dtype=np.float32).reshape(21, 3)

    # 1. Translate — wrist (landmark 0) becomes (0, 0, 0)
    lm -= lm[0].copy()

    # 2. Scale — divide by the largest absolute value so all coords ∈ [-1, 1]
    scale = np.max(np.abs(lm)) + 1e-6
    lm /= scale

    return lm.flatten()


# ─────────────────────────────────────────────
#  DATA AUGMENTATION
# ─────────────────────────────────────────────
def augment_landmarks(landmarks, n=3):
    """
    Return n augmented copies of a normalised landmark vector.
    Applied AFTER normalisation so scale is consistent.
    """
    augmented = []
    for _ in range(n):
        lm = landmarks.copy()
        lm += np.random.normal(0, 0.005, lm.shape)   # tiny noise
        lm *= np.random.uniform(0.92, 1.08)           # slight scale jitter
        lm += np.random.uniform(-0.02, 0.02)          # slight position jitter
        augmented.append(lm)
    return augmented


# ─────────────────────────────────────────────
#  LOAD & PREPROCESS
# ─────────────────────────────────────────────
df = pd.read_csv("hand_landmarks_all.csv")

X_raw = df.drop(["label", "category"], axis=1).values
y_raw = df["label"].values

print(f"Loaded {len(X_raw)} samples across {len(np.unique(y_raw))} classes")

# Normalise every row
X_norm = np.array([normalize_landmarks(row) for row in X_raw])

# Verify normalisation
sample = X_norm[0]
print(f"Normalisation check → wrist xyz: {sample[0]:.4f}, {sample[1]:.4f}, {sample[2]:.4f}  "
      f"(should be ~0)")
print(f"Value range → min: {X_norm.min():.4f}  max: {X_norm.max():.4f}  "
      f"(should be within [-1, 1])")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
joblib.dump(label_encoder, "label_encoder.pkl")
num_classes = len(label_encoder.classes_)
print(f"Classes ({num_classes}): {list(label_encoder.classes_)}")

# ─────────────────────────────────────────────
#  AUGMENT TRAINING DATA
# ─────────────────────────────────────────────
# Split first so augmentation only touches training set
X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Augment training set (3 extra copies per sample)
X_aug, y_aug = [], []
for x, y in zip(X_train, y_train):
    X_aug.append(x)
    y_aug.append(y)
    for x_a in augment_landmarks(x, n=3):
        X_aug.append(x_a)
        y_aug.append(y)

X_train = np.array(X_aug)
y_train = np.array(y_aug)
print(f"Training set after augmentation: {len(X_train)} samples")

# One-hot encode
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val,   num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,          # EarlyStopping will cut this short
    batch_size=32,
    callbacks=callbacks,
)


# ─────────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# Per-class report
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(max(8, num_classes), max(6, num_classes - 2)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved → confusion_matrix.png")

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],    label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy'); ax1.legend()
ax2.plot(history.history['loss'],    label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss'); ax2.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Training curves saved  → training_curves.png")


# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────
model.save("bim_hand_sign_model.keras")
print("\nModel saved → bim_hand_sign_model.keras")
print("Encoder saved → label_encoder.pkl")