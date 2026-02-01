import os
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# =========================================================
# CONFIG (keep it simple)
# =========================================================
DATA_ROOT = "data/raw"  # contains class folders: COVID/, Normal/, Viral Pneumonia/
CLASSES = ["COVID", "Normal", "Viral Pneumonia"]
IMG_SUBFOLDER = "images"  # inside each class folder

# Dataset metadata may say 256x256.
# We resize to 224x224 to match the common MobileNetV2 pretraining input size.
IMG_SIZE = (224, 224)

BATCH_SIZE = 32
SEED = 1234
EPOCHS = 6

OUTPUT_DIR = "results"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)


# =========================================================
# 1) Collect filepaths + labels
# =========================================================
filepaths = []
labels = []

for label, cls in enumerate(CLASSES):
    img_dir = os.path.join(DATA_ROOT, cls, IMG_SUBFOLDER)

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Missing folder: {img_dir}")

    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            filepaths.append(os.path.join(img_dir, fname))
            labels.append(label)

filepaths = np.array(filepaths)
labels = np.array(labels)

print("Total images:", len(filepaths))
for i, cls in enumerate(CLASSES):
    print(cls, ":", (labels == i).sum())


# =========================================================
# 2) Train/Val split 80/20 PER CLASS (manual stratified)
# =========================================================
train_files, train_labels = [], []
val_files, val_labels = [], []

for label, cls in enumerate(CLASSES):
    cls_files = filepaths[labels == label]

    idx = np.random.permutation(len(cls_files))  # shuffle inside class
    cls_files = cls_files[idx]

    split = int(0.8 * len(cls_files))

    train_files.extend(cls_files[:split])
    val_files.extend(cls_files[split:])

    train_labels.extend([label] * split)
    val_labels.extend([label] * (len(cls_files) - split))

train_files = np.array(train_files)
train_labels = np.array(train_labels)
val_files = np.array(val_files)
val_labels = np.array(val_labels)

print("\nSplit summary (80/20 per class):")
print("Train:", len(train_files))
print("Validation:", len(val_files))
for i, cls in enumerate(CLASSES):
    print(cls, "→ train:", (train_labels == i).sum(), "| val:", (val_labels == i).sum())

print("\nExample train file:", train_files[0])
print("Example val file:", val_files[0])


# =========================================================
# 3) tf.data pipeline: decode -> resize -> normalize
#    + data augmentation ONLY for training
# =========================================================
# Operator-dependent variability in CXR: small rotation/shift/zoom are reasonable.
# We avoid flips to preserve left-right anatomy.
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomRotation(factor=0.05, seed=SEED),       # ~ +/- 9 degrees
        keras.layers.RandomTranslation(0.05, 0.05, seed=SEED),     # up to 5% shift
        keras.layers.RandomZoom(0.10, seed=SEED),                  # up to 10% zoom
    ],
    name="augmentation",
)

def decode_resize_normalize(path, label, training=False):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)

    # normalize to [0,1]
    img = tf.cast(img, tf.float32) / 255.0

    # augmentation only on training set
    if training:
        img = data_augmentation(img)

    return img, label

def make_ds(files, labs, training=True):
    ds = tf.data.Dataset.from_tensor_slices((files, labs))
    if training:
        ds = ds.shuffle(buffer_size=min(5000, len(files)), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: decode_resize_normalize(x, y, training=training),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(train_files, train_labels, training=True)
val_ds   = make_ds(val_files,   val_labels,   training=False)


# =========================================================
# 4) Model: Transfer Learning (MobileNetV2)
#    - backbone frozen (simple baseline)
# =========================================================
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = keras.Sequential([
    keras.Input(shape=IMG_SIZE + (3,)),
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================================================
# 5) Train
# =========================================================
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model_path = os.path.join(MODEL_DIR, "xray_cnn_mobilenetv2.keras")
model.save(model_path)
print(f"\nSaved model to: {model_path}")


# =========================================================
# 6) Evaluation (report + confusion matrix)
# =========================================================
val_probs = model.predict(val_ds)
val_pred = np.argmax(val_probs, axis=1)

y_true = val_labels
y_pred = val_pred

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)

print("\nClassification report:\n", report)
print("\nConfusion matrix:\n", cm)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)


# =========================================================
# 7) Sensitivity (Recall) + Specificity per class (clinical-style)
#    Treat each class as "positive" vs "all others"
# =========================================================
sens_per_class = []
spec_per_class = []

for i in range(len(CLASSES)):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FN + FP)

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    sens_per_class.append(sensitivity)
    spec_per_class.append(specificity)

print("\nSensitivity (Recall) per class:")
for cls, v in zip(CLASSES, sens_per_class):
    print(f"  {cls}: {v:.4f}")

print("\nSpecificity per class:")
for cls, v in zip(CLASSES, spec_per_class):
    print(f"  {cls}: {v:.4f}")

with open(os.path.join(OUTPUT_DIR, "sens_spec_per_class.txt"), "w", encoding="utf-8") as f:
    f.write("Sensitivity (Recall) per class:\n")
    for cls, v in zip(CLASSES, sens_per_class):
        f.write(f"{cls}: {v:.4f}\n")
    f.write("\nSpecificity per class:\n")
    for cls, v in zip(CLASSES, spec_per_class):
        f.write(f"{cls}: {v:.4f}\n")


# =========================================================
# 8) (Optional) ROC-AUC (OvR) macro/weighted
# =========================================================
y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))

auc_macro = roc_auc_score(y_true_bin, val_probs, average="macro", multi_class="ovr")
auc_weighted = roc_auc_score(y_true_bin, val_probs, average="weighted", multi_class="ovr")

print(f"\nROC-AUC (OvR) macro:    {auc_macro:.4f}")
print(f"ROC-AUC (OvR) weighted: {auc_weighted:.4f}")

with open(os.path.join(OUTPUT_DIR, "roc_auc.txt"), "w", encoding="utf-8") as f:
    f.write(f"ROC-AUC (OvR) macro: {auc_macro:.4f}\n")
    f.write(f"ROC-AUC (OvR) weighted: {auc_weighted:.4f}\n")


# =========================================================
# 9) Plots
# =========================================================

# --- Confusion matrix plot ---
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Validation)")
plt.colorbar()
ticks = np.arange(len(CLASSES))
plt.xticks(ticks, CLASSES, rotation=45, ha="right")
plt.yticks(ticks, CLASSES)

thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
for r in range(cm.shape[0]):
    for c in range(cm.shape[1]):
        plt.text(c, r, str(cm[r, c]),
                 ha="center", va="center",
                 color="white" if cm[r, c] > thresh else "black")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

# --- Sensitivity + Specificity bar plot ---
x = np.arange(len(CLASSES))
width = 0.35

plt.figure()
plt.bar(x - width/2, sens_per_class, width, label="Sensitivity (Recall)")
plt.bar(x + width/2, spec_per_class, width, label="Specificity")
plt.xticks(x, CLASSES, rotation=20, ha="right")
plt.ylim(0, 1)
plt.title("Sensitivity and Specificity per class (Validation)")
plt.xlabel("Class")
plt.ylabel("Metric value (0–1)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sens_spec_per_class.png"), dpi=200)
plt.close()

# --- Loss curve ---
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=200)
plt.close()

# --- Accuracy curve ---
plt.figure()
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=200)
plt.close()

print("\nDone. Check /results for outputs.")
