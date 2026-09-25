"""
=============================================================================
Smart Waste Classification Using Deep Neural Network (DNN)
=============================================================================
Scenario: College campus waste segregation automation.
Classes: Paper, Plastic, Metal, Organic (4 Classes)

This script is fully self-contained and optimized to run on Google Colab 
(Free Tier CPU/T4 GPU).

Instructions for Google Colab:
1. Copy and paste this entire script into a single Colab cell or notebook.
2. If you have your own photos:
   - Create a folder named 'waste_dataset' with subfolders: 'paper', 'plastic', 'metal', 'organic'.
   - Upload 20-30 photos into each subfolder.
3. If you run without uploading, a synthetic demonstration dataset will be 
   generated automatically so you can test the entire pipeline end-to-end!
=============================================================================
"""

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

# Deep Learning Framework (TensorFlow / Keras - Pre-installed in Google Colab)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# 1. Reproducibility Configuration
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Check execution device
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"✅ GPU Detected: {physical_devices[0]}")
else:
    print("ℹ️ Running on CPU (Free tier Colab default). Code is lightweight and runs smoothly.")

# ---------------------------------------------------------------------------
# 2. Dataset Setup & Mock Generator (Zero-Friction Execution)
# ---------------------------------------------------------------------------
DATASET_DIR = "waste_dataset"
CLASSES = ["paper", "plastic", "metal", "organic"]
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
BATCH_SIZE = 16

def create_synthetic_waste_dataset(base_dir=DATASET_DIR, samples_per_class=30):
    """
    Generates realistic dummy images if user hasn't uploaded their phone photos yet.
    Ensures zero runtime errors during initial Colab trial run.
    """
    print(f"📦 Generating synthetic dataset in '{base_dir}' with {samples_per_class} images per class...")
    os.makedirs(base_dir, exist_ok=True)
    
    color_schemes = {
        "paper": ((240, 240, 240), (200, 200, 210)),      # Off-white / light grey
        "plastic": ((60, 140, 220), (100, 200, 240)),     # Translucent blues
        "metal": ((150, 150, 160), (90, 90, 100)),        # Metallic greys
        "organic": ((70, 160, 70), (160, 100, 40))        # Green / earthy brown
    }
    
    for cls in CLASSES:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        c1, c2 = color_schemes[cls]
        
        for i in range(samples_per_class):
            img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=c1)
            draw = ImageDraw.Draw(img)
            # Add geometric patterns to simulate object shapes and lighting variations
            offset = random.randint(10, 40)
            draw.rectangle(
                [offset, offset, IMG_WIDTH - offset, IMG_HEIGHT - offset], 
                fill=c2, 
                outline=(50, 50, 50)
            )
            draw.text((15, 15), f"{cls}_{i+1}", fill=(0, 0, 0))
            img.save(os.path.join(cls_dir, f"{cls}_{i+1:03d}.jpg"))
            
    print(f"✅ Generated {samples_per_class * len(CLASSES)} synthetic images across {len(CLASSES)} classes.")

# Verify whether user uploaded their phone photos or if synthetic data is needed
def check_or_prepare_dataset(base_dir=DATASET_DIR):
    need_mock = False
    if not os.path.exists(base_dir):
        need_mock = True
    else:
        for cls in CLASSES:
            cls_path = os.path.join(base_dir, cls)
            if not os.path.exists(cls_path) or len(os.listdir(cls_path)) < 5:
                need_mock = True
                break
                
    if need_mock:
        print("💡 No phone photos detected in 'waste_dataset'. Generating sample dataset for verification.")
        create_synthetic_waste_dataset(base_dir, samples_per_class=30)
    else:
        print("📸 Custom phone dataset detected in 'waste_dataset'!")

check_or_prepare_dataset(DATASET_DIR)

# ---------------------------------------------------------------------------
# 3. Train, Validation & Test Split (70% Train, 15% Val, 15% Test)
# ---------------------------------------------------------------------------
SPLIT_DIR = "split_dataset"

def create_dataset_splits(source_dir=DATASET_DIR, dest_dir=SPLIT_DIR, train_ratio=0.70, val_ratio=0.15):
    """
    Splits the collected dataset into:
    - 70% Training
    - 15% Validation
    - 15% Testing
    """
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
        
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
            
    for cls in CLASSES:
        cls_src = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(cls_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)
        
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        for f in train_files:
            shutil.copy(os.path.join(cls_src, f), os.path.join(dest_dir, "train", cls, f))
        for f in val_files:
            shutil.copy(os.path.join(cls_src, f), os.path.join(dest_dir, "val", cls, f))
        for f in test_files:
            shutil.copy(os.path.join(cls_src, f), os.path.join(dest_dir, "test", cls, f))
            
    print(f"📊 Dataset split complete:")
    for split in ["train", "val", "test"]:
        counts = [len(os.listdir(os.path.join(dest_dir, split, c))) for c in CLASSES]
        print(f"   - {split.upper():5s} set: {sum(counts)} images {dict(zip(CLASSES, counts))}")

create_dataset_splits(DATASET_DIR, SPLIT_DIR)

# ---------------------------------------------------------------------------
# 4. Data Loading & Augmentation Pipeline
# ---------------------------------------------------------------------------
# Using tf.keras.utils.image_dataset_from_directory for efficient pipeline
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "test"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Autotune for smooth I/O performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------------------------------------------------------------
# 5. Deep Neural Network (DNN) Architecture Design
# ---------------------------------------------------------------------------
def build_waste_classification_dnn():
    """
    Designs the Deep Neural Network:
    - Input Layer: Accepts (128, 128, 3) normalized RGB images.
    - Data Augmentation Layer: Combat overfitting on 20-30 images/class.
    - Deep Feature Extraction & Dense Layers:
      * Conv2D & Pooling layers extract hierarchical visual features.
      * Dense (512 units) + ReLU + BatchNormalization + Dropout(0.4)
      * Dense (256 units) + ReLU + BatchNormalization + Dropout(0.3)
      * Dense (128 units) + ReLU + Dropout(0.2)
    - Output Layer: 4 neurons with Softmax activation (Paper, Plastic, Metal, Organic).
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="Input_Layer"),
        
        # Preprocessing & Real-Time Data Augmentation
        layers.Rescaling(1.0 / 255.0, name="Normalization_Layer"),
        layers.RandomFlip("horizontal_and_vertical", name="Augmentation_Flip"),
        layers.RandomRotation(0.15, name="Augmentation_Rotation"),
        layers.RandomZoom(0.15, name="Augmentation_Zoom"),
        layers.RandomContrast(0.1, name="Augmentation_Contrast"),
        
        # Deep Convolutional Feature Extraction Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="Conv_Block_1"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="Conv_Block_2"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="Conv_Block_3"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flattening to feed into Deep Dense Layers
        layers.Flatten(name="Flatten_Layer"),
        
        # Deep Dense Hidden Layer 1 (512 units)
        layers.Dense(512, activation='relu', name="Dense_Hidden_1"),
        layers.BatchNormalization(),
        layers.Dropout(0.4, name="Dropout_1"),
        
        # Deep Dense Hidden Layer 2 (256 units)
        layers.Dense(256, activation='relu', name="Dense_Hidden_2"),
        layers.BatchNormalization(),
        layers.Dropout(0.3, name="Dropout_2"),
        
        # Deep Dense Hidden Layer 3 (128 units)
        layers.Dense(128, activation='relu', name="Dense_Hidden_3"),
        layers.Dropout(0.2, name="Dropout_3"),
        
        # Output Layer: 4 Classes with Softmax
        layers.Dense(len(CLASSES), activation='softmax', name="Output_Layer_Softmax")
    ], name="Smart_Waste_Classification_DNN")
    
    return model

model = build_waste_classification_dnn()
model.summary()

# ---------------------------------------------------------------------------
# 6. Training Configuration (Optimizer, Loss, Callbacks)
# ---------------------------------------------------------------------------
# Optimizer: Adam with adaptive learning rate
# Loss Function: Categorical Crossentropy for one-hot multi-class outputs
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Training Callbacks: EarlyStopping and Learning Rate Scheduling
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

EPOCHS = 35
print("\n🚀 Starting Deep Neural Network Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ---------------------------------------------------------------------------
# 7. Model Evaluation & Performance Curves
# ---------------------------------------------------------------------------
print("\n📈 Evaluating on Independent Test Dataset (15%)...")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n🎯 Test Set Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

# Plot Training & Validation Loss & Accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, linestyle='--')
axes[0].set_title('Model Accuracy vs. Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss Plot
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2, color='crimson')
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--', color='darkred')
axes[1].set_title('Model Loss vs. Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (Categorical Cross-Entropy)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 8. Confusion Matrix & Classification Report
# ---------------------------------------------------------------------------
# Gather predictions on the test dataset
y_true = []
y_pred_probs = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred_probs.extend(preds)

y_true = np.array(y_true)
y_pred = np.argmax(np.array(y_pred_probs), axis=1)

# Classification Report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Test Set Confusion Matrix - Waste Classification')
plt.xlabel('Predicted Category')
plt.ylabel('Ground Truth Category')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 9. Real-Time Inference Demo on Sample Test Items
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
sample_images, sample_labels = next(iter(test_ds))
sample_preds = model.predict(sample_images, verbose=0)

num_display = min(8, len(sample_images))
for i in range(num_display):
    ax = plt.subplot(2, 4, i + 1)
    img_display = sample_images[i].numpy().astype("uint8")
    plt.imshow(img_display)
    
    true_label = CLASSES[np.argmax(sample_labels[i])]
    pred_label = CLASSES[np.argmax(sample_preds[i])]
    confidence = np.max(sample_preds[i]) * 100
    
    color = "green" if true_label == pred_label else "red"
    plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", color=color, fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Save trained model weights
model.save("waste_classification_dnn_model.keras")
print("\n💾 Model successfully exported to 'waste_classification_dnn_model.keras'!")
