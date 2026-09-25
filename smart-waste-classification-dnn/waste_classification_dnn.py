"""
=============================================================================
Smart Waste Classification Using Deep Neural Network (DNN)
=============================================================================
Scenario: College campus waste segregation automation.
Classes: Paper, Plastic, Metal, Organic (4 Classes)

This script is fully self-contained and optimized to run on Google Colab 
(Free Tier CPU or T4 GPU).

Fixes applied:
1. Label Synchronization: Explicitly enforced `class_names=CLASSES` in Keras data
   loaders to resolve the alphabetical folder sorting bug that scrambled class labels.
2. Architecture Overfitting: Replaced 16.7M parameter Flatten layer with 
   GlobalAveragePooling2D and L2 weight decay, dropping parameters from 17M to ~125K.
3. Feature Richness: Distinctive synthetic generation patterns for clean initial testing.
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
from tensorflow.keras import layers, models, callbacks, regularizers
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
    print("ℹ️ Running on CPU (Free tier Colab default). Code is lightweight and runs fast.")

# ---------------------------------------------------------------------------
# 2. Dataset Setup & Mock Generator (Zero-Friction Execution)
# ---------------------------------------------------------------------------
DATASET_DIR = "waste_dataset"
# Standardized class list (strictly maintained across all loaders)
CLASSES = ["paper", "plastic", "metal", "organic"]
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
BATCH_SIZE = 16

def create_synthetic_waste_dataset(base_dir=DATASET_DIR, samples_per_class=30):
    """
    Generates distinct synthetic images if phone photos are not yet uploaded.
    Each class has characteristic colors and geometry simulating waste items:
    - paper: Off-white notebook pages with ruled lines and text textures.
    - plastic: Bright cyan/blue bottles and translucent shapes.
    - metal: Metallic grey/silver cans with specular reflection bands.
    - organic: Earthy green leaves and organic brown food contours.
    """
    print(f"📦 Generating synthetic dataset in '{base_dir}' with {samples_per_class} images per class...")
    os.makedirs(base_dir, exist_ok=True)
    
    for cls in CLASSES:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            if cls == "paper":
                # Off-white paper with horizontal ruled lines
                bg_color = (random.randint(240, 255), random.randint(240, 250), random.randint(235, 245))
                img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=bg_color)
                draw = ImageDraw.Draw(img)
                # Draw lines like a notepad/paper sheet
                for y in range(20, IMG_HEIGHT - 15, 14):
                    draw.line([(15, y), (IMG_WIDTH - 15, y)], fill=(180, 200, 220), width=1)
                # Margin line
                draw.line([(30, 10), (30, IMG_HEIGHT - 10)], fill=(230, 150, 150), width=2)
                draw.text((35, 15), f"Paper {i+1}", fill=(40, 40, 60))
                
            elif cls == "plastic":
                # Dark background with translucent blue/cyan bottle/cylinder
                bg_color = (random.randint(220, 235), random.randint(230, 245), random.randint(240, 255))
                img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=bg_color)
                draw = ImageDraw.Draw(img)
                # Bottle body & neck
                x0, y0 = random.randint(35, 45), random.randint(35, 45)
                x1, y1 = random.randint(85, 95), random.randint(95, 110)
                draw.rectangle([x0, y0, x1, y1], fill=(0, 170, 230), outline=(0, 100, 180))
                draw.rectangle([x0 + 15, y0 - 15, x1 - 15, y0], fill=(0, 120, 200)) # neck/cap
                draw.text((20, 15), f"Plastic {i+1}", fill=(0, 50, 120))
                
            elif cls == "metal":
                # Metallic silver/grey can with specular highlights
                bg_color = (random.randint(210, 220), random.randint(210, 220), random.randint(210, 220))
                img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=bg_color)
                draw = ImageDraw.Draw(img)
                # Cylindrical metal can with metallic band
                draw.rectangle([35, 25, 95, 105], fill=(140, 145, 155), outline=(70, 75, 85))
                # Shiny reflection streak
                draw.line([(55, 25), (55, 105)], fill=(230, 235, 245), width=6)
                draw.text((20, 12), f"Metal {i+1}", fill=(30, 30, 35))
                
            elif cls == "organic":
                # Green leaf / earthy fruit leftover
                bg_color = (random.randint(235, 245), random.randint(230, 240), random.randint(215, 225))
                img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=bg_color)
                draw = ImageDraw.Draw(img)
                # Elliptical leaf / apple shape
                cx, cy = random.randint(55, 75), random.randint(55, 75)
                r = random.randint(25, 35)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(45, 140, 45), outline=(25, 90, 25))
                draw.line([(cx, cy - r), (cx + 5, cy - r - 12)], fill=(120, 70, 20), width=3) # stem
                draw.text((20, 15), f"Organic {i+1}", fill=(20, 80, 20))
                
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
# CRITICAL: We pass class_names=CLASSES explicitly so labels are NOT permuted
# by Keras's default alphabetical directory sorting!
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    class_names=CLASSES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    class_names=CLASSES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, "test"),
    class_names=CLASSES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Verify synchronization
assert train_ds.class_names == CLASSES, f"Mismatch: {train_ds.class_names} vs {CLASSES}"
print(f"✅ Label order verified across all splits: {train_ds.class_names}")

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
    - Data Augmentation Layer: Combat overfitting on small custom dataset.
    - Deep Feature Extraction:
      * Conv Block 1: 32 filters (3x3), ReLU, BatchNormalization, MaxPooling2D(2, 2)
      * Conv Block 2: 64 filters (3x3), ReLU, BatchNormalization, MaxPooling2D(2, 2)
      * Conv Block 3: 128 filters (3x3), ReLU, BatchNormalization, MaxPooling2D(2, 2)
    - Structural Regularization: GlobalAveragePooling2D()
      * Replaces the 16.7M parameter Flatten layer with spatial pooling.
      * Prevents severe overfitting on small datasets.
    - Deep Dense Classification Stack:
      * Dense (128 units) + ReLU + L2 Regularization + BatchNormalization + Dropout(0.3)
      * Dense (64 units) + ReLU + Dropout(0.2)
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
        
        # Deep Convolutional Feature Extraction Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="Conv_Block_1"),
        layers.BatchNormalization(name="BN_1"),
        layers.MaxPooling2D((2, 2), name="MaxPool_1"),
        
        # Deep Convolutional Feature Extraction Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="Conv_Block_2"),
        layers.BatchNormalization(name="BN_2"),
        layers.MaxPooling2D((2, 2), name="MaxPool_2"),
        
        # Deep Convolutional Feature Extraction Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="Conv_Block_3"),
        layers.BatchNormalization(name="BN_3"),
        layers.MaxPooling2D((2, 2), name="MaxPool_3"),
        
        # Global Average Pooling (eliminates 16.7M unneeded dense parameters!)
        layers.GlobalAveragePooling2D(name="Global_Avg_Pool"),
        
        # Deep Dense Hidden Layer 1 (128 units with L2 regularization)
        layers.Dense(
            128, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(1e-4),
            name="Dense_Hidden_1"
        ),
        layers.BatchNormalization(name="BN_Dense_1"),
        layers.Dropout(0.3, name="Dropout_1"),
        
        # Deep Dense Hidden Layer 2 (64 units)
        layers.Dense(64, activation='relu', name="Dense_Hidden_2"),
        layers.Dropout(0.2, name="Dropout_2"),
        
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
# Loss Function: Categorical Crossentropy for multi-class classification
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
    min_lr=1e-5,
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
axes[0].set_ylim([0.0, 1.05])
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
# Extract ground truth and model predictions strictly aligned with test_ds
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
print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=CLASSES, 
    yticklabels=CLASSES
)
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
