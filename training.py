import pandas as pd
import numpy as np
import gc
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from model import ann_classification
import matplotlib.pyplot as plt


# %% 
# --- 1. CONFIGURATION ---
DATA_PATH = '/Users/claudiocaruso/Projects/Tesi Vibra AI/X_pca_dataset.parquet'

# Tumor-related labels from your mapping: Tumor(2), Necrosis (0)
TUMOR_LABELS = [2, 0, 20] 

# --- 2. DATA LOADING & FILTERING ---
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH, engine='pyarrow')
# df = df[df['Label'] != 20].copy() 

# --- 3. BINARY LABELING ---
# 1 = Tumor/Cancerous Tissue, 0 = Healthy/Normal Tissue
print("Mapping labels to Binary (0: Healthy, 1: Tumor)...")
df['Binary_Label'] = df['Label'].apply(lambda x: 1 if x in TUMOR_LABELS else 0)

# --- 4. CLASS BALANCING (DOWNSAMPLING) ---
# We balance classes to prevent the model from becoming biased towards the majority class
n_healthy = len(df[df['Binary_Label'] == 0])
n_tumor = len(df[df['Binary_Label'] == 1])
samples_per_class = min(n_healthy, n_tumor, 300000) 

print(f"Original Distribution - Healthy: {n_healthy}, Tumor: {n_tumor}")
print(f"Downsampling to {samples_per_class} samples per class...")

df_healthy = df[df['Binary_Label'] == 0].sample(samples_per_class, random_state=42)
df_tumor = df[df['Binary_Label'] == 1].sample(samples_per_class, random_state=42)
df_balanced = pd.concat([df_healthy, df_tumor]).sample(frac=1).reset_index(drop=True)

# Free original memory
del df, df_healthy, df_tumor; gc.collect()

# %%

# --- 6. DATA SPLITTING (70/15/15) ---
# Defining X as the spectra PCA matrix and y as the binary labels in arrays
pc_cols = [col for col in df_balanced.columns if col.startswith('PC')]
X = df_balanced[pc_cols].to_numpy()
y = df_balanced['Binary_Label'].to_numpy()

print("Splitting data into Train, Validation, and Test sets...")
# Stratify ensures the 50/50 ratio is preserved in all sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
# X_train = minmax_scale(X_train, axis=1)
# X_val = minmax_scale(X_val, axis=1)
# X_test = minmax_scale(X_test, axis=1)

spectral_length = len(pc_cols)


# %%

# --- 7. TRAINING ---
model = ann_classification(spectral_length,"M")

# Training Callbacks
callbacks = [
    # Stop training if validation AUC stops improving
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True),
    # Reduce learning rate when loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

print("\nStarting Binary Training Loop...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=512,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)
# %%

# --- 8. FINAL EVALUATION ---
print("\n" + "="*40)
print("FINAL TEST PERFORMANCE (UNSEEN DATA)")
print("="*40)

# 1. Standard evaluation using the compiled metrics (AUC, Precision, Recall)
results = model.evaluate(X_test, y_test, verbose=0)
for name, value in zip(model.metrics_names, results):
    print(f"{name.upper()}: {value:.4f}")

# --- CONFUSION MATRIX & REPORT ---
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 2. Generate predictions (probabilities between 0 and 1)
y_pred_probs = model.predict(X_test, verbose=0)

# 3. Convert probabilities to binary classes (Threshold = 0.5)
# If prob > 0.5 -> 1 (Tumoral), else -> 0 (Healthy)
y_pred = (y_pred_probs > 0.5).astype("int32")

class_names = ['Healthy', 'Tumoral']

# 4. Build Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 5. Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.title('Confusion Matrix: Binary Raman Classification\n(CNN 1st Derivative)')
plt.ylabel('True Label (Actual)')
plt.xlabel('Predicted Label (Model)')
plt.show()

# 6. Detailed Report
print("\n" + "="*40)
print("DETAILED CLASSIFICATION REPORT")
print("="*40)
print(classification_report(y_test, y_pred, target_names=class_names))

# %%
import os 

BASE_SAVE_DIR = '/Users/claudiocaruso/Projects/Tesi Vibra AI/'

# --- 10. SAVE THE TRAINED MODEL ---
print("\nSaving the model to the hard drive...")

save_dir_params = os.path.join(BASE_SAVE_DIR, "parameters")
os.makedirs(save_dir_params, exist_ok=True) 
    
model_path = os.path.join(save_dir_params, "raman_binary_cnn.keras")
model.save(model_path)

weights_path = os.path.join(save_dir_params, "raman_binary_cnn.weights.h5")
model.save_weights(weights_path)

print(f"Model and weights successfully saved at: {save_dir_params}")

# %%

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- 8. LEARNING CURVES VISUALIZATION ---
print("\nGenerating training plots...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss Plot
axes[0].plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e', linestyle='--', linewidth=2)
axes[0].set_title('Model Loss (Binary Crossentropy)', fontsize=14)
axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(loc='upper right')

# Accuracy Plot
axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linestyle='--', linewidth=2)
axes[1].set_title('Model Accuracy', fontsize=14)
axes[1].set_xlabel('Epochs', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].legend(loc='lower right')

# AUC Plot
axes[2].plot(history.history['auc'], label='Train AUC', color='#1f77b4', linewidth=2)
axes[2].plot(history.history['val_auc'], label='Validation AUC', color='#ff7f0e', linestyle='--', linewidth=2)
axes[2].set_title('Model AUC', fontsize=14)
axes[2].set_xlabel('Epochs', fontsize=12)
axes[2].set_ylabel('AUC', fontsize=12)
axes[2].legend(loc='lower right')

plt.tight_layout()
metrics_dir = os.path.join(BASE_SAVE_DIR, "metric_images")
os.makedirs(metrics_dir, exist_ok=True) 

plt.savefig(os.path.join(metrics_dir, "METRICS.png"), dpi=300)
plt.show()
# %%


# --- 9. FINAL EVALUATION & METRICS ---
print("\n" + "="*40)
print("FINAL TEST PERFORMANCE (UNSEEN DATA)")
print("="*40)

results = model.evaluate(X_test, y_test, verbose=0)
for name, value in zip(model.metrics_names, results):
    print(f"{name.upper()}: {value:.4f}")

# 1. Generate predictions (probabilities para 1 sola neurona)
y_pred_probs = model.predict(X_test, verbose=0)

# 2. Convertir probabilidades a clases binarias (Umbral 0.5)
y_pred_classes = (y_pred_probs > 0.5).astype("int32").flatten()
y_true_classes = y_test # En binario, y_test ya es 0 o 1

class_names = ['Non-tumor (0)', 'Damaged (1)']

# 3. Build NORMALIZED Confusion Matrix
cm_normalized = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')

# 4. Plotting the heatmap

plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.title('Normalized Confusion Matrix: Binary Classification')
plt.ylabel('True Label (Actual)')
plt.xlabel('Predicted Label (Model)')

plt.savefig(os.path.join(metrics_dir, "CONFUSSION_MATRIX.png"), dpi=300)
plt.show()

# 5. Detailed Report
print("\n" + "="*40)
print("DETAILED CLASSIFICATION REPORT")
print("="*40)
report_text = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
print(report_text)




