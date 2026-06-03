import os 
import pandas as pd
import numpy as np
import gc
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix, classification_report
from model import ann_classification


# %% 
# --- 1. CONFIGURATION ---
DATA_PATH = '/Users/claudiocaruso/Projects/Tesi Vibra AI/datasets/X_pca_dataset.parquet'

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
    
X_train = minmax_scale(X_train, axis=1)
X_val = minmax_scale(X_val, axis=1)
X_test = minmax_scale(X_test, axis=1)


# %%

# --- 7. TRAINING ---

def train_model(X_train, y_train, X_val, y_val, n_PC, model_size):
    """
    Build and train an ANN for binary Raman classification.

    Parameters
    ----------
    X_train : np.ndarray
        Training matrix, shape (samples, n_PC). Must already be sliced to n_PC columns.
    y_train : np.ndarray
        Training binary labels (0 = Healthy, 1 = Tumoral) as a column.
    X_val : np.ndarray
        Validation set, same column count as X_train.
    y_val : np.ndarray
        Validation binary labels, same form as y_train.
    n_PC : int
        Number of PCA components used as model input.
    model_size : str
        ANN architecture size. Must be in ['S', 'M', 'L'].

    Returns
    -------
    (model, history) : tuple
        model : tf.keras.Model
            The fitted Keras model.
        history : tf.keras.callbacks.History
            Per-epoch training metrics.
    """
    
    if n_PC != X_train.shape[1]:
        raise ValueError(f"n_PC={n_PC} does not match X_train shape {X_train.shape}")
    
    model = ann_classification(n_PC, model_size)
      
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
    
    return (model, history)


# %%

# --- 8. FINAL EVALUATION ---

def final_evaluation(model, X_test, y_test, n_PC, model_size, DIR_PATH):
    """
    Evaluate a trained model on the test set and persist its artifacts.

    Saves a normalized confusion matrix (PNG) and the classification report
    (TXT) into DIR_PATH, and returns a flat metrics dict for CSV collection.

    Parameters
    ----------
    model : tf.keras.Model
        The trained model to evaluate.
    X_test : np.ndarray
        Test features, same column count as the model input.
    y_test : np.ndarray
        Test binary labels (0 = Healthy, 1 = Tumoral).
    n_PC : int
        Number of PCA components used (for titles and the returned dict).
    model_size : str
        ANN architecture size 'S', 'M' or 'L' (for titles and the returned dict).
    DIR_PATH : str
        Folder where the confusion matrix and report are saved.

    Returns
    -------
    dict
        Flat metrics for this run: identifiers (n_pc, size), the compiled
        Keras metrics (loss, accuracy, sensitivity, auc) and the per-class
        precision/recall for the Tumoral class.
    """

    print("\n" + "="*40)
    print("FINAL TEST PERFORMANCE (UNSEEN DATA)")
    print("="*40)

    # 1. Standard evaluation using the compiled metrics (AUC, Precision, Recall)
    results = model.evaluate(X_test, y_test, verbose=0)
    eval_metrics = dict(zip(model.metrics_names, results))
    for name, value in eval_metrics.items():
        print(f"{name.upper()}: {value:.4f}")

    # --- CONFUSION MATRIX & REPORT ---
    # 2. Generate predictions (probabilities between 0 and 1)
    y_pred_probs = model.predict(X_test, verbose=0)

    # 3. Convert probabilities to binary classes (Threshold = 0.5)
    # If prob > 0.5 -> 1 (Tumoral), else -> 0 (Healthy)
    y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

    class_names = ['Healthy', 'Tumoral']

    # 4. Build Confusion Matrix
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

    # 5. Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.title(f'Confusion Matrix: Binary Raman Classification\n(ANN of size {model_size} on {n_PC} PCs)')
    plt.ylabel('True Label (Actual)')
    plt.xlabel('Predicted Label (Model)')
    plt.savefig(os.path.join(DIR_PATH, "CONFUSION_MATRIX.png"), dpi=300)
    plt.close()

    # 6. Detailed Report (printed, saved to file, and parsed for the CSV)
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                        output_dict=True)
    print("\n" + "="*40)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*40)
    print(report_text)

    with open(os.path.join(DIR_PATH, "classification_report.txt"), "w") as f:
        f.write(report_text)

    # 7. Flat metrics dict for all_results.csv
    return {
        'n_pc':              n_PC,
        'size':              model_size,
        **eval_metrics,
        'precision_tumoral': report_dict['Tumoral']['precision'],
        'recall_tumoral':    report_dict['Tumoral']['recall'],
    }

# %%

# --- 10. SAVE THE TRAINED MODEL ---

def save_model(model, DIR_PATH):
    print("\nSaving the model to the hard drive...")
    
    save_dir_params = os.path.join(DIR_PATH, "parameters")
    os.makedirs(save_dir_params, exist_ok=True) 
        
    model_path = os.path.join(save_dir_params, "raman_binary_ann.keras")
    model.save(model_path)
    
    weights_path = os.path.join(save_dir_params, "raman_binary_ann.weights.h5")
    model.save_weights(weights_path)
    
    print(f"Model and weights successfully saved at: {save_dir_params}")

# %%

# --- 8. LEARNING CURVES VISUALIZATION ---

def learning_curves_viz(history, DIR_PATH):
    """
    Plot and save the training learning curves (loss, accuracy, AUC).

    Parameters
    ----------
    history : tf.keras.callbacks.History
        History object returned by train_model, holding per-epoch metrics.
    DIR_PATH : str
        Folder where the 'metric_images/METRICS.png' figure is saved.

    Returns
    -------
    None
    """
    
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
    metrics_dir = os.path.join(DIR_PATH, "metric_images")
    os.makedirs(metrics_dir, exist_ok=True) 
    
    plt.savefig(os.path.join(metrics_dir, "METRICS.png"), dpi=300)
    plt.close()


# %% 

# --- 9. TRAINING ALL THE COMBINATIONS OF ANNs ---

DIR_PATH = '/Users/claudiocaruso/Projects/Tesi Vibra AI/'

PC_VARIANTS   = [3, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 483]
SIZE_VARIANTS = ['S', 'M', 'L']

records    = []
total_runs = len(PC_VARIANTS) * len(SIZE_VARIANTS)
run_count  = 0

for n_PC in PC_VARIANTS:
    # Slice the input matrices to the first n_PC components (once per n_PC)
    X_final_train = X_train[:, :n_PC]
    X_final_val   = X_val[:,   :n_PC]
    X_final_test  = X_test[:,  :n_PC]

    for model_size in SIZE_VARIANTS:
        run_count += 1
        print(f"\n{'#'*60}\n[{run_count}/{total_runs}] n_PC={n_PC} | size={model_size}\n{'#'*60}")

        # Each run gets its own nested folder: results/<size>/PC<n_PC>/
        run_dir = os.path.join(DIR_PATH, 'results', model_size, f'PC{n_PC}')
        os.makedirs(run_dir, exist_ok=True)

        # Free the previous model's graph from memory before building a new one
        tf.keras.backend.clear_session()

        model, history = train_model(X_final_train, y_train,
                                     X_final_val, y_val, n_PC, model_size)
        metrics = final_evaluation(model, X_final_test, y_test,
                                   n_PC, model_size, run_dir)
        learning_curves_viz(history, run_dir)
        save_model(model, run_dir)

        records.append(metrics)

# --- 10. SAVE THE AGGREGATED RESULTS TABLE ---
results_dir = os.path.join(DIR_PATH, 'results')
os.makedirs(results_dir, exist_ok=True)

df_results = pd.DataFrame(records)
df_results.to_csv(os.path.join(results_dir, 'all_ANN_results.csv'), index=False)

print(f"\nGrid search complete. {total_runs} runs saved under: {results_dir}")
print("\nTop 10 runs by recall on the Tumoral class:")
print(df_results.sort_values('recall_tumoral', ascending=False).head(10).to_string(index=False))


