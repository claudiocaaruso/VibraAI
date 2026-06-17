import gc
import json
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.ndimage import generic_filter
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


# Make ANN1/model.py importable while keeping this script in the VIBRA folder.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ANN_DIR = os.path.join(ROOT_DIR, "ANN1")
if ANN_DIR not in sys.path:
    sys.path.insert(0, ANN_DIR)

from model import ann_classification


# =============================================================================
# CONFIG
# =============================================================================

# Raw dataset: this script starts from Raman bands, not from a precomputed PCA file.
DATA_PATH = os.path.join(ROOT_DIR, "PCA", "spectral_dataset_clean.parquet")

# All outputs from this script go here.
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, "fold_norm_pca_ann")

# Binary experiment definition.
# Examples:
#   tumor vs tumor stroma:
#       CLASS_1_LABELS = [0, 2, 20]
#       CLASS_0_LABELS = [3]
#
#   tumor vs all other labels:
#       CLASS_1_LABELS = [0, 2, 20]
#       CLASS_0_LABELS = "all_other"
#
# Class 1 is the positive class used for sensitivity/recall.
CLASS_1_NAME = "class_1"
CLASS_0_NAME = "class_0"
CLASS_1_LABELS = [2]
CLASS_0_LABELS = [6]

# Normalization choices:
#   "none"                   -> no normalization
#   "filtered_labels"        -> per-map normalization using only selected labels in each split
#   "full_map"               -> per-map normalization using all labels from each map in each split
#   "spectrum"               -> normalize each spectrum independently using its own bands
#   "global_full"            -> train-fitted feature normalization using all labels from train samples
#   "global_filtered_labels" -> train-fitted feature normalization using only selected train rows
MAP_NORMALIZATION_MODE = "spectrum"
MAP_GROUP_COLUMNS = ["Sample_ID", "Map_ID"]
MAP_STD_EPS = 1e-8

# Binary balancing:
# if True, only the training partition of each fold is downsampled.
# Validation and test always retain all their original rows.
BALANCE_CLASSES = True

# Cross-validation.
N_FOLDS = 5
GROUP_COLUMN = "Sample_ID"
VALIDATION_SIZE_WITHIN_TRAIN = 0.1875

# PCA is fitted inside each fold using only the training rows.
N_COMPONENTS = 75
PCA_RANDOM_STATE = 42

# ANN.
MODEL_SIZE = "S"
EPOCHS = 200
BATCH_SIZE = 4096
CLASSIFICATION_THRESHOLD = 0.5

# Optional post-processing of predicted probabilities.
# The model is unchanged: smoothing is applied after prediction and before
# thresholding, metrics and prediction-map generation.
SMOOTH_PREDICTION_PROBS = True
PREDICTION_SMOOTHING_METHOD = "mean"  # "mean" or "median"
PREDICTION_SMOOTHING_KERNEL_SIZE = 3

# Reproducibility.
RANDOM_STATE = 44

# Optional heavier artifacts.
SAVE_PCA_ARTIFACTS = False
SAVE_PREDICTION_MAPS = True

# Sample-level ranking plot.
SAMPLE_RANKING_PRIMARY_METRIC = "class_1_f1"
SAMPLE_RANKING_SECONDARY_METRIC = "class_1_recall"

# Plot behavior. Keep True if running in Spyder and you want plots in the Plots pane.
SHOW_PLOTS = True


# =============================================================================
# SPLIT, DATA AND PREPROCESSING HELPERS
# =============================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_json_safe(value):
    """Convert NumPy/Pandas scalar values to plain Python objects for JSON."""
    if isinstance(value, dict):
        return {str(make_json_safe(key)): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def best_group_split(X, y, groups, test_size, random_state, n_candidates=200):
    """Return a group split with size and class balance close to target."""
    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=test_size,
        random_state=random_state,
    )
    global_positive_rate = y.mean()
    target_test_count = len(y) * test_size
    best_score = None
    best_indices = None

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        train_rate = y[train_idx].mean()
        test_rate = y[test_idx].mean()
        test_size_error = abs(len(test_idx) - target_test_count) / len(y)
        balance_error = abs(train_rate - global_positive_rate) + abs(test_rate - global_positive_rate)
        score = test_size_error + balance_error

        if best_score is None or score < best_score:
            best_score = score
            best_indices = (train_idx, test_idx)

    return best_indices


def get_band_columns(df):
    band_cols = [col for col in df.columns if col.startswith("band_")]
    if not band_cols:
        raise ValueError("No spectral band columns found. Expected columns named like band_1, band_2, ...")
    return band_cols


def filter_labels(df):
    """Keep only selected labels and create Binary_Label."""
    class_1_labels = list(CLASS_1_LABELS)

    if CLASS_0_LABELS == "all_other":
        df_work = df.copy()
        df_work["Binary_Label"] = df_work["Label"].apply(lambda x: 1 if x in class_1_labels else 0)
        class_0_labels = sorted([label for label in df_work["Label"].unique() if label not in class_1_labels])
    else:
        class_0_labels = list(CLASS_0_LABELS)
        keep_labels = class_0_labels + class_1_labels
        df_work = df[df["Label"].isin(keep_labels)].copy()
        df_work["Binary_Label"] = df_work["Label"].apply(lambda x: 1 if x in class_1_labels else 0)

    if df_work.empty:
        raise ValueError("No rows left after label filtering. Check CLASS_0_LABELS and CLASS_1_LABELS.")

    return df_work, class_0_labels, class_1_labels


def normalize_by_map(target_df, stats_context_df, feature_cols):
    """
    Normalize target rows using mean/std computed inside each map.

    A map is identified by MAP_GROUP_COLUMNS. The default is Sample_ID + Map_ID,
    so maps with the same Map_ID but different samples are not mixed.
    """
    missing = [col for col in MAP_GROUP_COLUMNS if col not in target_df.columns]
    if missing:
        raise ValueError(f"Missing map group columns in target_df: {missing}")

    missing = [col for col in MAP_GROUP_COLUMNS if col not in stats_context_df.columns]
    if missing:
        raise ValueError(f"Missing map group columns in stats_context_df: {missing}")

    target = target_df.copy().reset_index(drop=True)
    target["_row_order"] = np.arange(len(target))
    stats_context = stats_context_df[MAP_GROUP_COLUMNS + feature_cols].copy()

    means = (
        stats_context.groupby(MAP_GROUP_COLUMNS, sort=False)[feature_cols]
        .mean()
        .add_suffix("__map_mean")
        .reset_index()
    )
    stds = (
        stats_context.groupby(MAP_GROUP_COLUMNS, sort=False)[feature_cols]
        .std()
        .replace(0, 1)
        .fillna(1)
        .add_suffix("__map_std")
        .reset_index()
    )
    stats = means.merge(stds, on=MAP_GROUP_COLUMNS, how="inner")
    target = target.merge(stats, on=MAP_GROUP_COLUMNS, how="left")
    target = target.sort_values("_row_order").reset_index(drop=True)

    mean_cols = [f"{col}__map_mean" for col in feature_cols]
    std_cols = [f"{col}__map_std" for col in feature_cols]
    if target[mean_cols].isna().any().any():
        raise ValueError("Some target maps were not found in the normalization context.")

    values = target[feature_cols].to_numpy(dtype=np.float32)
    mean_values = target[mean_cols].to_numpy(dtype=np.float32)
    std_values = target[std_cols].to_numpy(dtype=np.float32)
    std_values = np.where(np.abs(std_values) < MAP_STD_EPS, 1.0, std_values)

    target.loc[:, feature_cols] = (values - mean_values) / std_values
    return target[target_df.columns].reset_index(drop=True)


def normalize_by_spectrum(target_df, feature_cols):
    """Normalize each spectrum independently using mean/std across its bands."""
    target = target_df.copy().reset_index(drop=True)
    values = target[feature_cols].to_numpy(dtype=np.float32)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    stds = np.where(np.abs(stds) < MAP_STD_EPS, 1.0, stds)
    target.loc[:, feature_cols] = (values - means) / stds
    return target


def fit_global_normalization(stats_context_df, feature_cols):
    """Fit feature-wise mean/std on a training-only context."""
    means = stats_context_df[feature_cols].mean()
    stds = stats_context_df[feature_cols].std().replace(0, 1).fillna(1)
    stds = stds.mask(stds.abs() < MAP_STD_EPS, 1.0)
    return means, stds


def apply_global_normalization(target_df, feature_cols, means, stds):
    """Apply train-fitted feature-wise mean/std to target rows."""
    target = target_df.copy().reset_index(drop=True)
    target.loc[:, feature_cols] = (
        target[feature_cols].astype(np.float32) - means
    ) / stds
    return target


def balance_binary_classes(df):
    """Downsample both classes to the size of the smaller class."""
    class_counts_before = df["Binary_Label"].value_counts().sort_index().to_dict()
    n_class_0 = class_counts_before.get(0, 0)
    n_class_1 = class_counts_before.get(1, 0)

    if n_class_0 == 0 or n_class_1 == 0:
        raise ValueError(f"Both binary classes must be present. Counts: {class_counts_before}")

    if not BALANCE_CLASSES:
        df_balanced = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        return df_balanced, class_counts_before, class_counts_before

    samples_per_class = min(n_class_0, n_class_1)
    df_0 = df[df["Binary_Label"] == 0].sample(n=samples_per_class, random_state=RANDOM_STATE)
    df_1 = df[df["Binary_Label"] == 1].sample(n=samples_per_class, random_state=RANDOM_STATE)
    df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    class_counts_after = df_balanced["Binary_Label"].value_counts().sort_index().to_dict()

    return df_balanced, class_counts_before, class_counts_after


def label_distribution_row(fold, split_name, y_values, groups):
    return {
        "fold": fold,
        "split": split_name,
        "rows": len(y_values),
        "class_0_rows": int((y_values == 0).sum()),
        "class_1_rows": int((y_values == 1).sum()),
        "class_1_rate": float(y_values.mean()) if len(y_values) else np.nan,
        "sample_id_count": int(pd.Series(groups).nunique()),
        "sample_ids": ", ".join(sorted(pd.Series(groups).astype(str).unique())),
    }


def smooth_prediction_probabilities(df_rows, y_probs):
    """Smooth predicted class-1 probabilities inside each Raman map."""
    if not SMOOTH_PREDICTION_PROBS:
        return np.asarray(y_probs).reshape(-1)

    needed_cols = MAP_GROUP_COLUMNS + ["x", "y"]
    missing = [col for col in needed_cols if col not in df_rows.columns]
    if missing:
        print(f"Skipping prediction smoothing: missing columns {missing}.")
        return np.asarray(y_probs).reshape(-1)

    if PREDICTION_SMOOTHING_KERNEL_SIZE < 2:
        return np.asarray(y_probs).reshape(-1)

    if PREDICTION_SMOOTHING_METHOD == "mean":
        smooth_func = np.nanmean
    elif PREDICTION_SMOOTHING_METHOD == "median":
        smooth_func = np.nanmedian
    else:
        raise ValueError("PREDICTION_SMOOTHING_METHOD must be one of: 'mean', 'median'.")

    work = df_rows[needed_cols].copy().reset_index(drop=True)
    work["_row_order"] = np.arange(len(work))
    work["_pred_prob"] = np.asarray(y_probs).reshape(-1)
    smoothed_parts = []

    for _, map_df in work.groupby(MAP_GROUP_COLUMNS, sort=False):
        grid = (
            map_df.pivot_table(index="y", columns="x", values="_pred_prob", aggfunc="mean")
            .sort_index()
            .sort_index(axis=1)
        )
        smoothed_grid = generic_filter(
            grid.to_numpy(dtype=np.float32),
            function=smooth_func,
            size=PREDICTION_SMOOTHING_KERNEL_SIZE,
            mode="nearest",
        )
        smoothed_frame = (
            pd.DataFrame(smoothed_grid, index=grid.index, columns=grid.columns)
            .stack(dropna=False)
            .rename("_smoothed_prob")
            .reset_index()
        )
        map_smoothed = map_df.merge(smoothed_frame, on=["y", "x"], how="left")
        smoothed_parts.append(map_smoothed[["_row_order", "_pred_prob", "_smoothed_prob"]])

    smoothed = pd.concat(smoothed_parts, ignore_index=True).sort_values("_row_order")
    return smoothed["_smoothed_prob"].fillna(smoothed["_pred_prob"]).to_numpy(dtype=np.float32)


# =============================================================================
# METRICS AND PLOTTING
# =============================================================================


def history_to_frame(history, fold_number):
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    frame.insert(0, "fold", fold_number)
    return frame


def summarize_histories(history_frames):
    history_all = pd.concat(history_frames, ignore_index=True)
    metric_cols = [col for col in history_all.columns if col not in ["fold", "epoch"]]
    grouped = history_all.groupby("epoch")[metric_cols]
    return history_all, grouped.mean().reset_index(), grouped.std().reset_index()


def plot_history(history_frame, metrics_dir, filename, title_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plots = [
        ("loss", "val_loss", "Model Loss (Binary Crossentropy)", "Loss"),
        ("accuracy", "val_accuracy", "Model Accuracy", "Accuracy"),
        ("auc", "val_auc", "Model AUC", "AUC"),
    ]

    for ax, (train_col, val_col, title, ylabel) in zip(axes, plots):
        if train_col in history_frame:
            ax.plot(history_frame["epoch"], history_frame[train_col], label="Train", linewidth=2)
        if val_col in history_frame:
            ax.plot(history_frame["epoch"], history_frame[val_col], label="Validation", linestyle="--", linewidth=2)
        ax.set_title(f"{title_prefix} - {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(metrics_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_mean_history(mean_history, std_history, metrics_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plots = [
        ("loss", "val_loss", "Model Loss (Binary Crossentropy)", "Loss"),
        ("accuracy", "val_accuracy", "Model Accuracy", "Accuracy"),
        ("auc", "val_auc", "Model AUC", "AUC"),
    ]

    epochs = mean_history["epoch"].values
    for ax, (train_col, val_col, title, ylabel) in zip(axes, plots):
        if train_col in mean_history:
            train_mean = mean_history[train_col].values
            train_std = std_history[train_col].fillna(0).values
            ax.plot(epochs, train_mean, label="Train Mean", color="#1f77b4", linewidth=2)
            ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, color="#1f77b4", alpha=0.15)
        if val_col in mean_history:
            val_mean = mean_history[val_col].values
            val_std = std_history[val_col].fillna(0).values
            ax.plot(epochs, val_mean, label="Validation Mean", color="#ff7f0e", linestyle="--", linewidth=2)
            ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, color="#ff7f0e", alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(metrics_dir, "METRICS_MEAN_5_FOLD.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrices(cm_total, class_names, metrics_dir):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Total Confusion Matrix Across Folds")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "CONFUSION_MATRIX_TOTAL_5_FOLD.png"), dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    row_sums = cm_total.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm_total, row_sums, out=np.zeros_like(cm_total, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix Across Folds")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "CONFUSION_MATRIX_NORMALIZED_5_FOLD.png"), dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    return cm_normalized


def compute_sample_results(fold_number, y_test, y_probs, y_pred, groups_test, class_names):
    rows = []
    y_probs = np.asarray(y_probs).reshape(-1)

    for sample_id in sorted(pd.Series(groups_test).astype(str).unique()):
        mask = pd.Series(groups_test).astype(str).to_numpy() == sample_id
        y_s = y_test[mask]
        p_s = y_probs[mask]
        pred_s = y_pred[mask]

        cm = confusion_matrix(y_s, pred_s, labels=[0, 1])
        report = classification_report(
            y_s,
            pred_s,
            labels=[0, 1],
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        if len(np.unique(y_s)) == 2:
            auc = roc_auc_score(y_s, p_s)
        else:
            auc = np.nan

        rows.append(
            {
                "fold": fold_number,
                "sample_id": sample_id,
                "rows": len(y_s),
                "class_0_rows": int((y_s == 0).sum()),
                "class_1_rows": int((y_s == 1).sum()),
                "class_1_rate": float(y_s.mean()) if len(y_s) else np.nan,
                "accuracy": float((pred_s == y_s).mean()) if len(y_s) else np.nan,
                "auc": auc,
                "class_0_precision": report[class_names[0]]["precision"],
                "class_0_recall": report[class_names[0]]["recall"],
                "class_0_f1": report[class_names[0]]["f1-score"],
                "class_1_precision": report[class_names[1]]["precision"],
                "class_1_recall": report[class_names[1]]["recall"],
                "class_1_f1": report[class_names[1]]["f1-score"],
                "confusion_tn": int(cm[0, 0]),
                "confusion_fp": int(cm[0, 1]),
                "confusion_fn": int(cm[1, 0]),
                "confusion_tp": int(cm[1, 1]),
            }
        )

    return rows


def plot_sample_difficulty_ranking(sample_results_df, metrics_dir, tables_dir):
    """Plot samples from hardest to easiest using class-1 performance metrics."""
    required_cols = [
        "sample_id",
        "rows",
        SAMPLE_RANKING_PRIMARY_METRIC,
        SAMPLE_RANKING_SECONDARY_METRIC,
        "confusion_fp",
        "confusion_fn",
    ]
    missing = [col for col in required_cols if col not in sample_results_df.columns]
    if missing:
        print(f"Skipping sample ranking plot: missing columns {missing}.")
        return None

    ranking_df = (
        sample_results_df.groupby("sample_id", as_index=False)
        .agg(
            rows=("rows", "sum"),
            class_1_rows=("class_1_rows", "sum"),
            class_1_f1=("class_1_f1", "mean"),
            class_1_recall=("class_1_recall", "mean"),
            accuracy=("accuracy", "mean"),
            auc=("auc", "mean"),
            confusion_fp=("confusion_fp", "sum"),
            confusion_fn=("confusion_fn", "sum"),
            confusion_tp=("confusion_tp", "sum"),
            confusion_tn=("confusion_tn", "sum"),
        )
        .sort_values([SAMPLE_RANKING_PRIMARY_METRIC, SAMPLE_RANKING_SECONDARY_METRIC], ascending=True)
        .reset_index(drop=True)
    )
    ranking_df.insert(0, "difficulty_rank", np.arange(1, len(ranking_df) + 1))
    ranking_df.to_csv(os.path.join(tables_dir, "sample_difficulty_ranking.csv"), index=False)

    plot_df = ranking_df.copy()
    plot_df["sample_id"] = plot_df["sample_id"].astype(str)

    n_samples = len(plot_df)
    fig_height = max(6, min(24, 0.42 * n_samples + 2))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_pos = np.arange(n_samples)
    bar_height = 0.38
    ax.barh(
        y_pos - bar_height / 2,
        plot_df[SAMPLE_RANKING_PRIMARY_METRIC],
        height=bar_height,
        label=SAMPLE_RANKING_PRIMARY_METRIC,
        color="#2a9d8f",
    )
    ax.barh(
        y_pos + bar_height / 2,
        plot_df[SAMPLE_RANKING_SECONDARY_METRIC],
        height=bar_height,
        label=SAMPLE_RANKING_SECONDARY_METRIC,
        color="#e76f51",
    )

    for idx, row in plot_df.iterrows():
        error_text = f"FP {int(row['confusion_fp'])} | FN {int(row['confusion_fn'])}"
        ax.text(
            1.01,
            idx,
            error_text,
            va="center",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["sample_id"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Metric value")
    ax.set_ylabel("Sample_ID")
    ax.set_title(
        "Sample Difficulty Ranking "
        f"(hardest to easiest by {SAMPLE_RANKING_PRIMARY_METRIC})"
    )
    ax.grid(True, axis="x", alpha=0.35)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(metrics_dir, "SAMPLE_DIFFICULTY_RANKING.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return ranking_df


def save_prediction_maps_if_enabled(df_test, y_probs, y_pred, fold_number, maps_dir):
    if not SAVE_PREDICTION_MAPS:
        return

    needed_cols = ["Sample_ID", "Map_ID", "x", "y", "Binary_Label"]
    if any(col not in df_test.columns for col in needed_cols):
        print("Skipping prediction maps: missing one of Sample_ID, Map_ID, x, y, Binary_Label.")
        return

    os.makedirs(maps_dir, exist_ok=True)
    plot_df = df_test[needed_cols].copy()
    plot_df["pred_prob"] = np.asarray(y_probs).reshape(-1)
    plot_df["pred_class"] = y_pred
    plot_df["error"] = (plot_df["pred_class"] != plot_df["Binary_Label"]).astype(int)

    for (sample_id, map_id), map_df in plot_df.groupby(["Sample_ID", "Map_ID"], sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        panels = [
            ("Binary_Label", "True label", "viridis"),
            ("pred_prob", "Predicted probability", "magma"),
            ("error", "Error", "Reds"),
        ]

        for ax, (value_col, title, cmap) in zip(axes, panels):
            heatmap = map_df.pivot_table(index="y", columns="x", values=value_col)
            im = ax.imshow(heatmap, origin="lower", cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f"Fold {fold_number} - Sample {sample_id} - Map {map_id}")
        plt.tight_layout()
        safe_sample = str(sample_id).replace("/", "_")
        safe_map = str(map_id).replace("/", "_")
        out_path = os.path.join(maps_dir, f"fold_{fold_number}_sample_{safe_sample}_map_{safe_map}.png")
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main():
    set_seed(RANDOM_STATE)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"run_{run_timestamp}_"
        f"{MODEL_SIZE}_pca{N_COMPONENTS}_"
        f"mapnorm-{MAP_NORMALIZATION_MODE}_"
        f"{N_FOLDS}fold"
    )
    base_save_dir = os.path.join(OUTPUT_BASE_DIR, run_name)
    metrics_dir = os.path.join(base_save_dir, "metric_images")
    tables_dir = os.path.join(base_save_dir, "tables")
    params_dir = os.path.join(base_save_dir, "parameters")
    maps_dir = os.path.join(base_save_dir, "prediction_maps")

    for directory in [metrics_dir, tables_dir, params_dir]:
        os.makedirs(directory, exist_ok=True)

    print("Loading raw spectral dataset...")
    df_raw = pd.read_parquet(DATA_PATH)
    band_cols = get_band_columns(df_raw)

    required_cols = ["Label", GROUP_COLUMN] + MAP_GROUP_COLUMNS
    missing_required_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_required_cols:
        raise ValueError(f"Missing required columns: {missing_required_cols}")

    print(f"Raw dataset shape: {df_raw.shape}")
    print(f"Spectral bands: {len(band_cols)}")
    print(f"Map normalization mode: {MAP_NORMALIZATION_MODE}")

    df_filtered, class_0_labels, class_1_labels = filter_labels(df_raw)
    class_names = [CLASS_0_NAME, CLASS_1_NAME]

    filtered_class_counts = df_filtered["Binary_Label"].value_counts().sort_index().to_dict()

    print("Binary label setup:")
    print(f"  {CLASS_0_NAME} labels: {class_0_labels}")
    print(f"  {CLASS_1_NAME} labels: {class_1_labels}")
    print(f"Filtered class counts: {filtered_class_counts}")
    print("Class balancing: training partition only")
    print(f"Filtered dataset shape: {df_filtered.shape}")
    print(f"Sample_ID groups: {df_filtered[GROUP_COLUMN].nunique()}")

    if df_filtered[GROUP_COLUMN].nunique() < N_FOLDS:
        raise ValueError(
            f"N_FOLDS={N_FOLDS}, but only {df_filtered[GROUP_COLUMN].nunique()} unique groups are available."
        )

    config = {
        "DATA_PATH": DATA_PATH,
        "OUTPUT_BASE_DIR": OUTPUT_BASE_DIR,
        "CLASS_0_NAME": CLASS_0_NAME,
        "CLASS_1_NAME": CLASS_1_NAME,
        "CLASS_0_LABELS": class_0_labels,
        "CLASS_1_LABELS": class_1_labels,
        "CLASS_0_LABELS_CONFIG": CLASS_0_LABELS,
        "CLASS_1_LABELS_CONFIG": CLASS_1_LABELS,
        "MAP_NORMALIZATION_MODE": MAP_NORMALIZATION_MODE,
        "MAP_GROUP_COLUMNS": MAP_GROUP_COLUMNS,
        "BALANCE_CLASSES": BALANCE_CLASSES,
        "BALANCING_SCOPE": "training_only",
        "filtered_class_counts": filtered_class_counts,
        "N_FOLDS": N_FOLDS,
        "GROUP_COLUMN": GROUP_COLUMN,
        "VALIDATION_SIZE_WITHIN_TRAIN": VALIDATION_SIZE_WITHIN_TRAIN,
        "N_COMPONENTS": N_COMPONENTS,
        "MODEL_SIZE": MODEL_SIZE,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "CLASSIFICATION_THRESHOLD": CLASSIFICATION_THRESHOLD,
        "SMOOTH_PREDICTION_PROBS": SMOOTH_PREDICTION_PROBS,
        "PREDICTION_SMOOTHING_METHOD": PREDICTION_SMOOTHING_METHOD,
        "PREDICTION_SMOOTHING_KERNEL_SIZE": PREDICTION_SMOOTHING_KERNEL_SIZE,
        "RANDOM_STATE": RANDOM_STATE,
        "PCA_RANDOM_STATE": PCA_RANDOM_STATE,
        "SAVE_PCA_ARTIFACTS": SAVE_PCA_ARTIFACTS,
        "SAVE_PREDICTION_MAPS": SAVE_PREDICTION_MAPS,
        "SAMPLE_RANKING_PRIMARY_METRIC": SAMPLE_RANKING_PRIMARY_METRIC,
        "SAMPLE_RANKING_SECONDARY_METRIC": SAMPLE_RANKING_SECONDARY_METRIC,
        "SHOW_PLOTS": SHOW_PLOTS,
    }
    with open(os.path.join(base_save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(config), f, indent=2)

    # Stratification reference: one row per pixel, grouped by Sample_ID.
    y_all = df_filtered["Binary_Label"].to_numpy(dtype=int)
    groups_all = df_filtered[GROUP_COLUMN].to_numpy()
    split_reference = np.zeros((len(df_filtered), 1), dtype=np.float32)

    outer_cv = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []
    sample_results = []
    fold_label_distribution = []
    history_frames = []
    cm_total = np.zeros((2, 2), dtype=int)

    for fold_number, (train_full_idx, test_idx) in enumerate(
        outer_cv.split(split_reference, y_all, groups=groups_all),
        start=1,
    ):
        print("\n" + "=" * 70)
        print(f"FOLD {fold_number}/{N_FOLDS}")
        print("=" * 70)

        fold_seed = RANDOM_STATE + fold_number
        set_seed(fold_seed)
        tf.keras.backend.clear_session()

        df_train_full = df_filtered.iloc[train_full_idx].copy()
        df_test = df_filtered.iloc[test_idx].copy()

        y_train_full = df_train_full["Binary_Label"].to_numpy(dtype=int)
        groups_train_full = df_train_full[GROUP_COLUMN].to_numpy()

        train_rel_idx, val_rel_idx = best_group_split(
            np.zeros((len(df_train_full), 1), dtype=np.float32),
            y_train_full,
            groups_train_full,
            test_size=VALIDATION_SIZE_WITHIN_TRAIN,
            random_state=fold_seed,
        )

        df_train_unbalanced = df_train_full.iloc[train_rel_idx].copy()
        df_val = df_train_full.iloc[val_rel_idx].copy()

        df_train, train_counts_before, train_counts_after = balance_binary_classes(
            df_train_unbalanced
        )

        train_groups = set(df_train_unbalanced[GROUP_COLUMN].astype(str))
        val_groups = set(df_val[GROUP_COLUMN].astype(str))
        test_groups = set(df_test[GROUP_COLUMN].astype(str))

        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise ValueError(f"Sample_ID leakage detected in fold {fold_number}.")

        print(
            "Group counts - "
            f"train: {len(train_groups)}, "
            f"validation: {len(val_groups)}, "
            f"test: {len(test_groups)}"
        )
        print(
            "Training class counts - "
            f"before balancing: {train_counts_before}, "
            f"after balancing: {train_counts_after}"
        )

        y_train = df_train["Binary_Label"].to_numpy(dtype=int)
        y_val = df_val["Binary_Label"].to_numpy(dtype=int)
        y_test = df_test["Binary_Label"].to_numpy(dtype=int)
        groups_test = df_test[GROUP_COLUMN].to_numpy()

        fold_label_distribution.extend(
            [
                label_distribution_row(
                    fold_number,
                    "train_before_balancing",
                    df_train_unbalanced["Binary_Label"].to_numpy(dtype=int),
                    df_train_unbalanced[GROUP_COLUMN].to_numpy(),
                ),
                label_distribution_row(fold_number, "train", y_train, df_train[GROUP_COLUMN].to_numpy()),
                label_distribution_row(fold_number, "validation", y_val, df_val[GROUP_COLUMN].to_numpy()),
                label_distribution_row(fold_number, "test", y_test, groups_test),
            ]
        )

        print(
            "Class 1 rate - "
            f"train: {y_train.mean():.3f}, "
            f"validation: {y_val.mean():.3f}, "
            f"test: {y_test.mean():.3f}"
        )

        if MAP_NORMALIZATION_MODE == "none":
            df_train_prepared = df_train
            df_val_prepared = df_val
            df_test_prepared = df_test
        elif MAP_NORMALIZATION_MODE == "filtered_labels":
            df_train_prepared = normalize_by_map(df_train, df_train_unbalanced, band_cols)
            df_val_prepared = normalize_by_map(df_val, df_val, band_cols)
            df_test_prepared = normalize_by_map(df_test, df_test, band_cols)
        elif MAP_NORMALIZATION_MODE == "full_map":
            train_context = df_raw[df_raw[GROUP_COLUMN].isin(df_train[GROUP_COLUMN].unique())]
            val_context = df_raw[df_raw[GROUP_COLUMN].isin(df_val[GROUP_COLUMN].unique())]
            test_context = df_raw[df_raw[GROUP_COLUMN].isin(df_test[GROUP_COLUMN].unique())]
            df_train_prepared = normalize_by_map(df_train, train_context, band_cols)
            df_val_prepared = normalize_by_map(df_val, val_context, band_cols)
            df_test_prepared = normalize_by_map(df_test, test_context, band_cols)
        elif MAP_NORMALIZATION_MODE == "spectrum":
            df_train_prepared = normalize_by_spectrum(df_train, band_cols)
            df_val_prepared = normalize_by_spectrum(df_val, band_cols)
            df_test_prepared = normalize_by_spectrum(df_test, band_cols)
        elif MAP_NORMALIZATION_MODE == "global_full":
            train_context = df_raw[df_raw[GROUP_COLUMN].isin(df_train[GROUP_COLUMN].unique())]
            global_means, global_stds = fit_global_normalization(train_context, band_cols)
            df_train_prepared = apply_global_normalization(df_train, band_cols, global_means, global_stds)
            df_val_prepared = apply_global_normalization(df_val, band_cols, global_means, global_stds)
            df_test_prepared = apply_global_normalization(df_test, band_cols, global_means, global_stds)
        elif MAP_NORMALIZATION_MODE == "global_filtered_labels":
            global_means, global_stds = fit_global_normalization(df_train_unbalanced, band_cols)
            df_train_prepared = apply_global_normalization(df_train, band_cols, global_means, global_stds)
            df_val_prepared = apply_global_normalization(df_val, band_cols, global_means, global_stds)
            df_test_prepared = apply_global_normalization(df_test, band_cols, global_means, global_stds)
        else:
            raise ValueError(
                "MAP_NORMALIZATION_MODE must be one of: 'none', 'filtered_labels', "
                "'full_map', 'spectrum', 'global_full', 'global_filtered_labels'."
            )

        print("Fitting PCA on training rows only...")
        pca = PCA(n_components=N_COMPONENTS, random_state=PCA_RANDOM_STATE)
        X_train = pca.fit_transform(df_train_prepared[band_cols].to_numpy(dtype=np.float32))
        X_val = pca.transform(df_val_prepared[band_cols].to_numpy(dtype=np.float32))
        X_test = pca.transform(df_test_prepared[band_cols].to_numpy(dtype=np.float32))

        if SAVE_PCA_ARTIFACTS:
            np.save(os.path.join(params_dir, f"pca_components_fold_{fold_number}.npy"), pca.components_)
            pd.DataFrame(
                {
                    "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                }
            ).to_csv(os.path.join(tables_dir, f"pca_explained_variance_fold_{fold_number}.csv"), index=False)

        model = ann_classification(num_components=N_COMPONENTS, size=MODEL_SIZE)
        if fold_number == 1:
            model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=10,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        history_frame = history_to_frame(history, fold_number)
        history_frame.to_csv(os.path.join(tables_dir, f"history_fold_{fold_number}.csv"), index=False)
        history_frames.append(history_frame)
        plot_history(history_frame, metrics_dir, f"METRICS_FOLD_{fold_number}.png", f"Fold {fold_number}")

        test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        y_pred_probs = model.predict(X_test, verbose=0).reshape(-1)
        y_pred_probs = smooth_prediction_probabilities(df_test, y_pred_probs)
        y_pred_classes = (y_pred_probs > CLASSIFICATION_THRESHOLD).astype("int32")

        cm = confusion_matrix(y_test, y_pred_classes, labels=[0, 1])
        cm_total += cm

        report_dict = classification_report(
            y_test,
            y_pred_classes,
            labels=[0, 1],
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        fold_auc = roc_auc_score(y_test, y_pred_probs) if len(np.unique(y_test)) == 2 else np.nan
        fold_result = {
            "fold": fold_number,
            "epochs_ran": len(history.history["loss"]),
            "train_rows_before_balancing": len(df_train_unbalanced),
            "train_rows": len(y_train),
            "validation_rows": len(y_val),
            "test_rows": len(y_test),
            "train_groups": len(train_groups),
            "validation_groups": len(val_groups),
            "test_groups": len(test_groups),
            "train_class_1_rate": y_train.mean(),
            "validation_class_1_rate": y_val.mean(),
            "test_class_1_rate": y_test.mean(),
            "train_class_0_before_balancing": train_counts_before.get(0, 0),
            "train_class_1_before_balancing": train_counts_before.get(1, 0),
            "train_class_0_after_balancing": train_counts_after.get(0, 0),
            "train_class_1_after_balancing": train_counts_after.get(1, 0),
            "smooth_prediction_probs": SMOOTH_PREDICTION_PROBS,
            "prediction_smoothing_method": PREDICTION_SMOOTHING_METHOD if SMOOTH_PREDICTION_PROBS else "none",
            "prediction_smoothing_kernel_size": PREDICTION_SMOOTHING_KERNEL_SIZE if SMOOTH_PREDICTION_PROBS else 0,
            "test_loss": test_metrics.get("loss"),
            "keras_test_accuracy_at_0_5": test_metrics.get("accuracy"),
            "keras_test_sensitivity_at_0_5": test_metrics.get("sensitivity"),
            "keras_test_auc_raw": test_metrics.get("auc"),
            "test_accuracy": accuracy_score(y_test, y_pred_classes),
            "test_sensitivity": report_dict[class_names[1]]["recall"],
            "test_auc": fold_auc,
            "test_auc_sklearn": fold_auc,
            "class_0_precision": report_dict[class_names[0]]["precision"],
            "class_0_recall": report_dict[class_names[0]]["recall"],
            "class_0_f1": report_dict[class_names[0]]["f1-score"],
            "class_1_precision": report_dict[class_names[1]]["precision"],
            "class_1_recall": report_dict[class_names[1]]["recall"],
            "class_1_f1": report_dict[class_names[1]]["f1-score"],
            "confusion_tn": int(cm[0, 0]),
            "confusion_fp": int(cm[0, 1]),
            "confusion_fn": int(cm[1, 0]),
            "confusion_tp": int(cm[1, 1]),
            "train_sample_ids": ", ".join(sorted(train_groups)),
            "validation_sample_ids": ", ".join(sorted(val_groups)),
            "test_sample_ids": ", ".join(sorted(test_groups)),
        }
        fold_results.append(fold_result)

        sample_results.extend(
            compute_sample_results(
                fold_number,
                y_test,
                y_pred_probs,
                y_pred_classes,
                groups_test,
                class_names,
            )
        )

        save_prediction_maps_if_enabled(df_test, y_pred_probs, y_pred_classes, fold_number, maps_dir)

        model_path = os.path.join(params_dir, f"ann_{MODEL_SIZE.lower()}_pca{N_COMPONENTS}_fold_{fold_number}.keras")
        weights_path = os.path.join(params_dir, f"ann_{MODEL_SIZE.lower()}_pca{N_COMPONENTS}_fold_{fold_number}.weights.h5")
        model.save(model_path)
        model.save_weights(weights_path)

        print(
            f"Fold {fold_number} test - "
            f"accuracy: {fold_result['test_accuracy']:.4f}, "
            f"auc: {fold_result['test_auc']:.4f}, "
            f"class_1 recall: {fold_result['class_1_recall']:.4f}, "
            f"class_1 precision: {fold_result['class_1_precision']:.4f}"
        )

        del model, X_train, X_val, X_test
        gc.collect()

    fold_results_df = pd.DataFrame(fold_results)
    sample_results_df = pd.DataFrame(sample_results)
    fold_label_distribution_df = pd.DataFrame(fold_label_distribution)

    fold_results_df.to_csv(os.path.join(tables_dir, "fold_results.csv"), index=False)
    sample_results_df.to_csv(os.path.join(tables_dir, "sample_results.csv"), index=False)
    fold_label_distribution_df.to_csv(os.path.join(tables_dir, "fold_label_distribution.csv"), index=False)
    sample_ranking_df = plot_sample_difficulty_ranking(sample_results_df, metrics_dir, tables_dir)

    history_all, history_mean, history_std = summarize_histories(history_frames)
    history_all.to_csv(os.path.join(tables_dir, "history_all_folds.csv"), index=False)
    history_mean.to_csv(os.path.join(tables_dir, "history_mean.csv"), index=False)
    history_std.to_csv(os.path.join(tables_dir, "history_std.csv"), index=False)

    numeric_cols = fold_results_df.select_dtypes(include=[np.number]).columns.drop("fold")
    summary_mean = fold_results_df[numeric_cols].mean()
    summary_std = fold_results_df[numeric_cols].std()

    summary_row = {
        "run_name": run_name,
        "model_size": MODEL_SIZE,
        "n_components": N_COMPONENTS,
        "n_folds": N_FOLDS,
        "map_normalization_mode": MAP_NORMALIZATION_MODE,
        "balance_classes": BALANCE_CLASSES,
        "balancing_scope": "training_only",
        "smooth_prediction_probs": SMOOTH_PREDICTION_PROBS,
        "prediction_smoothing_method": PREDICTION_SMOOTHING_METHOD if SMOOTH_PREDICTION_PROBS else "none",
        "prediction_smoothing_kernel_size": PREDICTION_SMOOTHING_KERNEL_SIZE if SMOOTH_PREDICTION_PROBS else 0,
        "class_0_labels": str(class_0_labels),
        "class_1_labels": str(class_1_labels),
        "mean_test_auc": summary_mean.get("test_auc"),
        "std_test_auc": summary_std.get("test_auc"),
        "mean_test_accuracy": summary_mean.get("test_accuracy"),
        "std_test_accuracy": summary_std.get("test_accuracy"),
        "mean_class_1_precision": summary_mean.get("class_1_precision"),
        "std_class_1_precision": summary_std.get("class_1_precision"),
        "mean_class_1_recall": summary_mean.get("class_1_recall"),
        "std_class_1_recall": summary_std.get("class_1_recall"),
        "mean_class_1_f1": summary_mean.get("class_1_f1"),
        "std_class_1_f1": summary_std.get("class_1_f1"),
        "mean_epochs_ran": summary_mean.get("epochs_ran"),
        "std_epochs_ran": summary_std.get("epochs_ran"),
    }
    pd.DataFrame([summary_row]).to_csv(os.path.join(tables_dir, "comparative_results.csv"), index=False)

    cm_normalized = plot_confusion_matrices(cm_total, class_names, metrics_dir)
    plot_mean_history(history_mean, history_std, metrics_dir)

    run_summary_path = os.path.join(base_save_dir, "run_summary.txt")
    with open(run_summary_path, "w", encoding="utf-8") as f:
        f.write("Fold-wise Map Normalization + PCA + ANN Binary Classification\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Run directory: {base_save_dir}\n\n")

        f.write("Configuration\n")
        f.write("-" * 70 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Dataset\n")
        f.write("-" * 70 + "\n")
        f.write(f"Raw dataset shape: {df_raw.shape}\n")
        f.write(f"Filtered dataset shape: {df_filtered.shape}\n")
        f.write(f"Spectral bands: {len(band_cols)}\n")
        f.write(f"Unique {GROUP_COLUMN}: {df_filtered[GROUP_COLUMN].nunique()}\n")
        f.write(f"Filtered class counts: {filtered_class_counts}\n")
        f.write("Balancing scope: training partition only, separately in each fold\n")
        f.write("Validation and test rows: complete and not balanced\n\n")

        f.write("Mean Test Metrics Across Folds\n")
        f.write("-" * 70 + "\n")
        for name in numeric_cols:
            f.write(f"{name}: {summary_mean[name]:.6f} +/- {summary_std[name]:.6f}\n")
        f.write("\n")

        f.write("Total Confusion Matrix Across Folds\n")
        f.write("-" * 70 + "\n")
        f.write(str(cm_total))
        f.write("\n\n")

        f.write("Normalized Total Confusion Matrix Across Folds\n")
        f.write("-" * 70 + "\n")
        f.write(str(cm_normalized))
        f.write("\n\n")

        f.write("Output Tables\n")
        f.write("-" * 70 + "\n")
        f.write("tables/fold_results.csv\n")
        f.write("tables/sample_results.csv\n")
        if sample_ranking_df is not None:
            f.write("tables/sample_difficulty_ranking.csv\n")
        f.write("tables/fold_label_distribution.csv\n")
        f.write("tables/history_all_folds.csv\n")
        f.write("tables/history_mean.csv\n")
        f.write("tables/history_std.csv\n")
        f.write("tables/comparative_results.csv\n")

    print("\n" + "=" * 70)
    print("5-FOLD SUMMARY")
    print("=" * 70)
    for metric_name in ["test_accuracy", "test_auc", "class_1_precision", "class_1_recall", "class_1_f1"]:
        print(f"{metric_name}: {summary_mean[metric_name]:.4f} +/- {summary_std[metric_name]:.4f}")
    print(f"\nRun saved at: {base_save_dir}")


if __name__ == "__main__":
    main()
