import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import generic_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import ann_classification

from data_giulia import class_counts


def snv(X, eps=1e-8):
    """Standard Normal Variate normalization, applied independently to each spectrum."""
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds = np.where(np.abs(stds) < eps, 1.0, stds)
    return (X - means) / stds


def balance_indices(y, idx, balance_classes=True, seed=42):
    """Downsample every class within idx to the minority-class size."""
    idx = np.asarray(idx)
    y_subset = y[idx]
    counts = class_counts(y_subset)

    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"Both binary classes must be present in training. Counts: {counts}")

    rng = np.random.default_rng(seed)
    if not balance_classes:
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        return shuffled

    samples_per_class = min(counts.values())
    keep = np.concatenate(
        [
            rng.choice(idx[y_subset == class_value], samples_per_class, replace=False)
            for class_value in [0, 1]
        ]
    )
    rng.shuffle(keep)
    return keep


def make_folds(y, groups, n_folds, validation_size_within_train, seed=42):
    """Return leakage-safe (train, validation, test) index tuples grouped by Sample_ID."""
    outer_cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []

    for train_full_idx, test_idx in outer_cv.split(np.zeros((len(y), 1)), y, groups=groups):
        inner_split = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_size_within_train,
            random_state=seed,
        )
        train_rel_idx, val_rel_idx = next(
            inner_split.split(
                np.zeros((len(train_full_idx), 1)),
                y[train_full_idx],
                groups[train_full_idx],
            )
        )
        folds.append((train_full_idx[train_rel_idx], train_full_idx[val_rel_idx], test_idx))

    return folds


def prepare_fold(X, train_idx, val_idx, test_idx, n_components, use_snv, pca_seed=42, snv_eps=1e-8):
    """Apply optional SNV and fit PCA on training rows only."""
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    if use_snv:
        X_train = snv(X_train, eps=snv_eps)
        X_val = snv(X_val, eps=snv_eps)
        X_test = snv(X_test, eps=snv_eps)

    pca = PCA(n_components=n_components, random_state=pca_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca, pca


def train_model(X_train, y_train, X_val, y_val, model_size, epochs, batch_size, verbose=1):
    """Build and fit an ANN with early stopping on validation AUC."""
    model = ann_classification(num_components=X_train.shape[1], size=model_size)
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
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history


def smooth_prediction_probabilities(
    df_rows,
    y_probs,
    map_group_columns,
    smooth_prediction_probs,
    smoothing_method,
    kernel_size,
):
    """Smooth predicted class-1 probabilities inside each Raman map."""
    y_probs = np.asarray(y_probs).reshape(-1)
    if not smooth_prediction_probs:
        return y_probs

    needed_cols = map_group_columns + ["x", "y"]
    missing = [col for col in needed_cols if col not in df_rows.columns]
    if missing:
        print(f"Skipping prediction smoothing: missing columns {missing}.")
        return y_probs

    if kernel_size < 2:
        return y_probs

    if smoothing_method == "mean":
        smooth_func = np.nanmean
    elif smoothing_method == "median":
        smooth_func = np.nanmedian
    else:
        raise ValueError("smoothing_method must be one of: 'mean', 'median'.")

    work = df_rows[needed_cols].copy().reset_index(drop=True)
    work["_row_order"] = np.arange(len(work))
    work["_pred_prob"] = y_probs
    smoothed_parts = []

    for _, map_df in work.groupby(map_group_columns, sort=False):
        grid = (
            map_df.pivot_table(index="y", columns="x", values="_pred_prob", aggfunc="mean")
            .sort_index()
            .sort_index(axis=1)
        )
        smoothed_grid = generic_filter(
            grid.to_numpy(dtype=np.float32),
            function=smooth_func,
            size=kernel_size,
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
