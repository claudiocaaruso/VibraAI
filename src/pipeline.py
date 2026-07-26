"""
Reusable machinery for the Raman binary-classification pipeline.

Covers preprocessing (SNV, PCA), class balancing, leakage-safe fold
generation, model training, and metric computation. PCA is always fitted
on the training portion only — see `prepare_fold`.

SNV, group-aware 5-fold CV, and training-only class balancing are fixed
parts of the pipeline rather than run-time toggles.
"""
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import generic_filter
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.model import ann_classification

Y_PROB_BIAS = 0.49


def set_seed(seed=43):
    """Seed Python, NumPy, and TensorFlow RNGs for reproducible model init/training."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ── preprocessing ─────────────────────────────────────────────────────────────

def snv(X):
    """Standard Normal Variate: per-spectrum mean-centre and scale to unit std.

    Operates row-wise, so it uses no statistics shared across samples and is
    therefore leakage-safe regardless of how the data is split.
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / sd


def balance_indices(y, idx, cap=500000, seed=43):
    """Downsample every class within `idx` to the size of the minority class.

    Returns a shuffled subset of `idx`. Applied to TRAINING indices only so
    validation/test keep their natural class distribution.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y[idx])
    n = min(int((y[idx] == c).sum()) for c in classes)
    n = min(n, cap)
    keep = np.concatenate([
        rng.choice(idx[y[idx] == c], n, replace=False) for c in classes
    ])
    rng.shuffle(keep)
    return keep


def prepare_fold(X, tr_idx, val_idx, te_idx, max_pc, seed=43):
    """SNV, then PCA fitted on the TRAINING split only.

    Returns the train/val/test matrices projected onto `max_pc` components.
    Slice the result `[:, :n_pc]` to obtain any smaller component count — this
    is identical to refitting PCA with `n_pc` components, since PCA components
    are ordered and nested.
    """
    X_tr, X_val, X_te = snv(X[tr_idx]), snv(X[val_idx]), snv(X[te_idx])
    pca = PCA(n_components=max_pc, random_state=seed).fit(X_tr)
    return pca.transform(X_tr), pca.transform(X_val), pca.transform(X_te)


# ── leakage-safe splitting ────────────────────────────────────────────────────

def make_folds(y, groups, cv=5, val_frac=0.1875, seed=43):
    """Generate a list of (train_idx, val_idx, test_idx) tuples via group-aware
    stratified k-fold CV.

    The test split is the held-out outer fold; the validation split is carved
    from that fold's training portion. Every Sample_ID stays entirely within
    one split, which is essential here (only ~25 samples, highly correlated
    pixels).
    """
    idx = np.arange(len(y))
    outer = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)

    folds = []
    for tr_full, te in outer.split(idx, y, groups):
        inner = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_rel, val_rel = next(inner.split(tr_full, y[tr_full], groups[tr_full]))
        folds.append((tr_full[tr_rel], tr_full[val_rel], te))
    return folds


# ── training & evaluation ─────────────────────────────────────────────────────

def train_model(X_tr, y_tr, X_val, y_val, architecture, verbose=1, epochs=100, batch_size=512):
    """Build and fit an ANN with early stopping on validation AUC."""
    model = ann_classification(X_tr.shape[1], architecture)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                         patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=4, min_lr=1e-6),
    ]
    history = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=callbacks,
                        verbose=verbose)
    return model, history


def smooth_probabilities(df_te, y_prob, kernel_size):
    """Spatially smooth predicted probabilities with a kernel_size x kernel_size
    mean filter, applied independently within each (Sample_ID, Map_ID) map —
    the post-processing step described in the thesis (Section 4.9).

    `df_te` must have Sample_ID/Map_ID/x/y columns aligned (same row order)
    with `y_prob`. Boundary pixels reuse the nearest in-map value
    (mode='nearest'); missing grid cells are ignored via nanmean.
    kernel_size < 2 returns `y_prob` unchanged.
    """
    if kernel_size < 2:
        return y_prob

    work = df_te[['Sample_ID', 'Map_ID', 'x', 'y']].reset_index(drop=True).copy()
    work['prob']   = np.asarray(y_prob)
    work['_order'] = np.arange(len(work))

    smoothed = np.empty(len(work))
    for _, map_df in work.groupby(['Sample_ID', 'Map_ID'], sort=False):
        grid = (map_df.pivot_table(index='y', columns='x', values='prob')
                      .sort_index().sort_index(axis=1))
        filtered = generic_filter(grid.to_numpy(dtype=np.float64), np.nanmean,
                                  size=kernel_size, mode='nearest')
        flat = (pd.DataFrame(filtered, index=grid.index, columns=grid.columns)
                  .stack().rename('smoothed').reset_index())
        merged = map_df.merge(flat, on=['y', 'x'], how='left')
        smoothed[merged['_order'].to_numpy()] = merged['smoothed'].to_numpy()
    return smoothed


METRIC_NAMES = ['accuracy', 'auc', 'precision', 'recall', 'f1']


def evaluate_fold(model, X_te, y_te):
    """Predict on the test split and return (metrics_dict, y_prob).

    Precision / recall / F1 are reported for the positive (Tumoral) class.
    To evaluate with spatial smoothing instead, smooth the `y_prob` this
    returns via `smooth_probabilities` and recompute the metrics yourself.
    """
    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob > Y_PROB_BIAS).astype(int)
    try:
        auc = roc_auc_score(y_te, y_prob)
    except ValueError:           # single class present in this test split
        auc = np.nan
    metrics = {
        'accuracy':  accuracy_score(y_te, y_pred),
        'auc':       auc,
        'precision': precision_score(y_te, y_pred, zero_division=0),
        'recall':    recall_score(y_te, y_pred, zero_division=0),
        'f1':        f1_score(y_te, y_pred, zero_division=0),
    }
    return metrics, y_prob


def summarize_metrics(metrics_rows):
    """Return (per_fold_df, summary_dict) with mean/std across folds.

    Std uses ddof=1 when there is more than one fold, else 0.
    """
    df = pd.DataFrame(metrics_rows)
    summary = {}
    for col in METRIC_NAMES:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        summary[f'{col}_mean'] = np.nanmean(vals)
        summary[f'{col}_std']  = np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0
    return df, summary
