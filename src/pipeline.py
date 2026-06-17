"""
Reusable machinery for the Raman binary-classification pipeline.

Covers preprocessing (SNV, PCA), class balancing, leakage-safe fold
generation, model training, and metric computation. PCA is always fitted
on the training portion only — see `prepare_fold`.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)

from src.model import ann_classification


# ── preprocessing ─────────────────────────────────────────────────────────────

def snv(X):
    """Standard Normal Variate: per-spectrum mean-centre and scale to unit std.

    Operates row-wise, so it uses no statistics shared across samples and is
    therefore leakage-safe regardless of how the data is split.
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / sd


def balance_indices(y, idx, cap=None, seed=42):
    """Downsample every class within `idx` to the size of the minority class.

    Returns a shuffled subset of `idx`. Applied to TRAINING indices only so
    validation/test keep their natural class distribution.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y[idx])
    n = min(int((y[idx] == c).sum()) for c in classes)
    if cap is not None:
        n = min(n, cap)
    keep = np.concatenate([
        rng.choice(idx[y[idx] == c], n, replace=False) for c in classes
    ])
    rng.shuffle(keep)
    return keep


def prepare_fold(X, tr_idx, val_idx, te_idx, max_pc, use_snv, seed=42):
    """SNV (optional) then PCA fitted on the TRAINING split only.

    Returns the train/val/test matrices projected onto `max_pc` components.
    Slice the result `[:, :n_pc]` to obtain any smaller component count — this
    is identical to refitting PCA with `n_pc` components, since PCA components
    are ordered and nested.
    """
    X_tr, X_val, X_te = X[tr_idx], X[val_idx], X[te_idx]
    if use_snv:
        X_tr, X_val, X_te = snv(X_tr), snv(X_val), snv(X_te)
    pca = PCA(n_components=max_pc, random_state=seed).fit(X_tr)
    return pca.transform(X_tr), pca.transform(X_val), pca.transform(X_te)


# ── leakage-safe splitting ────────────────────────────────────────────────────

def make_folds(y, groups, cv, group_aware=True, val_frac=0.15, test_frac=0.15, seed=42):
    """Generate a list of (train_idx, val_idx, test_idx) tuples.

    cv=None  -> a single train/val/test split (one element).
    cv=k     -> k folds; the test split is the held-out outer fold and the
                validation split is carved from that fold's training portion.

    With group_aware=True every Sample_ID stays entirely within one split,
    which is essential here (only ~25 samples, highly correlated pixels).
    """
    idx = np.arange(len(y))

    # --- single split ---
    if cv is None:
        if group_aware:
            outer = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
            trv, te = next(outer.split(idx, y, groups))
            inner = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac),
                                      random_state=seed)
            tr_rel, val_rel = next(inner.split(trv, y[trv], groups[trv]))
        else:
            trv, te = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=y)
            tr_rel, val_rel = train_test_split(np.arange(len(trv)),
                                               test_size=val_frac / (1 - test_frac),
                                               random_state=seed, stratify=y[trv])
        return [(trv[tr_rel], trv[val_rel], te)]

    # --- k-fold cross-validation ---
    if group_aware:
        outer = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
        splitter = outer.split(idx, y, groups)
    else:
        outer = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        splitter = outer.split(idx, y)

    folds = []
    for tr_full, te in splitter:
        if group_aware:
            inner = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr_rel, val_rel = next(inner.split(tr_full, y[tr_full], groups[tr_full]))
        else:
            tr_rel, val_rel = train_test_split(np.arange(len(tr_full)), test_size=val_frac,
                                               random_state=seed, stratify=y[tr_full])
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


METRIC_NAMES = ['accuracy', 'auc', 'precision', 'recall', 'f1']


def evaluate_fold(model, X_te, y_te):
    """Predict on the test split and return (metrics_dict, y_prob).

    Precision / recall / F1 are reported for the positive (Tumoral) class.
    """
    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob > 0.45).astype(int)
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
