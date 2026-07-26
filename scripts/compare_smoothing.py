"""
Compare raw vs. spatially-smoothed predictions across the full 5-fold CV for
one configuration (thesis Section 4.9 / 5.7).

Trains one model per fold, smooths every fold's test predictions with
KERNEL_SIZE (1 = off), pools them across all 5 folds the same way
scripts/train.py pools its ROC curve and confusion matrix, then plots a
true-labels / predicted-probability / error map for every individual Raman
map in the test set.

Fully separate from scripts/train.py — nothing here is needed for the
regular training/grid-search runs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import plots
from src.pipeline import (Y_PROB_BIAS, balance_indices, make_folds,
                          prepare_fold, set_seed, smooth_probabilities,
                          summarize_metrics, train_model)

DATA_PATH = ROOT / 'datasets' / 'spectral_dataset.parquet'

TUMOR_LABELS   = [2, 20]
EXCLUDE_LABELS = [-1, 15, 0, 19, 23, 8, 3, 10, 5, 9, 4]
# EXCLUDE_LABELS = [-1, 15]
ARCHITECTURE   = 'S'
N_PC           = 10
BATCH_SIZE     = 4096
KERNEL_SIZE    = 3              # w x w mean filter; 1 = off (raw, no smoothing)
SAMPLE_ID      = None           # e.g. '2787_024' to only plot that sample's maps; None = plot all

set_seed(43)

print("Loading spectral dataset …")
df = pd.read_parquet(DATA_PATH, engine='pyarrow')
band_cols = [c for c in df.columns if c.startswith('band_')]
df = df[~df['Label'].isin(EXCLUDE_LABELS)].copy()
df['is_tumor'] = df['Label'].isin(TUMOR_LABELS).astype(int)

X      = df[band_cols].to_numpy(dtype=np.float32)
y      = df['is_tumor'].to_numpy()
groups = df['Sample_ID'].to_numpy()
meta   = df[['Sample_ID', 'Map_ID', 'x', 'y']]

folds   = make_folds(y, groups)
n_folds = len(folds)
label   = 'raw' if KERNEL_SIZE == 1 else f'{KERNEL_SIZE}x{KERNEL_SIZE} smoothing'


def metrics_at(y_true, y_prob):
    y_pred = (y_prob > Y_PROB_BIAS).astype(int)
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'auc':       roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }


# ── train one model per fold, smooth its test predictions ─────────────────────

hist_per_fold = []
smoothed_per_fold = []   # (y_te, y_prob, df_te) per fold, already smoothed

for fi, (tr_idx, val_idx, te_idx) in enumerate(folds, start=1):
    print(f"\n{'#'*64}\n  fold {fi}/{n_folds} | {ARCHITECTURE}/PC{N_PC}\n{'#'*64}")
    tr_idx = balance_indices(y, tr_idx)

    X_tr, X_val, X_te = prepare_fold(X, tr_idx, val_idx, te_idx, N_PC)
    y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]
    df_te = meta.iloc[te_idx].reset_index(drop=True)

    tf.keras.backend.clear_session()
    set_seed(43)
    model, history = train_model(X_tr, y_tr, X_val, y_val, ARCHITECTURE, batch_size=BATCH_SIZE)
    y_prob_raw = model.predict(X_te, verbose=0).flatten()
    y_prob = smooth_probabilities(df_te, y_prob_raw, KERNEL_SIZE)

    hist_per_fold.append(history.history)
    smoothed_per_fold.append((y_te, y_prob, df_te))

plots.plot_training_curves(hist_per_fold, title=f'Training curves – {ARCHITECTURE}_PC{N_PC}',
                           show=True)

# ── pool predictions across folds: metrics, ROC, confusion ────────────────────

roc_data     = [(y_te, y_prob) for y_te, y_prob, _ in smoothed_per_fold]
fold_metrics = [metrics_at(y_te, y_prob) for y_te, y_prob, _ in smoothed_per_fold]

_, summary = summarize_metrics(fold_metrics)
print(f"\n=== {label} === AUC {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f} "
      f"| acc {summary['accuracy_mean']:.4f} | recall {summary['recall_mean']:.4f} "
      f"| f1 {summary['f1_mean']:.4f}")

plots.plot_roc(roc_data, title=f'ROC – {label}', show=True)
plots.plot_confusion(roc_data, title=f'Confusion – {label}', show=True)

# ── true labels / predicted probability / error map, per Raman map ───────────

for y_te, y_prob, df_te in smoothed_per_fold:
    for (sample_id, map_id), map_rows in df_te.groupby(['Sample_ID', 'Map_ID'], sort=False):
        if SAMPLE_ID is not None and sample_id != SAMPLE_ID:
            continue
        pos = map_rows.index.to_numpy()   # positional, since df_te's index was reset
        plots.plot_prediction_map(
            map_rows['x'], map_rows['y'], y_te[pos], y_prob[pos], Y_PROB_BIAS,
            title=f'Sample {sample_id} / {map_id} – {label}', show=True,
        )
