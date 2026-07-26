"""
Compare raw vs. spatially-smoothed predictions across the full 5-fold CV for
one configuration (thesis Section 4.9 / 5.7).

Trains one model per fold, then — for each kernel size in KERNEL_SIZES —
pools the (possibly smoothed) test predictions across all 5 folds, the same
way scripts/train.py pools its ROC curve and confusion matrix.

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
ARCHITECTURE   = 'S'
N_PC           = 10
BATCH_SIZE     = 4096
KERNEL_SIZES   = [1, 3, 5]     # 1 = raw (no smoothing)

set_seed(41)

print("Loading spectral dataset …")
df = pd.read_parquet(DATA_PATH, engine='pyarrow')
band_cols = [c for c in df.columns if c.startswith('band_')]
df = df[~df['Label'].isin(EXCLUDE_LABELS)].copy()
df['y'] = df['Label'].isin(TUMOR_LABELS).astype(int)

X      = df[band_cols].to_numpy(dtype=np.float32)
y      = df['y'].to_numpy()
groups = df['Sample_ID'].to_numpy()
meta   = df[['Sample_ID', 'Map_ID', 'x', 'y']]

folds   = make_folds(y, groups)
n_folds = len(folds)


def metrics_at(y_true, y_prob):
    y_pred = (y_prob > Y_PROB_BIAS).astype(int)
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'auc':       roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }


# ── train one model per fold, keep each fold's raw test predictions ───────────

hist_per_fold = []
raw_per_fold  = []   # (y_te, y_prob_raw, df_te) per fold

for fi, (tr_idx, val_idx, te_idx) in enumerate(folds, start=1):
    print(f"\n{'#'*64}\n  fold {fi}/{n_folds} | {ARCHITECTURE}/PC{N_PC}\n{'#'*64}")
    tr_idx = balance_indices(y, tr_idx)

    X_tr, X_val, X_te = prepare_fold(X, tr_idx, val_idx, te_idx, N_PC)
    y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]
    df_te = meta.iloc[te_idx]

    tf.keras.backend.clear_session()
    set_seed(43)
    model, history = train_model(X_tr, y_tr, X_val, y_val, ARCHITECTURE, batch_size=BATCH_SIZE)
    y_prob_raw = model.predict(X_te, verbose=0).flatten()

    hist_per_fold.append(history.history)
    raw_per_fold.append((y_te, y_prob_raw, df_te))

plots.plot_training_curves(hist_per_fold, title=f'Training curves – {ARCHITECTURE}_PC{N_PC}',
                           show=True)

# ── for each kernel size, pool the (smoothed) predictions across all folds ───

for k in KERNEL_SIZES:
    label = 'raw' if k == 1 else f'{k}x{k}'
    roc_data, fold_metrics = [], []
    for y_te, y_prob_raw, df_te in raw_per_fold:
        y_prob = smooth_probabilities(df_te, y_prob_raw, k)
        roc_data.append((y_te, y_prob))
        fold_metrics.append(metrics_at(y_te, y_prob))

    _, summary = summarize_metrics(fold_metrics)
    print(f"\n=== {label} === AUC {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f} "
          f"| acc {summary['accuracy_mean']:.4f} | recall {summary['recall_mean']:.4f} "
          f"| f1 {summary['f1_mean']:.4f}")

    plots.plot_roc(roc_data, title=f'ROC – {label}', show=True)
    plots.plot_confusion(roc_data, title=f'Confusion – {label}', show=True)
