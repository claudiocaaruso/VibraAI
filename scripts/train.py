"""
Train ANN models for binary Raman spectroscopy classification.

Two execution modes, selected by MODE below:

  'single' — one configuration (architecture / PCA dim / SNV / CV). Shows all
             plots for inspection; saves nothing unless SAVE_RESULTS is set.
  'grid'   — sweeps ARCHITECTURE_VARIANTS x PC_VARIANTS for the single SNV and
             CV settings configured below. Displays no plots and saves all
             metrics, summaries and figures under results/grid/.

PCA is fitted on the training split of each fold only (no leakage); for CV it
is re-fitted independently per fold. Set the config block, then run.

See docs/pipeline.md for a full walkthrough of every function and the flow.
"""
import gc
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from itertools import product
from pathlib import Path
from src import plots
from src.pipeline import (balance_indices, evaluate_fold, make_folds,
                          prepare_fold, summarize_metrics, train_model)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / 'datasets' / 'spectral_dataset.parquet'

# ══════════════════════════════ CONFIG ════════════════════════════════════════

MODE = 'single'                 # 'single' or 'grid'

# --- dataset & labels (fully configurable) ---
TUMOR_LABELS   = [2,20]     # mapped to the positive class (1 = Tumoral)
EXCLUDE_LABELS = [-1, 15, 0, 19, 23, 8, 3, 10]       # dropped from the dataset before training
SUBSAMPLE      = None        # cap on TOTAL rows kept (random, leakage-free); None = all

# --- single-mode configuration ---
ARCHITECTURE = 'L'              # 'S' | 'M' | 'L'
N_PC         = 483             # number of PCA components
SNV          = True             # SNV normalisation on/off
CV           = 5             # None (single split) | 5 | 10

# --- grid-mode variants (only these two sweep; SNV and CV stay fixed above) ---
ARCHITECTURE_VARIANTS = ['S', 'M', 'L']
PC_VARIANTS           = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 200, 300, 483]

# --- splitting / balancing ---
GROUP_AWARE = True              # keep each Sample_ID within one split (recommended)
BALANCE     = True              # downsample majority class within the training split
BALANCE_CAP = 500000           # max rows PER CLASS kept after balancing the train split

# --- training hyper-parameters ---
EPOCHS     = 100                # max epochs 
BATCH_SIZE = int(8192)
# Further training knobs live deeper in the code:
#   - early-stopping patience, LR schedule  -> src/pipeline.py  (train_model)
#   - layers / optimizer / loss / metrics   -> src/model.py     (ann_classification)

# --- output ---
SAVE_RESULTS = (MODE == 'grid')   # single mode: set True to also persist outputs
SHOW_PLOTS   = (MODE == 'single')

# ═══════════════════════════ APPLY CONFIG ═════════════════════════════════════

if not SHOW_PLOTS:
    plt.switch_backend('Agg')     # headless: figures are saved, never displayed

# ══════════════════════════════ LOAD DATA ═════════════════════════════════════

print("Loading spectral dataset …")
df = pd.read_parquet(DATA_PATH, engine='pyarrow')
band_cols = [c for c in df.columns if c.startswith('band_')]

df = df[~df['Label'].isin(EXCLUDE_LABELS)].copy()
df['y'] = df['Label'].isin(TUMOR_LABELS).astype(int)

if SUBSAMPLE and len(df) > SUBSAMPLE:
    df = df.sample(SUBSAMPLE, random_state=42).reset_index(drop=True)

X      = df[band_cols].to_numpy(dtype=np.float32)
y      = df['y'].to_numpy()
groups = df['Sample_ID'].to_numpy()
del df; gc.collect()

print(f"X: {X.shape} | Tumoral: {y.sum()} | Healthy: {(y == 0).sum()} "
      f"| {len(np.unique(groups))} samples")

# ══════════════════════════ BUILD THE RUN GRID ════════════════════════════════

if MODE == 'single':
    architecture_list, pc_list = [ARCHITECTURE], [N_PC]
else:
    architecture_list, pc_list = ARCHITECTURE_VARIANTS, PC_VARIANTS

max_pc      = max(pc_list)
results_dir = ROOT / 'results' / ('grid' if MODE == 'grid' else 'single')
all_summaries, all_folds = [], []

# ══════════════════════════════ MAIN LOOP ═════════════════════════════════════
# SNV and CV are fixed for the run. For each fold we fit PCA once at max_pc, then
# the inner architecture/n_pc loops reuse it via slicing. Per-fold model results
# are accumulated per (architecture, n_pc) and aggregated after all folds.

folds   = make_folds(y, groups, CV, group_aware=GROUP_AWARE)
n_folds = len(folds)
acc     = defaultdict(lambda: {'hist': [], 'metrics': [], 'roc': []})

for fi, (tr_idx, val_idx, te_idx) in enumerate(folds, start=1):
    if BALANCE:
        tr_idx = balance_indices(y, tr_idx, cap=BALANCE_CAP)

    X_tr, X_val, X_te = prepare_fold(X, tr_idx, val_idx, te_idx, max_pc, SNV)
    y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]

    for architecture, n_pc in product(architecture_list, pc_list):
        print(f"\n{'#'*64}\n  cv={CV} snv={SNV} | fold {fi}/{n_folds} "
              f"| architecture={architecture} n_pc={n_pc}\n{'#'*64}")
        tf.keras.backend.clear_session()
        model, history = train_model(
            X_tr[:, :n_pc], y_tr, X_val[:, :n_pc], y_val, architecture,
            verbose=1 if MODE == 'single' else 0,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
        )
        metrics, y_prob = evaluate_fold(model, X_te[:, :n_pc], y_te)
        metrics |= {'cv': CV, 'snv': SNV, 'architecture': architecture, 'n_pc': n_pc, 'fold': fi}

        store = acc[(architecture, n_pc)]
        store['hist'].append(history.history)
        store['metrics'].append(metrics)
        store['roc'].append((y_te, y_prob))
        print(f"  AUC={metrics['auc']:.4f} acc={metrics['accuracy']:.4f} "
              f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

    del X_tr, X_val, X_te; gc.collect()

# ══════════════════════ AGGREGATE EVERY CONFIGURATION ═════════════════════════

for (architecture, n_pc), store in acc.items():
    per_fold_df, summary = summarize_metrics(store['metrics'])
    summary |= {'cv': CV, 'snv': SNV, 'architecture': architecture,
                'n_pc': n_pc, 'n_folds': n_folds}
    all_summaries.append(summary)
    all_folds.extend(store['metrics'])

    tag = f"cv{CV}_snv{int(SNV)}_{architecture}_PC{n_pc}"
    print(f"\n=== {tag} === AUC {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f} "
          f"| recall {summary['recall_mean']:.4f} | f1 {summary['f1_mean']:.4f}")

    save_dir = None
    if SAVE_RESULTS:
        save_dir = results_dir / f'cv{CV}' / f'snv{int(SNV)}' / architecture / f'PC{n_pc}'
        save_dir.mkdir(parents=True, exist_ok=True)
        per_fold_df.to_csv(save_dir / 'fold_metrics.csv', index=False)

    def _p(name):                       # figure path or None
        return str(save_dir / name) if save_dir else None

    aucs = [m['auc'] for m in store['metrics']]
    plots.plot_training_curves(store['hist'], title=f'Training curves – {tag}',
                               save_path=_p('training_curves.png'), show=SHOW_PLOTS)
    plots.plot_roc(store['roc'], title=f'ROC – {tag}',
                   save_path=_p('roc.png'), show=SHOW_PLOTS)
    plots.plot_confusion(store['roc'], title=f'Confusion – {tag}',
                         save_path=_p('confusion.png'), show=SHOW_PLOTS)
    if n_folds > 1:
        plots.plot_fold_auc(aucs, title=f'AUC per fold – {tag}',
                            save_path=_p('fold_auc.png'), show=SHOW_PLOTS)

# ══════════════════════════════ SUMMARIES ═════════════════════════════════════

summary_df = pd.DataFrame(all_summaries)
print("\n" + "=" * 70)
print("Top configurations by mean AUC:")
print(summary_df.sort_values('auc_mean', ascending=False)
      .head(10).to_string(index=False))

if SAVE_RESULTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(results_dir / 'all_summaries.csv', index=False)
    pd.DataFrame(all_folds).to_csv(results_dir / 'all_folds.csv', index=False)
    print(f"\nResults saved to: {results_dir}")
