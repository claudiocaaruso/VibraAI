# Raman Classification Pipeline

Binary classification (Healthy vs. Tumoral) of Raman spectra with an ANN, on
top of SNV normalisation and PCA dimensionality reduction. This document
explains the end-to-end flow and every function involved.

## File map

| File | Role |
|------|------|
| `scripts/train.py` | **Orchestrator** you run. Config block + visible top-to-bottom flow. |
| `src/pipeline.py`  | Machinery: preprocessing, splitting, training, metrics. |
| `src/plots.py`     | All visualisations. |
| `src/model.py`     | ANN architecture definitions. |
| `scripts/pca.py`   | Standalone PCA exploration (not part of training). |
| `scripts/explore_dataset.py` | Standalone EDA (not part of training). |

The dataset (`datasets/spectral_dataset.parquet`) has one row per pixel:
metadata columns (`Sample_ID`, `Map_ID`, `x`, `y`, `Label`) plus 483 intensity
columns `band_0 … band_482`. There are ~25 unique samples.

---

## How a run works (high-level)

```
load raw spectra
  └─ drop EXCLUDE_LABELS, map TUMOR_LABELS → 1 (Tumoral) else 0 (Healthy)
  └─ optional SUBSAMPLE of rows
make_folds            → list of (train, val, test) index splits
for each fold:
    balance_indices   → downsample majority class in TRAIN only
    prepare_fold      → SNV (optional) + PCA fitted on TRAIN only, at max_pc
    for each (architecture, n_pc):
        slice the PCA output to n_pc components
        train_model   → build + fit ANN (early stopping on val AUC)
        evaluate_fold → metrics + probabilities on TEST
aggregate per (architecture, n_pc):
    summarize_metrics → mean ± std across folds
    plots             → training curves, ROC, confusion, fold-AUC bar
save / show
```

The single most important property: **any step that learns statistics (PCA) is
fitted on the training split only**, and re-fitted independently for every fold.
SNV is computed per spectrum (row-wise), so it never shares information across
samples and is leakage-safe anywhere.

---

## Execution modes

Set `MODE` in `scripts/train.py`:

- **`'single'`** — runs one configuration (`ARCHITECTURE`, `N_PC`, `SNV`, `CV`).
  Shows every plot interactively; saves nothing unless you set `SAVE_RESULTS = True`.
- **`'grid'`** — sweeps `ARCHITECTURE_VARIANTS × PC_VARIANTS`. **`SNV` and `CV`
  stay fixed** at their single-value settings (only architecture and PCA dim
  vary). No plots are displayed; everything is written under `results/grid/`.

`SHOW_PLOTS` and `SAVE_RESULTS` default from `MODE` but can be overridden.

---

## Configuration reference (`scripts/train.py`)

### Dataset & labels
- **`TUMOR_LABELS`** — label values mapped to the positive class (`1`, Tumoral).
  Everything else (after exclusions) becomes Healthy (`0`).
- **`EXCLUDE_LABELS`** — label values dropped before training (e.g. background
  `-1`, unusable class `15`).
- **`SUBSAMPLE`** — cap on the **total number of rows** kept, drawn at random
  across the whole dataset *before* splitting. Its only purpose is to keep
  memory and runtime manageable: fitting PCA on 483 bands over ~1M rows, ×folds,
  is heavy. Because the rows are dropped uniformly at random and *before* the
  train/test split, it introduces **no leakage** — it just reduces the sample
  count everywhere. Set to `None` to use all rows.

### Single-mode config
- **`ARCHITECTURE`** — `'S'`, `'M'` or `'L'` (see `src/model.py`).
- **`N_PC`** — number of principal components fed to the ANN.
- **`SNV`** — enable/disable SNV normalisation.
- **`CV`** — `None` (one 70/15/15 split), `5`, or `10` folds.

### Grid-mode variants
- **`ARCHITECTURE_VARIANTS`**, **`PC_VARIANTS`** — the two axes swept in grid mode.

### Splitting / balancing
- **`GROUP_AWARE`** — when `True`, every `Sample_ID` stays entirely within one
  split (train *or* val *or* test, never two). With only ~25 samples and highly
  correlated pixels, random pixel-level splitting would let near-duplicate
  spectra appear in both train and test, massively inflating scores. Keep `True`
  for honest estimates.
- **`BALANCE`** — when `True`, the **training split** is downsampled so each
  class has equal size. Validation and test keep their natural (imbalanced)
  distribution, which is the realistic setting for evaluation.
- **`BALANCE_CAP`** — an upper bound on the **per-class** size after balancing.
  If the minority class is larger than this cap, both classes are capped to it.
  This bounds training-set size (and therefore training time) even when lots of
  data is available. Example: minority class has 400k rows, `BALANCE_CAP =
  300_000` → training uses 300k + 300k = 600k rows.

### Training hyper-parameters
- **`EPOCHS`** — maximum epochs; early stopping usually halts earlier.
- **`BATCH_SIZE`** — mini-batch size.
- Other knobs are intentionally kept close to where they act:
  - **Early-stopping patience, LR-reduction schedule** → `src/pipeline.py`,
    inside `train_model` (the `EarlyStopping` and `ReduceLROnPlateau` callbacks).
  - **Layer sizes, optimizer, loss, compiled metrics** → `src/model.py`,
    inside `ann_classification`.

### Output
- **`SAVE_RESULTS`** — persist CSVs and figures.
- **`SHOW_PLOTS`** — display figures interactively (forces a headless backend
  when `False`).

---

## Function reference

### `src/pipeline.py`

**`snv(X)`**
Standard Normal Variate. For each spectrum (row) subtracts its mean and divides
by its std, so every spectrum has mean 0 and std 1. Removes multiplicative
intensity differences between pixels. Row-wise → leakage-safe.

**`balance_indices(y, idx, cap=None, seed=42)`**
Given label array `y` and a subset of row indices `idx`, returns a shuffled
subset where every class is downsampled to the size of the minority class
(optionally further capped by `cap`). Applied to training indices only.

**`prepare_fold(X, tr_idx, val_idx, te_idx, max_pc, use_snv, seed=42)`**
Builds the model inputs for one fold:
1. Slices train/val/test matrices from `X`.
2. Applies SNV if `use_snv`.
3. Fits `PCA(n_components=max_pc)` **on the training matrix only**.
4. Returns train/val/test projected onto `max_pc` components.
Because PCA components are ordered and nested, slicing the output `[:, :n_pc]`
is identical to having fitted PCA with `n_pc` components — this lets the grid
try many component counts from a single PCA fit per fold.

**`make_folds(y, groups, cv, group_aware=True, val_frac=0.15, test_frac=0.15, seed=42)`**
Produces a list of `(train_idx, val_idx, test_idx)` tuples.
- `cv=None` → one split (`test_frac` test, `val_frac` validation, rest train).
- `cv=k` → `k` folds; each fold's test set is one held-out outer fold, and its
  validation set is carved from that fold's training portion.
- `group_aware=True` uses `StratifiedGroupKFold` / `GroupShuffleSplit` so no
  `Sample_ID` spans two splits; `False` uses ordinary stratified splitting.
The uniform `(train, val, test)` shape means no-CV behaves like "1 fold", so all
evaluation modes are directly comparable.

**`train_model(X_tr, y_tr, X_val, y_val, architecture, verbose=1, epochs=100, batch_size=512)`**
Builds the ANN (`ann_classification`) and fits it with two callbacks:
`EarlyStopping` (monitors `val_auc`, restores best weights) and
`ReduceLROnPlateau` (halves LR when `val_loss` plateaus). Returns
`(model, history)`.

**`evaluate_fold(model, X_te, y_te)`**
Predicts on the test split and returns `(metrics_dict, y_prob)`. Metrics:
accuracy, AUC, and precision/recall/F1 for the positive (Tumoral) class. AUC is
`NaN` if the test split happens to contain a single class.

**`summarize_metrics(metrics_rows)`**
Takes the list of per-fold metric dicts and returns `(per_fold_df, summary)`
where `summary` has `<metric>_mean` and `<metric>_std` (std uses `ddof=1` when
there is more than one fold, else `0`).

### `src/plots.py`

Every plot function takes `save_path=None` (write a PNG) and `show=False`
(display vs. close). When more than one fold is supplied, curves/ROC aggregate
with a mean line and a ±1 std band; a single fold is drawn raw.

- **`plot_training_curves(histories, …)`** — Loss / Accuracy / AUC vs. epoch,
  train vs. validation. Folds are truncated to the shortest length (early
  stopping yields unequal epoch counts) before averaging.
- **`plot_roc(roc_data, …)`** — ROC per fold plus the mean ROC (interpolated on
  a common FPR grid) with a std band; legend reports mean AUC ± std.
- **`plot_confusion(roc_data, …)`** — row-normalised confusion matrix,
  aggregated (summed counts) across folds.
- **`plot_fold_auc(aucs, …)`** — bar chart of test AUC per fold with the mean as
  a dashed line. Used to spot consistently hard folds/samples.

### `src/model.py`

**`ann_classification(num_components, size)`**
Returns a compiled Keras `Sequential` binary classifier (sigmoid output, Adam,
binary cross-entropy, metrics: accuracy, recall=`sensitivity`, AUC). `size`
selects the architecture:
- `'S'`: 16 → 8 → 1
- `'M'`: 64 → 32 → 1
- `'L'`: 128 → 64 → 16 → 1

---

## Outputs

In grid mode (or single mode with `SAVE_RESULTS = True`), results are written to
`results/<grid|single>/`:

```
results/grid/
├─ all_summaries.csv          # one row per (architecture, n_pc): mean ± std of each metric
├─ all_folds.csv              # one row per fold per configuration
└─ cv{CV}/snv{0|1}/{architecture}/PC{n_pc}/
   ├─ fold_metrics.csv
   ├─ training_curves.png
   ├─ roc.png
   ├─ confusion.png
   └─ fold_auc.png            # only when CV is used (more than one fold)
```
