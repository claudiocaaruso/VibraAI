"""
Visualisations for the Raman classification pipeline.

Every function follows the same convention:
    save_path=None  -> do not write a file
    show=False      -> close the figure instead of displaying it

So grid runs pass save_path=<file>, show=False (nothing pops up), while
single-config runs pass show=True to inspect interactively.

When several folds are supplied the curves/ROC are aggregated with a mean
line and a ±1 std band; with a single fold the raw fold is shown.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.metrics import auc as auc_score
from sklearn.metrics import confusion_matrix, roc_curve

from src.model import count_params

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 13,
        "figure.titlesize": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
    }
)

CLASS_NAMES = ['Tumor stroma', 'Tumor']
Y_PROB_BIAS = 0.49

def _finish(fig, save_path, show, wspace=None):
    fig.tight_layout()
    if wspace is not None:          # applied after tight_layout, which would otherwise undo it
        fig.subplots_adjust(wspace=wspace)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


_CURVE_YLIMS = {'loss': (0.35, 0.75), 'auc': (0.5, 1.0)}


def plot_training_curves(histories, title='', save_path=None, show=False):
    """Loss / AUC over epochs, train vs validation.

    `histories` is a list of `history.history` dicts (one per fold). With
    more than one fold the mean ±1 std across folds is drawn; folds are
    truncated to the shortest length (early stopping gives unequal epoch
    counts). Y-axis ranges are fixed (see `_CURVE_YLIMS`) rather than
    auto-scaled, so curves are visually comparable across different
    runs/figures. No per-panel y-labels — the panel title already names the
    metric.
    """
    min_len = min(len(h['loss']) for h in histories)
    epochs  = np.arange(1, min_len + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, label in zip(axes, ['loss', 'auc'],
                              ['Loss (BCE)', 'AUC']):
        for prefix, name, color in [('', 'Train', '#1f77b4'),
                                    ('val_', 'Validation', '#ff7f0e')]:
            arr  = np.array([h[f'{prefix}{key}'][:min_len] for h in histories])
            mean = arr.mean(axis=0)
            ax.plot(epochs, mean, color=color, lw=2,
                    ls='--' if prefix else '-', label=name)
            if len(histories) > 1:
                std = arr.std(axis=0)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(label, fontsize=18)
        ax.set_xlabel('Epochs', fontsize=18)
        ax.set_ylim(*_CURVE_YLIMS[key])
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=18); ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=18)
    _finish(fig, save_path, show, wspace=0.4)


_ROC_CM_FIGSIZE = (7, 7)   # shared by plot_roc/plot_confusion so the two line up in LaTeX


def plot_roc(roc_data, title='', save_path=None, show=False):
    """ROC per fold plus the mean ROC and ±1 std band across folds.

    `roc_data` is a list of (y_true, y_prob) tuples, one per fold.
    """
    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    fig, ax = plt.subplots(figsize=_ROC_CM_FIGSIZE)

    for i, (y_true, y_prob) in enumerate(roc_data, start=1):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        aucs.append(auc_score(fpr, tpr))
        interp = np.interp(mean_fpr, fpr, tpr); interp[0] = 0.0
        tprs.append(interp)
        if len(roc_data) > 1:
            ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i} (AUC={aucs[-1]:.3f})')
        else:
            ax.plot(fpr, tpr, lw=2, color='#1f77b4', label=f'ROC (AUC={aucs[-1]:.3f})')

    if len(roc_data) > 1:
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        std_tpr  = np.std(tprs, axis=0)
        ax.plot(mean_fpr, mean_tpr, color='b', lw=2.5,
                label=f'Mean (AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f})')
        ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),
                        np.minimum(mean_tpr + std_tpr, 1), color='b', alpha=0.2,
                        label='± 1 std')

    ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=18)
    ax.set_ylabel('True Positive Rate', fontsize=18)
    ax.set_title(title or 'ROC', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(loc='lower right', fontsize=18)
    ax.set_aspect('equal', adjustable='box')
    _finish(fig, save_path, show)


def plot_confusion(roc_data, title='', save_path=None, show=False):
    """Confusion matrix, row-normalised per fold then averaged across folds.

    Matches the macro-averaged recall/tnr reported in the training summary
    (mean of per-fold metrics), rather than pooling raw counts across folds.
    """
    cms = [confusion_matrix(y_true, (y_prob > Y_PROB_BIAS).astype(int), labels=[0, 1],
                            normalize='true')
           for y_true, y_prob in roc_data]
    cm = np.mean(cms, axis=0)

    fig, ax = plt.subplots(figsize=_ROC_CM_FIGSIZE)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar=False, square=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={'size': 18})
    ax.set_xlabel('Predicted', fontsize=18)
    ax.set_ylabel('True', fontsize=18)
    ax.set_title(title or 'Confusion Matrix', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    _finish(fig, save_path, show)


def plot_fold_auc(aucs, title='', save_path=None, show=False):
    """Bar chart of test AUC per fold with the mean drawn as a dashed line."""
    aucs  = np.asarray(aucs, dtype=float)
    folds = np.arange(1, len(aucs) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(folds, aucs, color='#4c72b0')
    mean = np.nanmean(aucs)
    ax.axhline(mean, color='r', ls='--', label=f'Mean = {mean:.3f}')
    ax.set_xlabel('Fold'); ax.set_ylabel('AUC'); ax.set_xticks(folds)
    ax.set_ylim(0, 1); ax.set_title(title or 'AUC per fold'); ax.legend()
    _finish(fig, save_path, show)


def plot_sample_f1(sample_ids, f1_scores, save_path=None, show=False):
    """Horizontal bar chart of F1 per Sample_ID, sorted ascending, mean as a dashed line.

    Each sample is held out exactly once under 5-fold group-aware CV, so this
    is a single F1 per sample (a post-hoc breakdown of each fold's test
    predictions by Sample_ID), not a mean across folds. F1 (with
    zero_division=0) is always defined, unlike AUC, which needs both classes
    present in a sample's pixels — many samples here are single-class.
    """
    f1_scores = np.asarray(f1_scores, dtype=float)
    order = np.argsort(f1_scores)
    sample_ids = np.asarray(sample_ids)[order]
    f1_scores = f1_scores[order]
    y_pos = np.arange(len(f1_scores))

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(f1_scores) + 2))
    ax.barh(y_pos, f1_scores, color='#4c72b0')
    mean = np.nanmean(f1_scores)
    ax.axvline(mean, color='r', ls='--', label=f'Mean = {mean:.3f}')
    ax.set_yticks(y_pos); ax.set_yticklabels(sample_ids, fontsize=14)
    ax.set_xlabel('F1', fontsize=16); ax.set_xlim(0, 1)
    ax.tick_params(axis='x', labelsize=14)
    ax.legend(loc='lower right', fontsize=16)
    _finish(fig, save_path, show)


# ── grid-search comparison plots ────────────────────────────────────────────
# These read the summary/fold tables written by scripts/train.py (MODE='grid')
# rather than in-memory training state — see scripts/analyze_grid.py.

def plot_grid_heatmap(summary_df, metric='auc_mean', title='', save_path=None, show=False):
    """Heatmap of a summary metric across architecture x n_pc combinations."""
    pivot = summary_df.pivot(index='architecture', columns='n_pc', values=metric)
    fig, ax = plt.subplots(figsize=(1.2 * len(pivot.columns) + 2, 3))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    ax.set_xlabel('PCA components'); ax.set_ylabel('Architecture')
    ax.set_title(title or metric)
    _finish(fig, save_path, show)


def plot_grid_lines(summary_df, metric='auc', title='', save_path=None, show=False):
    """Mean ± std of a metric vs n_pc, one line per architecture."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for architecture, group in summary_df.groupby('architecture'):
        group = group.sort_values('n_pc')
        n_pc  = group['n_pc'].to_numpy()
        mean  = group[f'{metric}_mean'].to_numpy()
        std   = group[f'{metric}_std'].to_numpy()
        ax.plot(n_pc, mean, marker='o', label=architecture)
        ax.fill_between(n_pc, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel('PCA components'); ax.set_ylabel(metric.upper())
    ax.set_title(title or f'{metric.upper()} vs PCA components')
    ax.legend(title='Architecture'); ax.grid(True, alpha=0.3)
    _finish(fig, save_path, show)


def plot_grid_leaderboard(summary_df, metric='auc', top_n=10, title='', save_path=None, show=False):
    """Horizontal bar chart of the top-N combinations ranked by a summary metric."""
    top = summary_df.nlargest(top_n, f'{metric}_mean')
    labels = [f'{row.architecture}_PC{row.n_pc}' for row in top.itertuples()]
    y_pos  = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 2))
    ax.barh(y_pos, top[f'{metric}_mean'], xerr=top[f'{metric}_std'], color='#4c72b0')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper())
    ax.set_title(title or f'Top {top_n} by {metric.upper()}')
    _finish(fig, save_path, show)


def plot_grid_boxplot(folds_df, combos, metric='auc', title='', save_path=None, show=False):
    """Per-fold distribution of a metric for a chosen list of (architecture, n_pc) combos."""
    mask = np.zeros(len(folds_df), dtype=bool)
    for architecture, n_pc in combos:
        mask |= (folds_df['architecture'] == architecture) & (folds_df['n_pc'] == n_pc)
    subset = folds_df[mask].copy()
    subset['combo'] = subset['architecture'] + '_PC' + subset['n_pc'].astype(str)

    fig, ax = plt.subplots(figsize=(1.5 * len(combos) + 2, 5))
    sns.boxplot(data=subset, x='combo', y=metric, ax=ax, color='#4c72b0')
    sns.stripplot(data=subset, x='combo', y=metric, ax=ax, color='black', alpha=0.6)
    ax.set_xlabel('Combination'); ax.set_ylabel(metric.upper())
    ax.set_title(title or f'Per-fold {metric.upper()} — top combinations')
    _finish(fig, save_path, show)


def plot_grid_complexity(summary_df, metric='auc', title='', save_path=None, show=False):
    """Trainable-parameter count vs a summary metric, colored by architecture."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for architecture, group in summary_df.groupby('architecture'):
        params = [count_params(architecture, n) for n in group['n_pc']]
        ax.errorbar(params, group[f'{metric}_mean'], yerr=group[f'{metric}_std'],
                    fmt='o', label=architecture)
    ax.set_xlabel('Trainable parameters'); ax.set_ylabel(metric.upper())
    ax.set_title(title or f'{metric.upper()} vs model complexity')
    ax.legend(title='Architecture'); ax.grid(True, alpha=0.3)
    _finish(fig, save_path, show)


HEALTHY_TUMOR_CMAP = LinearSegmentedColormap.from_list('healthy_tumor', ['forestgreen', 'darkred'])


def plot_prediction_map(x, y, y_true, y_prob, threshold, save_path=None, show=False):
    """True labels, predicted probability, and prediction errors for one Raman map.

    Three panels sharing spatial (x, y) axes: true tumoral/healthy labels,
    the predicted probability (already smoothed by the caller, if at all) on
    the same green-to-red scale as the true labels for a direct visual
    comparison, and a plain correct/error map from thresholding that
    probability (no FP/FN distinction — just where predictions are wrong).
    """
    def grid(values):
        return (pd.DataFrame({'x': x, 'y': y, 'v': values})
                  .pivot_table(index='y', columns='x', values='v')
                  .sort_index().sort_index(axis=1))

    y_pred = (np.asarray(y_prob) > threshold).astype(int)
    is_error = (y_pred != np.asarray(y_true)).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(grid(y_true), cmap=HEALTHY_TUMOR_CMAP, vmin=0, vmax=1, origin='lower')
    axes[0].set_title('True labels', fontsize=18)
    cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[0, 1], fraction=0.046)
    cbar0.ax.set_yticklabels(['Healthy', 'Tumor'], fontsize=18)

    im = axes[1].imshow(grid(y_prob), cmap=HEALTHY_TUMOR_CMAP, vmin=0, vmax=1, origin='lower')
    axes[1].set_title('Predicted probability', fontsize=18)
    cbar1 = fig.colorbar(im, ax=axes[1], fraction=0.046)
    cbar1.ax.tick_params(labelsize=18)

    im2 = axes[2].imshow(grid(is_error), cmap=ListedColormap(['white', 'red']),
                         vmin=0, vmax=1, origin='lower')
    cbar2 = fig.colorbar(im2, ax=axes[2], ticks=[0, 1], fraction=0.046)
    cbar2.ax.set_yticklabels(['Correct', 'Error'], fontsize=18)
    axes[2].set_title('Prediction errors', fontsize=18)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=16)

    _finish(fig, save_path, show)
