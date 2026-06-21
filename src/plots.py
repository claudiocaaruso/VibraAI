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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc as auc_score
from sklearn.metrics import confusion_matrix, roc_curve

CLASS_NAMES = ['Healthy', 'Tumoral']
Y_PROB_BIAS = 0.47

def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_curves(histories, title='', save_path=None, show=False):
    """Loss / Accuracy / AUC over epochs, train vs validation.

    `histories` is a list of `history.history` dicts (one per fold). With more
    than one fold the mean ±1 std across folds is drawn; folds are truncated to
    the shortest length (early stopping gives unequal epoch counts).
    """
    min_len = min(len(h['loss']) for h in histories)
    epochs  = np.arange(1, min_len + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key, label in zip(axes, ['loss', 'accuracy', 'auc'],
                              ['Loss (BCE)', 'Accuracy', 'AUC']):
        for prefix, name, color in [('', 'Train', '#1f77b4'),
                                    ('val_', 'Validation', '#ff7f0e')]:
            arr  = np.array([h[f'{prefix}{key}'][:min_len] for h in histories])
            mean = arr.mean(axis=0)
            ax.plot(epochs, mean, color=color, lw=2,
                    ls='--' if prefix else '-', label=name)
            if len(histories) > 1:
                std = arr.std(axis=0)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(label); ax.set_xlabel('Epoch'); ax.set_ylabel(label)
        ax.legend(); ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)
    _finish(fig, save_path, show)


def plot_roc(roc_data, title='', save_path=None, show=False):
    """ROC per fold plus the mean ROC and ±1 std band across folds.

    `roc_data` is a list of (y_true, y_prob) tuples, one per fold.
    """
    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    fig, ax = plt.subplots(figsize=(7, 6))

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
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title or 'ROC'); ax.legend(loc='lower right', fontsize=8)
    _finish(fig, save_path, show)


def plot_confusion(roc_data, title='', save_path=None, show=False):
    """Row-normalised confusion matrix, aggregated (summed) across folds."""
    total = np.zeros((2, 2), dtype=float)
    for y_true, y_prob in roc_data:
        total += confusion_matrix(y_true, (y_prob > Y_PROB_BIAS).astype(int), labels=[0, 1])
    cm = total / total.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title or 'Confusion Matrix')
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
