import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_history(history_frame, metrics_dir, filename, title_prefix, show_plots):
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
    plt.savefig(os.path.join(metrics_dir, filename), dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_mean_history(mean_history, std_history, metrics_dir, show_plots):
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
    plt.savefig(os.path.join(metrics_dir, "METRICS_MEAN_5_FOLD.png"), dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrices(cm_total, class_names, metrics_dir, show_plots):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Total Confusion Matrix Across Folds")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "CONFUSION_MATRIX_TOTAL_5_FOLD.png"), dpi=300, bbox_inches="tight")
    if show_plots:
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
    if show_plots:
        plt.show()
    else:
        plt.close()

    return cm_normalized


def plot_sample_difficulty_ranking(
    sample_results_df,
    metrics_dir,
    tables_dir,
    primary_metric,
    secondary_metric,
    show_plots,
):
    required_cols = ["sample_id", "rows", primary_metric, secondary_metric, "confusion_fp", "confusion_fn"]
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
        .sort_values([primary_metric, secondary_metric], ascending=True)
        .reset_index(drop=True)
    )
    ranking_df.insert(0, "difficulty_rank", np.arange(1, len(ranking_df) + 1))
    ranking_df.to_csv(os.path.join(tables_dir, "sample_difficulty_ranking.csv"), index=False)

    plot_df = ranking_df.copy()
    plot_df["sample_id"] = plot_df["sample_id"].astype(str)
    fig_height = max(6, min(24, 0.42 * len(plot_df) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y_pos = np.arange(len(plot_df))
    bar_height = 0.38

    ax.barh(y_pos - bar_height / 2, plot_df[primary_metric], height=bar_height, label=primary_metric, color="#2a9d8f")
    ax.barh(y_pos + bar_height / 2, plot_df[secondary_metric], height=bar_height, label=secondary_metric, color="#e76f51")

    for idx, row in plot_df.iterrows():
        ax.text(1.01, idx, f"FP {int(row['confusion_fp'])} | FN {int(row['confusion_fn'])}",
                va="center", fontsize=8, transform=ax.get_yaxis_transform())

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["sample_id"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Metric value")
    ax.set_ylabel("Sample_ID")
    ax.set_title(f"Sample Difficulty Ranking (hardest to easiest by {primary_metric})")
    ax.grid(True, axis="x", alpha=0.35)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "SAMPLE_DIFFICULTY_RANKING.png"), dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return ranking_df


def save_prediction_maps_if_enabled(df_test, y_probs, y_pred, fold_number, maps_dir, save_prediction_maps, show_plots):
    if not save_prediction_maps:
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
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
