import os

import numpy as np
import pandas as pd


def create_run_dirs(base_save_dir):
    dirs = {
        "metrics": os.path.join(base_save_dir, "metric_images"),
        "tables": os.path.join(base_save_dir, "tables"),
        "params": os.path.join(base_save_dir, "parameters"),
        "maps": os.path.join(base_save_dir, "prediction_maps"),
    }
    for directory in [dirs["metrics"], dirs["tables"], dirs["params"]]:
        os.makedirs(directory, exist_ok=True)
    return dirs


def save_pca_artifacts(pca, fold_number, params_dir, tables_dir):
    np.save(os.path.join(params_dir, f"pca_components_fold_{fold_number}.npy"), pca.components_)
    pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(os.path.join(tables_dir, f"pca_explained_variance_fold_{fold_number}.csv"), index=False)


def save_run_summary(
    run_summary_path,
    base_save_dir,
    config,
    df_raw_shape,
    df_filtered_shape,
    n_bands,
    n_groups,
    filtered_class_counts,
    numeric_cols,
    summary_mean,
    summary_std,
    cm_total,
    cm_normalized,
    sample_ranking_df,
):
    from datetime import datetime

    with open(run_summary_path, "w", encoding="utf-8") as f:
        f.write("Fold-wise SNV + PCA + ANN Binary Classification\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Run directory: {base_save_dir}\n\n")

        f.write("Configuration\n")
        f.write("-" * 70 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Dataset\n")
        f.write("-" * 70 + "\n")
        f.write(f"Raw dataset shape: {df_raw_shape}\n")
        f.write(f"Filtered dataset shape: {df_filtered_shape}\n")
        f.write(f"Spectral bands: {n_bands}\n")
        f.write(f"Unique groups: {n_groups}\n")
        f.write(f"Filtered class counts: {filtered_class_counts}\n")
        f.write("Feature mode: center spectrum only\n")
        f.write("Balancing scope: training partition only, separately in each fold\n")
        f.write("Validation and test rows: complete and not balanced\n\n")

        f.write("Mean Test Metrics Across Folds\n")
        f.write("-" * 70 + "\n")
        for name in numeric_cols:
            f.write(f"{name}: {summary_mean[name]:.6f} +/- {summary_std[name]:.6f}\n")
        f.write("\n")

        f.write("Total Confusion Matrix Across Folds\n")
        f.write("-" * 70 + "\n")
        f.write(str(cm_total))
        f.write("\n\n")

        f.write("Normalized Total Confusion Matrix Across Folds\n")
        f.write("-" * 70 + "\n")
        f.write(str(cm_normalized))
        f.write("\n\n")

        f.write("Output Tables\n")
        f.write("-" * 70 + "\n")
        f.write("tables/fold_results.csv\n")
        f.write("tables/sample_results.csv\n")
        if sample_ranking_df is not None:
            f.write("tables/sample_difficulty_ranking.csv\n")
        f.write("tables/fold_label_distribution.csv\n")
        f.write("tables/history_all_folds.csv\n")
        f.write("tables/history_mean.csv\n")
        f.write("tables/history_std.csv\n")
        f.write("tables/comparative_results.csv\n")
