import gc
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
for path in [SCRIPT_DIR, REPO_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from data_giulia import (
    class_counts,
    filter_labels,
    first_existing_path,
    get_band_columns,
    save_config,
    set_seed,
)
from evaluation_giulia import (
    compute_sample_results,
    evaluate_predictions,
    history_to_frame,
    label_distribution_row,
    summarize_histories,
    summarize_fold_metrics,
)
from io_giulia import create_run_dirs, save_pca_artifacts, save_run_summary
from pipeline_giulia import balance_indices, make_folds, prepare_fold, train_model
from plots_giulia import (
    plot_confusion_matrices,
    plot_history,
    plot_mean_history,
    plot_sample_difficulty_ranking,
    save_prediction_maps_if_enabled,
)


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = first_existing_path(
    [
        os.path.join(REPO_ROOT, "datasets", "spectral_dataset_clean.parquet"),
        os.path.join(REPO_ROOT, "datasets", "spectral_dataset.parquet"),
        os.path.join(WORKSPACE_ROOT, "PCA", "spectral_dataset_clean.parquet"),
    ]
)
# Outputs are saved outside the cloned repository.
OUTPUT_BASE_DIR = os.path.join(WORKSPACE_ROOT, "train_results")

#   "single" -> one complete run with detailed outputs, maps and models
#   "grid"   -> many lightweight runs, only comparison tables
MODE = "single"

EXPERIMENT_NAME = "2-20_VS_all_other__SMOOTHING"

CLASS_1_NAME = "class_1"
CLASS_0_NAME = "class_0"
CLASS_1_LABELS = [2, 20]
CLASS_0_LABELS = "all_other"

USE_SNV = True
SNV_STD_EPS = 1e-8
MAP_GROUP_COLUMNS = ["Sample_ID", "Map_ID"]

BALANCE_CLASSES = True
N_FOLDS = 5
GROUP_COLUMN = "Sample_ID"
VALIDATION_SIZE_WITHIN_TRAIN = 0.1875

N_COMPONENTS = 50
PCA_RANDOM_STATE = 42

MODEL_SIZE = "S"
EPOCHS = 100
BATCH_SIZE = 8192
CLASSIFICATION_THRESHOLD = 0.49

# Grid mode variants. These are ignored when MODE = "single".
MODEL_SIZE_VARIANTS = ['S', 'M', 'L']
N_COMPONENTS_VARIANTS = [50]
BATCH_SIZE_VARIANTS = [8192]
CLASSIFICATION_THRESHOLD_VARIANTS = [0.49]
SAVE_GRID_HISTORY_PLOTS = True
SHOW_GRID_HISTORY_PLOTS = False

SMOOTH_PREDICTION_PROBS = True
PREDICTION_SMOOTHING_METHOD = "mean"
PREDICTION_SMOOTHING_KERNEL_SIZE = 5

RANDOM_STATE = 43

SAVE_PCA_ARTIFACTS = False
SAVE_PREDICTION_MAPS = True

SAMPLE_RANKING_PRIMARY_METRIC = "class_1_f1"
SAMPLE_RANKING_SECONDARY_METRIC = "class_1_recall"

SHOW_PLOTS = True


def sanitize_experiment_name(experiment_name):
    """Return a filesystem-friendly experiment name for output folders."""
    cleaned = "".join(char if char.isalnum() or char in ["-", "_"] else "_" for char in experiment_name.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned


def build_config(class_0_labels, class_1_labels, filtered_class_counts):
    return {
        "DATA_PATH": DATA_PATH,
        "OUTPUT_BASE_DIR": OUTPUT_BASE_DIR,
        "MODE": MODE,
        "MODE_DESCRIPTION": (
            "single: complete run with tables, plots, maps and models; "
            "grid: comparison table, optionally with training history plots"
        ),
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "CLASS_0_NAME": CLASS_0_NAME,
        "CLASS_1_NAME": CLASS_1_NAME,
        "CLASS_0_LABELS": class_0_labels,
        "CLASS_1_LABELS": class_1_labels,
        "CLASS_0_LABELS_CONFIG": CLASS_0_LABELS,
        "CLASS_1_LABELS_CONFIG": CLASS_1_LABELS,
        "USE_SNV": USE_SNV,
        "NORMALIZATION": "SNV per spectrum" if USE_SNV else "none",
        "FEATURE_MODE": "center_spectrum_only",
        "MAP_GROUP_COLUMNS": MAP_GROUP_COLUMNS,
        "BALANCE_CLASSES": BALANCE_CLASSES,
        "BALANCING_SCOPE": "training_only",
        "VALIDATION_AND_TEST_BALANCED": False,
        "filtered_class_counts": filtered_class_counts,
        "N_FOLDS": N_FOLDS,
        "GROUP_COLUMN": GROUP_COLUMN,
        "VALIDATION_SIZE_WITHIN_TRAIN": VALIDATION_SIZE_WITHIN_TRAIN,
        "N_COMPONENTS": N_COMPONENTS,
        "MODEL_SIZE": MODEL_SIZE,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "CLASSIFICATION_THRESHOLD": CLASSIFICATION_THRESHOLD,
        "MODEL_SIZE_VARIANTS": MODEL_SIZE_VARIANTS,
        "N_COMPONENTS_VARIANTS": N_COMPONENTS_VARIANTS,
        "BATCH_SIZE_VARIANTS": BATCH_SIZE_VARIANTS,
        "CLASSIFICATION_THRESHOLD_VARIANTS": CLASSIFICATION_THRESHOLD_VARIANTS,
        "SAVE_GRID_HISTORY_PLOTS": SAVE_GRID_HISTORY_PLOTS,
        "SHOW_GRID_HISTORY_PLOTS": SHOW_GRID_HISTORY_PLOTS,
        "SMOOTH_PREDICTION_PROBS": SMOOTH_PREDICTION_PROBS,
        "SMOOTHING_SCOPE": "post_processing_only_after_prediction",
        "PREDICTION_SMOOTHING_METHOD": PREDICTION_SMOOTHING_METHOD,
        "PREDICTION_SMOOTHING_KERNEL_SIZE": PREDICTION_SMOOTHING_KERNEL_SIZE,
        "RANDOM_STATE": RANDOM_STATE,
        "PCA_RANDOM_STATE": PCA_RANDOM_STATE,
        "SAVE_PCA_ARTIFACTS": SAVE_PCA_ARTIFACTS,
        "SAVE_PREDICTION_MAPS": SAVE_PREDICTION_MAPS,
        "SAMPLE_RANKING_PRIMARY_METRIC": SAMPLE_RANKING_PRIMARY_METRIC,
        "SAMPLE_RANKING_SECONDARY_METRIC": SAMPLE_RANKING_SECONDARY_METRIC,
        "SHOW_PLOTS": SHOW_PLOTS,
    }


def prefix_train_metrics(metrics):
    """Rename evaluation metrics computed on the training split."""
    prefixed = {}
    for key, value in metrics.items():
        if key.startswith("test_"):
            prefixed[f"train_{key.removeprefix('test_')}"] = value
        else:
            prefixed[f"train_{key}"] = value
    return prefixed


def run_grid():
    """Run a lightweight grid search and save only comparison tables."""
    set_seed(RANDOM_STATE)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = sanitize_experiment_name(EXPERIMENT_NAME)
    run_name = f"grid_{run_timestamp}"
    if experiment_name:
        run_name = f"{run_name}_{experiment_name}"
    base_save_dir = os.path.join(OUTPUT_BASE_DIR, run_name)
    tables_dir = os.path.join(base_save_dir, "tables")
    grid_history_dir = os.path.join(base_save_dir, "grid_history")
    grid_metrics_dir = os.path.join(base_save_dir, "grid_metric_images")
    os.makedirs(tables_dir, exist_ok=True)
    if SAVE_GRID_HISTORY_PLOTS:
        os.makedirs(grid_history_dir, exist_ok=True)
        os.makedirs(grid_metrics_dir, exist_ok=True)

    print("Loading raw spectral dataset...")
    df_raw = pd.read_parquet(DATA_PATH)
    band_cols = get_band_columns(df_raw)

    required_cols = ["Label", GROUP_COLUMN] + MAP_GROUP_COLUMNS
    missing_required_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_required_cols:
        raise ValueError(f"Missing required columns: {missing_required_cols}")

    df_filtered, class_0_labels, class_1_labels = filter_labels(df_raw, CLASS_1_LABELS, CLASS_0_LABELS)
    class_names = [CLASS_0_NAME, CLASS_1_NAME]
    filtered_class_counts = df_filtered["Binary_Label"].value_counts().sort_index().to_dict()
    config = build_config(class_0_labels, class_1_labels, filtered_class_counts)
    save_config(config, os.path.join(base_save_dir, "config.json"))

    print("GRID MODE")
    print(f"Filtered dataset shape: {df_filtered.shape}")
    print(f"Class counts: {filtered_class_counts}")
    print(f"Results table directory: {tables_dir}")
    print(f"Save grid history plots: {SAVE_GRID_HISTORY_PLOTS}")

    y_all = df_filtered["Binary_Label"].to_numpy(dtype=int)
    groups_all = df_filtered[GROUP_COLUMN].to_numpy()
    X_all = df_filtered[band_cols].to_numpy(dtype=np.float32)
    folds = make_folds(y_all, groups_all, N_FOLDS, VALIDATION_SIZE_WITHIN_TRAIN, seed=RANDOM_STATE)

    fold_rows = []
    summary_rows = []
    configs = list(product(MODEL_SIZE_VARIANTS, N_COMPONENTS_VARIANTS, BATCH_SIZE_VARIANTS, CLASSIFICATION_THRESHOLD_VARIANTS))

    for config_number, (model_size, n_components, batch_size, threshold) in enumerate(configs, start=1):
        print("\n" + "=" * 70)
        print(
            f"GRID CONFIG {config_number}/{len(configs)} - "
            f"model={model_size}, pca={n_components}, batch={batch_size}, threshold={threshold}"
        )
        print("=" * 70)

        config_fold_rows = []
        history_frames = []
        config_id = f"config_{config_number:03d}_{model_size}_pca{n_components}_batch{batch_size}_thr{str(threshold).replace('.', 'p')}"
        if SAVE_GRID_HISTORY_PLOTS:
            config_history_dir = os.path.join(grid_history_dir, config_id)
            config_metrics_dir = os.path.join(grid_metrics_dir, config_id)
            os.makedirs(config_history_dir, exist_ok=True)
            os.makedirs(config_metrics_dir, exist_ok=True)

        for fold_number, (train_idx_unbalanced, val_idx, test_idx) in enumerate(folds, start=1):
            fold_seed = RANDOM_STATE + fold_number
            set_seed(fold_seed)
            tf.keras.backend.clear_session()

            train_idx = balance_indices(y_all, train_idx_unbalanced, balance_classes=BALANCE_CLASSES, seed=fold_seed)
            y_train = y_all[train_idx]
            y_val = y_all[val_idx]
            y_test = y_all[test_idx]
            df_train = df_filtered.iloc[train_idx].copy()
            df_test = df_filtered.iloc[test_idx].copy()

            X_train, X_val, X_test, _ = prepare_fold(
                X_all,
                train_idx,
                val_idx,
                test_idx,
                n_components=n_components,
                use_snv=USE_SNV,
                pca_seed=PCA_RANDOM_STATE,
                snv_eps=SNV_STD_EPS,
            )

            model, history = train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                model_size,
                EPOCHS,
                batch_size,
                verbose=0,
            )

            if SAVE_GRID_HISTORY_PLOTS:
                history_frame = history_to_frame(history, fold_number)
                history_output_frame = history_frame.copy()
                history_output_frame.insert(1, "config_id", config_id)
                history_output_frame.insert(2, "model_size", model_size)
                history_output_frame.insert(3, "n_components", n_components)
                history_output_frame.insert(4, "batch_size", batch_size)
                history_output_frame.insert(5, "classification_threshold", threshold)
                history_output_frame.to_csv(
                    os.path.join(config_history_dir, f"history_fold_{fold_number}.csv"),
                    index=False,
                )
                history_frames.append(history_frame)
                plot_history(
                    history_frame,
                    config_metrics_dir,
                    f"METRICS_FOLD_{fold_number}.png",
                    f"{config_id} - Fold {fold_number}",
                    SHOW_GRID_HISTORY_PLOTS,
                )

            train_raw_probs = model.predict(X_train, verbose=0).reshape(-1)
            train_prediction_metrics, _, _, _ = evaluate_predictions(
                df_train,
                y_train,
                train_raw_probs,
                threshold,
                class_names,
                MAP_GROUP_COLUMNS,
                SMOOTH_PREDICTION_PROBS,
                PREDICTION_SMOOTHING_METHOD,
                PREDICTION_SMOOTHING_KERNEL_SIZE,
            )
            y_raw_probs = model.predict(X_test, verbose=0).reshape(-1)
            prediction_metrics, _, _, _ = evaluate_predictions(
                df_test,
                y_test,
                y_raw_probs,
                threshold,
                class_names,
                MAP_GROUP_COLUMNS,
                SMOOTH_PREDICTION_PROBS,
                PREDICTION_SMOOTHING_METHOD,
                PREDICTION_SMOOTHING_KERNEL_SIZE,
            )

            row = {
                "config_id": config_id,
                "model_size": model_size,
                "n_components": n_components,
                "batch_size": batch_size,
                "classification_threshold": threshold,
                "fold": fold_number,
                "epochs_ran": len(history.history["loss"]),
                "train_rows": len(y_train),
                "validation_rows": len(y_val),
                "test_rows": len(y_test),
                "train_class_1_rate": y_train.mean(),
                "validation_class_1_rate": y_val.mean(),
                "test_class_1_rate": y_test.mean(),
            }
            row.update(prefix_train_metrics(train_prediction_metrics))
            row.update(prediction_metrics)
            config_fold_rows.append(row)
            fold_rows.append(row)
            print(
                f"  fold {fold_number}/{N_FOLDS}: "
                f"train_auc={row['train_auc']:.4f}, test_auc={row['test_auc']:.4f}, "
                f"train_class_1_f1={row['train_class_1_f1']:.4f}, test_class_1_f1={row['class_1_f1']:.4f}"
            )

            del model, X_train, X_val, X_test
            gc.collect()

        if SAVE_GRID_HISTORY_PLOTS and history_frames:
            history_all, history_mean, history_std = summarize_histories(history_frames)
            history_all.to_csv(os.path.join(config_history_dir, "history_all_folds.csv"), index=False)
            history_mean.to_csv(os.path.join(config_history_dir, "history_mean.csv"), index=False)
            history_std.to_csv(os.path.join(config_history_dir, "history_std.csv"), index=False)
            plot_mean_history(history_mean, history_std, config_metrics_dir, SHOW_GRID_HISTORY_PLOTS)

        config_df = pd.DataFrame(config_fold_rows)
        _, means, stds = summarize_fold_metrics(config_df, exclude_cols=["fold"])
        summary_rows.append(
            {
                "config_id": config_id,
                "model_size": model_size,
                "n_components": n_components,
                "batch_size": batch_size,
                "classification_threshold": threshold,
                "n_folds": N_FOLDS,
                "use_snv": USE_SNV,
                "balance_classes": BALANCE_CLASSES,
                "smooth_prediction_probs": SMOOTH_PREDICTION_PROBS,
                "prediction_smoothing_method": PREDICTION_SMOOTHING_METHOD if SMOOTH_PREDICTION_PROBS else "none",
                "prediction_smoothing_kernel_size": PREDICTION_SMOOTHING_KERNEL_SIZE if SMOOTH_PREDICTION_PROBS else 0,
                "mean_test_auc": means.get("test_auc"),
                "std_test_auc": stds.get("test_auc"),
                "mean_test_accuracy": means.get("test_accuracy"),
                "std_test_accuracy": stds.get("test_accuracy"),
                "mean_train_auc": means.get("train_auc"),
                "std_train_auc": stds.get("train_auc"),
                "mean_train_accuracy": means.get("train_accuracy"),
                "std_train_accuracy": stds.get("train_accuracy"),
                "mean_train_class_1_precision": means.get("train_class_1_precision"),
                "std_train_class_1_precision": stds.get("train_class_1_precision"),
                "mean_train_class_1_recall": means.get("train_class_1_recall"),
                "std_train_class_1_recall": stds.get("train_class_1_recall"),
                "mean_train_class_1_f1": means.get("train_class_1_f1"),
                "std_train_class_1_f1": stds.get("train_class_1_f1"),
                "mean_class_1_precision": means.get("class_1_precision"),
                "std_class_1_precision": stds.get("class_1_precision"),
                "mean_class_1_recall": means.get("class_1_recall"),
                "std_class_1_recall": stds.get("class_1_recall"),
                "mean_class_1_f1": means.get("class_1_f1"),
                "std_class_1_f1": stds.get("class_1_f1"),
                "mean_epochs_ran": means.get("epochs_ran"),
                "std_epochs_ran": stds.get("epochs_ran"),
            }
        )

    grid_folds_df = pd.DataFrame(fold_rows)
    grid_summary_df = pd.DataFrame(summary_rows).sort_values("mean_class_1_f1", ascending=False)
    grid_folds_df.to_csv(os.path.join(tables_dir, "grid_fold_results.csv"), index=False)
    grid_summary_df.to_csv(os.path.join(tables_dir, "grid_comparison.csv"), index=False)

    print("\nTop grid configurations by mean_class_1_f1:")
    print(grid_summary_df.head(10).to_string(index=False))
    print(f"\nGrid results saved at: {base_save_dir}")


def main():
    if MODE == "grid":
        run_grid()
        return

    set_seed(RANDOM_STATE)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = sanitize_experiment_name(EXPERIMENT_NAME)
    run_name = f"run_{run_timestamp}"
    if experiment_name:
        run_name = f"{run_name}_{experiment_name}"
    run_name = f"{run_name}_{MODEL_SIZE}_pca{N_COMPONENTS}_snv-{int(USE_SNV)}_{N_FOLDS}fold"
    base_save_dir = os.path.join(OUTPUT_BASE_DIR, run_name)
    dirs = create_run_dirs(base_save_dir)

    print("Loading raw spectral dataset...")
    df_raw = pd.read_parquet(DATA_PATH)
    band_cols = get_band_columns(df_raw)

    required_cols = ["Label", GROUP_COLUMN] + MAP_GROUP_COLUMNS
    missing_required_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_required_cols:
        raise ValueError(f"Missing required columns: {missing_required_cols}")

    print(f"Raw dataset shape: {df_raw.shape}")
    print(f"Spectral bands: {len(band_cols)}")
    print(f"SNV normalization: {USE_SNV}")

    df_filtered, class_0_labels, class_1_labels = filter_labels(df_raw, CLASS_1_LABELS, CLASS_0_LABELS)
    class_names = [CLASS_0_NAME, CLASS_1_NAME]
    filtered_class_counts = df_filtered["Binary_Label"].value_counts().sort_index().to_dict()

    print("Binary label setup:")
    print(f"  {CLASS_0_NAME} labels: {class_0_labels}")
    print(f"  {CLASS_1_NAME} labels: {class_1_labels}")
    print(f"Filtered class counts: {filtered_class_counts}")
    print("Class balancing: training partition only")
    print(f"Filtered dataset shape: {df_filtered.shape}")
    print(f"Sample_ID groups: {df_filtered[GROUP_COLUMN].nunique()}")

    if df_filtered[GROUP_COLUMN].nunique() < N_FOLDS:
        raise ValueError(
            f"N_FOLDS={N_FOLDS}, but only {df_filtered[GROUP_COLUMN].nunique()} unique groups are available."
        )

    config = build_config(class_0_labels, class_1_labels, filtered_class_counts)
    save_config(config, os.path.join(base_save_dir, "config.json"))

    y_all = df_filtered["Binary_Label"].to_numpy(dtype=int)
    groups_all = df_filtered[GROUP_COLUMN].to_numpy()
    X_all = df_filtered[band_cols].to_numpy(dtype=np.float32)
    folds = make_folds(y_all, groups_all, N_FOLDS, VALIDATION_SIZE_WITHIN_TRAIN, seed=RANDOM_STATE)

    fold_results = []
    sample_results = []
    fold_label_distribution = []
    history_frames = []
    cm_total = np.zeros((2, 2), dtype=int)

    for fold_number, (train_idx_unbalanced, val_idx, test_idx) in enumerate(folds, start=1):
        print("\n" + "=" * 70)
        print(f"FOLD {fold_number}/{N_FOLDS}")
        print("=" * 70)

        fold_seed = RANDOM_STATE + fold_number
        set_seed(fold_seed)
        tf.keras.backend.clear_session()

        train_idx = balance_indices(y_all, train_idx_unbalanced, balance_classes=BALANCE_CLASSES, seed=fold_seed)

        df_train_unbalanced = df_filtered.iloc[train_idx_unbalanced].copy()
        df_train = df_filtered.iloc[train_idx].copy()
        df_test = df_filtered.iloc[test_idx].copy()

        y_train_unbalanced = y_all[train_idx_unbalanced]
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        y_test = y_all[test_idx]
        groups_test = groups_all[test_idx]
        train_counts_before = class_counts(y_train_unbalanced)
        train_counts_after = class_counts(y_train)

        train_groups = set(groups_all[train_idx_unbalanced].astype(str))
        val_groups = set(groups_all[val_idx].astype(str))
        test_groups = set(groups_all[test_idx].astype(str))

        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise ValueError(f"Sample_ID leakage detected in fold {fold_number}.")

        print(
            "Group counts - "
            f"train: {len(train_groups)}, "
            f"validation: {len(val_groups)}, "
            f"test: {len(test_groups)}"
        )
        print(f"Training class counts - before balancing: {train_counts_before}, after balancing: {train_counts_after}")
        print(
            "Class 1 rate - "
            f"train: {y_train.mean():.3f}, "
            f"validation: {y_val.mean():.3f}, "
            f"test: {y_test.mean():.3f}"
        )

        fold_label_distribution.extend(
            [
                label_distribution_row(fold_number, "train_before_balancing", y_train_unbalanced, groups_all[train_idx_unbalanced]),
                label_distribution_row(fold_number, "train", y_train, groups_all[train_idx]),
                label_distribution_row(fold_number, "validation", y_val, groups_all[val_idx]),
                label_distribution_row(fold_number, "test", y_test, groups_test),
            ]
        )

        print("Fitting PCA on training rows only...")
        X_train, X_val, X_test, pca = prepare_fold(
            X_all,
            train_idx,
            val_idx,
            test_idx,
            n_components=N_COMPONENTS,
            use_snv=USE_SNV,
            pca_seed=PCA_RANDOM_STATE,
            snv_eps=SNV_STD_EPS,
        )

        if SAVE_PCA_ARTIFACTS:
            save_pca_artifacts(pca, fold_number, dirs["params"], dirs["tables"])

        model, history = train_model(X_train, y_train, X_val, y_val, MODEL_SIZE, EPOCHS, BATCH_SIZE, verbose=1)
        if fold_number == 1:
            model.summary()

        history_frame = history_to_frame(history, fold_number)
        history_frame.to_csv(os.path.join(dirs["tables"], f"history_fold_{fold_number}.csv"), index=False)
        history_frames.append(history_frame)
        plot_history(history_frame, dirs["metrics"], f"METRICS_FOLD_{fold_number}.png", f"Fold {fold_number}", SHOW_PLOTS)

        test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        train_raw_probs = model.predict(X_train, verbose=0).reshape(-1)
        train_prediction_metrics, _, _, _ = evaluate_predictions(
            df_train,
            y_train,
            train_raw_probs,
            CLASSIFICATION_THRESHOLD,
            class_names,
            MAP_GROUP_COLUMNS,
            SMOOTH_PREDICTION_PROBS,
            PREDICTION_SMOOTHING_METHOD,
            PREDICTION_SMOOTHING_KERNEL_SIZE,
        )
        y_raw_probs = model.predict(X_test, verbose=0).reshape(-1)
        prediction_metrics, cm, y_pred_probs, y_pred_classes = evaluate_predictions(
            df_test,
            y_test,
            y_raw_probs,
            CLASSIFICATION_THRESHOLD,
            class_names,
            MAP_GROUP_COLUMNS,
            SMOOTH_PREDICTION_PROBS,
            PREDICTION_SMOOTHING_METHOD,
            PREDICTION_SMOOTHING_KERNEL_SIZE,
        )
        cm_total += cm

        fold_result = {
            "fold": fold_number,
            "epochs_ran": len(history.history["loss"]),
            "train_rows_before_balancing": len(df_train_unbalanced),
            "train_rows": len(y_train),
            "validation_rows": len(y_val),
            "test_rows": len(y_test),
            "train_groups": len(train_groups),
            "validation_groups": len(val_groups),
            "test_groups": len(test_groups),
            "train_class_1_rate": y_train.mean(),
            "validation_class_1_rate": y_val.mean(),
            "test_class_1_rate": y_test.mean(),
            "train_class_0_before_balancing": train_counts_before.get(0, 0),
            "train_class_1_before_balancing": train_counts_before.get(1, 0),
            "train_class_0_after_balancing": train_counts_after.get(0, 0),
            "train_class_1_after_balancing": train_counts_after.get(1, 0),
            "feature_mode": "center_spectrum_only",
            "smooth_prediction_probs": SMOOTH_PREDICTION_PROBS,
            "prediction_smoothing_method": PREDICTION_SMOOTHING_METHOD if SMOOTH_PREDICTION_PROBS else "none",
            "prediction_smoothing_kernel_size": PREDICTION_SMOOTHING_KERNEL_SIZE if SMOOTH_PREDICTION_PROBS else 0,
            "test_loss": test_metrics.get("loss"),
            "keras_test_accuracy_at_0_5": test_metrics.get("accuracy"),
            "keras_test_sensitivity_at_0_5": test_metrics.get("sensitivity"),
            "keras_test_auc_raw": test_metrics.get("auc"),
            "train_sample_ids": ", ".join(sorted(train_groups)),
            "validation_sample_ids": ", ".join(sorted(val_groups)),
            "test_sample_ids": ", ".join(sorted(test_groups)),
        }
        fold_result.update(prefix_train_metrics(train_prediction_metrics))
        fold_result.update(prediction_metrics)
        fold_results.append(fold_result)
        sample_results.extend(compute_sample_results(fold_number, y_test, y_pred_probs, y_pred_classes, groups_test, class_names))
        save_prediction_maps_if_enabled(df_test, y_pred_probs, y_pred_classes, fold_number, dirs["maps"], SAVE_PREDICTION_MAPS, SHOW_PLOTS)

        model_path = os.path.join(dirs["params"], f"ann_{MODEL_SIZE.lower()}_pca{N_COMPONENTS}_fold_{fold_number}.keras")
        weights_path = os.path.join(dirs["params"], f"ann_{MODEL_SIZE.lower()}_pca{N_COMPONENTS}_fold_{fold_number}.weights.h5")
        model.save(model_path)
        model.save_weights(weights_path)

        print(
            f"Fold {fold_number} train - "
            f"accuracy: {fold_result['train_accuracy']:.4f}, "
            f"auc: {fold_result['train_auc']:.4f}, "
            f"class_1 recall: {fold_result['train_class_1_recall']:.4f}, "
            f"class_1 precision: {fold_result['train_class_1_precision']:.4f}"
        )
        print(
            f"Fold {fold_number} test - "
            f"accuracy: {fold_result['test_accuracy']:.4f}, "
            f"auc: {fold_result['test_auc']:.4f}, "
            f"class_1 recall: {fold_result['class_1_recall']:.4f}, "
            f"class_1 precision: {fold_result['class_1_precision']:.4f}"
        )

        del model, X_train, X_val, X_test
        gc.collect()

    fold_results_df = pd.DataFrame(fold_results)
    sample_results_df = pd.DataFrame(sample_results)
    fold_label_distribution_df = pd.DataFrame(fold_label_distribution)

    fold_results_df.to_csv(os.path.join(dirs["tables"], "fold_results.csv"), index=False)
    sample_results_df.to_csv(os.path.join(dirs["tables"], "sample_results.csv"), index=False)
    fold_label_distribution_df.to_csv(os.path.join(dirs["tables"], "fold_label_distribution.csv"), index=False)
    sample_ranking_df = plot_sample_difficulty_ranking(
        sample_results_df,
        dirs["metrics"],
        dirs["tables"],
        SAMPLE_RANKING_PRIMARY_METRIC,
        SAMPLE_RANKING_SECONDARY_METRIC,
        SHOW_PLOTS,
    )

    history_all, history_mean, history_std = summarize_histories(history_frames)
    history_all.to_csv(os.path.join(dirs["tables"], "history_all_folds.csv"), index=False)
    history_mean.to_csv(os.path.join(dirs["tables"], "history_mean.csv"), index=False)
    history_std.to_csv(os.path.join(dirs["tables"], "history_std.csv"), index=False)

    numeric_cols, summary_mean, summary_std = summarize_fold_metrics(fold_results_df, exclude_cols=["fold"])

    summary_row = {
        "run_name": run_name,
        "model_size": MODEL_SIZE,
        "n_components": N_COMPONENTS,
        "n_folds": N_FOLDS,
        "use_snv": USE_SNV,
        "feature_mode": "center_spectrum_only",
        "balance_classes": BALANCE_CLASSES,
        "balancing_scope": "training_only",
        "smooth_prediction_probs": SMOOTH_PREDICTION_PROBS,
        "prediction_smoothing_method": PREDICTION_SMOOTHING_METHOD if SMOOTH_PREDICTION_PROBS else "none",
        "prediction_smoothing_kernel_size": PREDICTION_SMOOTHING_KERNEL_SIZE if SMOOTH_PREDICTION_PROBS else 0,
        "class_0_labels": str(class_0_labels),
        "class_1_labels": str(class_1_labels),
        "mean_test_auc": summary_mean.get("test_auc"),
        "std_test_auc": summary_std.get("test_auc"),
        "mean_test_accuracy": summary_mean.get("test_accuracy"),
        "std_test_accuracy": summary_std.get("test_accuracy"),
        "mean_train_auc": summary_mean.get("train_auc"),
        "std_train_auc": summary_std.get("train_auc"),
        "mean_train_accuracy": summary_mean.get("train_accuracy"),
        "std_train_accuracy": summary_std.get("train_accuracy"),
        "mean_train_class_1_precision": summary_mean.get("train_class_1_precision"),
        "std_train_class_1_precision": summary_std.get("train_class_1_precision"),
        "mean_train_class_1_recall": summary_mean.get("train_class_1_recall"),
        "std_train_class_1_recall": summary_std.get("train_class_1_recall"),
        "mean_train_class_1_f1": summary_mean.get("train_class_1_f1"),
        "std_train_class_1_f1": summary_std.get("train_class_1_f1"),
        "mean_class_1_precision": summary_mean.get("class_1_precision"),
        "std_class_1_precision": summary_std.get("class_1_precision"),
        "mean_class_1_recall": summary_mean.get("class_1_recall"),
        "std_class_1_recall": summary_std.get("class_1_recall"),
        "mean_class_1_f1": summary_mean.get("class_1_f1"),
        "std_class_1_f1": summary_std.get("class_1_f1"),
        "mean_epochs_ran": summary_mean.get("epochs_ran"),
        "std_epochs_ran": summary_std.get("epochs_ran"),
    }
    pd.DataFrame([summary_row]).to_csv(os.path.join(dirs["tables"], "comparative_results.csv"), index=False)

    cm_normalized = plot_confusion_matrices(cm_total, class_names, dirs["metrics"], SHOW_PLOTS)
    plot_mean_history(history_mean, history_std, dirs["metrics"], SHOW_PLOTS)
    save_run_summary(
        os.path.join(base_save_dir, "run_summary.txt"),
        base_save_dir,
        config,
        df_raw.shape,
        df_filtered.shape,
        len(band_cols),
        df_filtered[GROUP_COLUMN].nunique(),
        filtered_class_counts,
        numeric_cols,
        summary_mean,
        summary_std,
        cm_total,
        cm_normalized,
        sample_ranking_df,
    )

    print("\n" + "=" * 70)
    print("5-FOLD SUMMARY")
    print("=" * 70)
    for metric_name in [
        "train_accuracy",
        "train_auc",
        "train_class_1_precision",
        "train_class_1_recall",
        "train_class_1_f1",
        "test_accuracy",
        "test_auc",
        "class_1_precision",
        "class_1_recall",
        "class_1_f1",
    ]:
        print(f"{metric_name}: {summary_mean[metric_name]:.4f} +/- {summary_std[metric_name]:.4f}")
    print(f"\nRun saved at: {base_save_dir}")


if __name__ == "__main__":
    main()
