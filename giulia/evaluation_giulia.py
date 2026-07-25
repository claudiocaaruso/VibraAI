import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from pipeline_giulia import smooth_prediction_probabilities


def label_distribution_row(fold, split_name, y_values, groups):
    return {
        "fold": fold,
        "split": split_name,
        "rows": len(y_values),
        "class_0_rows": int((y_values == 0).sum()),
        "class_1_rows": int((y_values == 1).sum()),
        "class_1_rate": float(y_values.mean()) if len(y_values) else np.nan,
        "sample_id_count": int(pd.Series(groups).nunique()),
        "sample_ids": ", ".join(sorted(pd.Series(groups).astype(str).unique())),
    }


def history_to_frame(history, fold_number):
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    frame.insert(0, "fold", fold_number)
    return frame


def summarize_histories(history_frames):
    min_epoch_count = min(len(frame) for frame in history_frames)
    trimmed_frames = [frame.iloc[:min_epoch_count].copy() for frame in history_frames]
    history_all = pd.concat(trimmed_frames, ignore_index=True)
    metric_cols = [col for col in history_all.columns if col not in ["fold", "epoch"]]
    grouped = history_all.groupby("epoch")[metric_cols]
    return history_all, grouped.mean().reset_index(), grouped.std().reset_index()


def evaluate_predictions(
    df_rows,
    y_true,
    y_raw_probs,
    threshold,
    class_names,
    map_group_columns,
    smooth_prediction_probs,
    smoothing_method,
    smoothing_kernel_size,
):
    """Apply optional post-processing and compute binary classification metrics."""
    y_probs = smooth_prediction_probabilities(
        df_rows,
        y_raw_probs,
        map_group_columns,
        smooth_prediction_probs,
        smoothing_method,
        smoothing_kernel_size,
    )
    y_pred = (y_probs > threshold).astype("int32")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) == 2 else np.nan

    metrics = {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_sensitivity": report[class_names[1]]["recall"],
        "test_auc": auc,
        "test_auc_sklearn": auc,
        "class_0_precision": report[class_names[0]]["precision"],
        "class_0_recall": report[class_names[0]]["recall"],
        "class_0_f1": report[class_names[0]]["f1-score"],
        "class_1_precision": report[class_names[1]]["precision"],
        "class_1_recall": report[class_names[1]]["recall"],
        "class_1_f1": report[class_names[1]]["f1-score"],
        "confusion_tn": int(cm[0, 0]),
        "confusion_fp": int(cm[0, 1]),
        "confusion_fn": int(cm[1, 0]),
        "confusion_tp": int(cm[1, 1]),
    }
    return metrics, cm, y_probs, y_pred


def summarize_fold_metrics(fold_results_df, exclude_cols=None):
    """Return means and standard deviations for numeric fold-level metrics."""
    exclude_cols = [] if exclude_cols is None else exclude_cols
    numeric_cols = fold_results_df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop([col for col in exclude_cols if col in numeric_cols])
    return numeric_cols, fold_results_df[numeric_cols].mean(), fold_results_df[numeric_cols].std()


def compute_sample_results(fold_number, y_test, y_probs, y_pred, groups_test, class_names):
    rows = []
    y_probs = np.asarray(y_probs).reshape(-1)

    for sample_id in sorted(pd.Series(groups_test).astype(str).unique()):
        mask = pd.Series(groups_test).astype(str).to_numpy() == sample_id
        y_s = y_test[mask]
        p_s = y_probs[mask]
        pred_s = y_pred[mask]

        cm = confusion_matrix(y_s, pred_s, labels=[0, 1])
        report = classification_report(
            y_s,
            pred_s,
            labels=[0, 1],
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        auc = roc_auc_score(y_s, p_s) if len(np.unique(y_s)) == 2 else np.nan

        rows.append(
            {
                "fold": fold_number,
                "sample_id": sample_id,
                "rows": len(y_s),
                "class_0_rows": int((y_s == 0).sum()),
                "class_1_rows": int((y_s == 1).sum()),
                "class_1_rate": float(y_s.mean()) if len(y_s) else np.nan,
                "accuracy": float((pred_s == y_s).mean()) if len(y_s) else np.nan,
                "auc": auc,
                "class_0_precision": report[class_names[0]]["precision"],
                "class_0_recall": report[class_names[0]]["recall"],
                "class_0_f1": report[class_names[0]]["f1-score"],
                "class_1_precision": report[class_names[1]]["precision"],
                "class_1_recall": report[class_names[1]]["recall"],
                "class_1_f1": report[class_names[1]]["f1-score"],
                "confusion_tn": int(cm[0, 0]),
                "confusion_fp": int(cm[0, 1]),
                "confusion_fn": int(cm[1, 0]),
                "confusion_tp": int(cm[1, 1]),
            }
        )

    return rows
