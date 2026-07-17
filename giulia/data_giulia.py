import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf


def first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_json_safe(value):
    """Convert NumPy/Pandas scalar values to plain Python objects for JSON."""
    if isinstance(value, dict):
        return {str(make_json_safe(key)): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def save_config(config, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(config), f, indent=2)


def get_band_columns(df):
    band_cols = [col for col in df.columns if col.startswith("band_")]
    if not band_cols:
        raise ValueError("No spectral band columns found. Expected columns named like band_1, band_2, ...")
    return band_cols


def filter_labels(df, class_1_labels_config, class_0_labels_config):
    """Keep selected labels and create Binary_Label."""
    class_1_labels = list(class_1_labels_config)

    if class_0_labels_config == "all_other":
        df_work = df.copy()
        df_work["Binary_Label"] = df_work["Label"].apply(lambda x: 1 if x in class_1_labels else 0)
        class_0_labels = sorted([label for label in df_work["Label"].unique() if label not in class_1_labels])
    else:
        class_0_labels = list(class_0_labels_config)
        keep_labels = class_0_labels + class_1_labels
        df_work = df[df["Label"].isin(keep_labels)].copy()
        df_work["Binary_Label"] = df_work["Label"].apply(lambda x: 1 if x in class_1_labels else 0)

    if df_work.empty:
        raise ValueError("No rows left after label filtering. Check class label configuration.")

    return df_work, class_0_labels, class_1_labels


def class_counts(y_values):
    """Return class counts with stable 0/1 keys."""
    y_values = np.asarray(y_values, dtype=int)
    return {
        0: int((y_values == 0).sum()),
        1: int((y_values == 1).sum()),
    }
