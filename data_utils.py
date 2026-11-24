"""
Data settings and helper functions shared by all steps (1–5).

    load_split("train" | "val")      -> split for training / hyperparameter tuning
    load_test_once("model_name")     -> test set for the final evaluation (only once per model)
    age_group(age)                   -> age-range label (for stratification and group-wise reports)
    regression_metrics(y, prediction) -> MAE, RMSE, R² (all models use the same metrics)
    group_errors(samples, prediction) -> error tables by gender / ethnicity / age group
    load_images(samples, size, name) -> downscaled images (uint8), cached on disk
    print_table(title, rows)         -> left-aligned table in the console
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SPLITS_CSV = os.path.join(DATA_DIR, "splits.csv")
TEST_USAGE_LOG = os.path.join(DATA_DIR, "test_usage.json")

# Split settings: fixed for all models
SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Loss and metric choice: fixed for all models (rationale: 1_dataset_and_problem.py)
LOSS_NAME = "huber"
HUBER_DELTA = 5.0  # years; the target (age) is used unscaled

AGE_GROUP_ORDER = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER_NAMES = {0: "male", 1: "female"}
ETHNICITY_NAMES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}


def age_group(age):
    """Converts an age to a range label: 0-2, 3-9, 10-19, ..., 60-69, 70+."""
    if age <= 2:
        return "0-2"
    if age <= 9:
        return "3-9"
    if age >= 70:
        return "70+"
    decade = (age // 10) * 10
    return f"{decade}-{decade + 9}"


def regression_metrics(y_true, y_pred):
    """Returns MAE, RMSE (years) and R²."""
    y_true = np.asarray(y_true, dtype=float)
    errors = y_true - np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "R2": float(1 - np.sum(errors ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }


def group_errors(samples, predictions):
    """Returns error tables by gender, ethnicity and age group.

    Each table has: n, the group's mean age, MAE, RMSE and Bias.
    Bias = mean(prediction - true): + means the group is predicted older, - younger than it is.
    """
    data = pd.DataFrame({
        "Gender": samples["gender"].map(GENDER_NAMES),
        "Ethnicity": samples["ethnicity"].map(ETHNICITY_NAMES),
        "Age group": samples["age"].map(age_group),
        "age": samples["age"],
        "error": np.asarray(predictions, dtype=float) - samples["age"].to_numpy(),
    })

    tables = {}
    for column in ["Gender", "Ethnicity", "Age group"]:
        groups = data.groupby(column)
        table = pd.DataFrame({
            "n": groups.size(),
            "Mean age": groups["age"].mean(),
            "MAE": groups["error"].apply(lambda e: e.abs().mean()),
            "RMSE": groups["error"].apply(lambda e: np.sqrt((e ** 2).mean())),
            "Bias": groups["error"].mean(),
        })
        if column == "Age group":
            table = table.reindex(AGE_GROUP_ORDER)
        tables[column] = table
    return tables


def decode_images(samples, size):
    """Opens the JPEGs and downscales them to size x size (no cache); returns (N, size, size, 3) uint8."""
    return np.stack([
        np.asarray(Image.open(os.path.join(ROOT, path)).convert("RGB")
                   .resize((size, size), Image.BILINEAR))
        for path in samples["path"]
    ])


def load_images(samples, size, cache_name):
    """Downscales the images to size x size and returns them as an (N, size, size, 3) uint8 array.

    On the first call the result is written to data/cache/<cache_name>_<size>.npz; later calls
    read directly from the cache if the file list is the same (so JPEGs are not decoded every epoch).
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}_{size}.npz")
    filenames = samples["filename"].to_numpy()

    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if np.array_equal(cached["filenames"], filenames):
            return cached["images"]

    images = decode_images(samples, size)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, images=images, filenames=filenames.astype(str))
    return images


def print_table(title, rows):
    """Prints a table with left-aligned columns, treating the first row as the header."""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    line = "-+-".join("-" * width for width in widths)

    print(f"\n{title}")
    for index, row in enumerate(rows):
        print(" | ".join(str(cell).ljust(width) for cell, width in zip(row, widths)))
        if index == 0:
            print(line)


def load_step_module(filename, name):
    """Loads step files whose names start with a digit (such as 2_own_cnn.py) as modules."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_section(title):
    """Prints a section header in the console."""
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")


def _read_split(name):
    if not os.path.exists(SPLITS_CSV):
        raise FileNotFoundError("Split not found. Run `python 1_dataset_and_problem.py` first.")
    splits = pd.read_csv(SPLITS_CSV)
    return splits[splits["split"] == name].reset_index(drop=True)


def load_split(name):
    """Returns the train or val split. The test set cannot be opened with this function."""
    if name == "test":
        raise PermissionError("The test set can only be opened in the final evaluation, "
                              "with load_test_once(model_name).")
    if name not in ("train", "val"):
        raise ValueError(f"Unknown split: {name!r}")
    return _read_split(name)


def load_test_once(model_name):
    """Opens the test set once for the given final model and records the usage."""
    usage = {}
    if os.path.exists(TEST_USAGE_LOG):
        with open(TEST_USAGE_LOG, encoding="utf-8") as file:
            usage = json.load(file)

    if model_name in usage:
        raise PermissionError(f"'{model_name}' has already used the test set ({usage[model_name]}).")

    usage[model_name] = datetime.now().isoformat(timespec="seconds")
    with open(TEST_USAGE_LOG, "w", encoding="utf-8") as file:
        json.dump(usage, file, indent=2, ensure_ascii=False)

    return _read_split("test")


def release_test_use(model_name):
    """Removes the test-usage record of a model whose test evaluation CRASHED before any result was produced.

    Only for that case: no metric was computed or shown, so the test set has not influenced anything; without
    this, a crash (e.g. a missing file) would make the model's test evaluation impossible forever.
    """
    with open(TEST_USAGE_LOG, encoding="utf-8") as file:
        usage = json.load(file)
    usage.pop(model_name, None)
    with open(TEST_USAGE_LOG, "w", encoding="utf-8") as file:
        json.dump(usage, file, indent=2, ensure_ascii=False)
