"""
Step 1 — Dataset and problem definition (age estimation with UTKFace).

Usage:
    python 1_dataset_and_problem.py              # uses existing outputs if available
    python 1_dataset_and_problem.py --rebuild    # rescans the UTKFace folder
    python 1_dataset_and_problem.py --resplit    # regenerates the train/val/test split

Outputs (data/ folder):
    utkface_clean.csv    : cleaned image list (filename, path, age, gender, ethnicity)
    utkface_rejected.csv : removed files and the reasons (filename, reason)
    splits.csv           : clean list + split column (train / val / test)
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from data_utils import (DATA_DIR, SPLITS_CSV, SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
                        AGE_GROUP_ORDER, GENDER_NAMES, ETHNICITY_NAMES, HUBER_DELTA,
                        age_group, load_split, regression_metrics, group_errors, print_table)

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(ROOT, "UTKFace")
CLEAN_CSV = os.path.join(DATA_DIR, "utkface_clean.csv")
REJECTED_CSV = os.path.join(DATA_DIR, "utkface_rejected.csv")


# =====================================================================================
# [x] Choose a public, academically licensed image dataset whose target is a continuous
#     number. Age estimation with UTKFace.
# =====================================================================================
# UTKFace: 200x200 aligned face images, open for academic (non-commercial) use.
# Target: age (0–116, a continuous number). The labels are stored in the file name:
#     [age]_[gender]_[ethnicity]_[date-time].jpg.chip.jpg
#     gender    : 0 = male, 1 = female
#     ethnicity : 0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Others
#
# The raw folder contains three kinds of faulty files, which are removed:
#     bad_name : files that do not match the name format (missing field, space, missing dot, bad date)
#     not_jpeg : files whose content is not JPEG (does not start with the JPEG signature FF D8 FF)
#     corrupt  : files with a correct signature that cannot be read to the end (corrupt / truncated)

NAME_PATTERN = re.compile(r"^(\d{1,3})_([01])_([0-4])_\d{17}\.jpg(?:\.chip\.jpg)?$")
JPEG_SIGNATURE = b"\xff\xd8\xff"
MAX_AGE = 116


def parse_filename(filename):
    """Reads (age, gender, ethnicity) from the file name. Returns None if the format is wrong."""
    match = NAME_PATTERN.match(filename)
    if match is None:
        return None
    age, gender, ethnicity = (int(value) for value in match.groups())
    if not 1 <= age <= MAX_AGE:
        return None
    return age, gender, ethnicity


def find_image_problem(path):
    """Returns None if the image is intact, otherwise the name of the problem."""
    with open(path, "rb") as file:
        if file.read(3) != JPEG_SIGNATURE:
            return "not_jpeg"
    try:
        with Image.open(path) as image:
            image.load()  # decodes all pixels; raises an error on a corrupt/truncated file
    except Exception:
        return "corrupt"
    return None


def build_clean_dataset():
    """Scans the UTKFace folder and writes the intact and the removed images to CSV."""
    clean_rows, rejected_rows = [], []

    for filename in sorted(os.listdir(IMAGE_DIR)):
        labels = parse_filename(filename)
        if labels is None:
            rejected_rows.append({"filename": filename, "reason": "bad_name"})
            continue

        problem = find_image_problem(os.path.join(IMAGE_DIR, filename))
        if problem is not None:
            rejected_rows.append({"filename": filename, "reason": problem})
            continue

        age, gender, ethnicity = labels
        clean_rows.append({"filename": filename, "path": f"UTKFace/{filename}",
                           "age": age, "gender": gender, "ethnicity": ethnicity})

    clean = pd.DataFrame(clean_rows)
    rejected = pd.DataFrame(rejected_rows, columns=["filename", "reason"])

    os.makedirs(DATA_DIR, exist_ok=True)
    clean.to_csv(CLEAN_CSV, index=False)
    rejected.to_csv(REJECTED_CSV, index=False)

    print(f"Scanned: {len(clean) + len(rejected)}  |  Intact: {len(clean)}  |  Removed: {len(rejected)}")
    print(rejected["reason"].value_counts().to_string())
    return clean


def load_clean_dataset(rebuild):
    """Reads the clean list if it was created before, otherwise (or with --rebuild) regenerates it."""
    if os.path.exists(CLEAN_CSV) and not rebuild:
        print(f"Using existing clean list: {CLEAN_CSV}")
        return pd.read_csv(CLEAN_CSV)
    return build_clean_dataset()


# =====================================================================================
# [x] Use at least ~2,000 images. If you take a subset of a large set, explain how you chose it.
# =====================================================================================
# No subset is taken: all images left after cleaning (23,213) are used.
# This is far above the ~2,000 lower bound; no selection bias is introduced.

MIN_IMAGES = 2000


def describe_dataset(clean):
    """Checks the dataset size and prints the target/group distributions."""
    assert len(clean) >= MIN_IMAGES, f"At least {MIN_IMAGES} images are required, there are {len(clean)}."

    print(f"\nNumber of images used: {len(clean)}")
    print("\nAge (target) statistics:")
    print(clean["age"].describe().round(2).to_string())
    print("\nGender distribution:")
    print(clean["gender"].map(GENDER_NAMES).value_counts().to_string())
    print("\nEthnicity distribution:")
    print(clean["ethnicity"].map(ETHNICITY_NAMES).value_counts().to_string())


# =====================================================================================
# [x] Make the train/val/test split with a fixed seed and use the same split for all models.
# =====================================================================================
# - Ratios 70% / 15% / 15%, seed = 42 (defined in data_utils.py).
# - Split stratified by age group x gender: the age and gender distribution stays the same in each split.
# - The split is written to data/splits.csv once and preserved; all models read this file with
#   data_utils.load_split(). So every model is trained and measured on the same images.


def make_splits(clean, resplit):
    """Creates the train/val/test split (or reads the existing one)."""
    if os.path.exists(SPLITS_CSV) and not resplit:
        print(f"\nKeeping existing split: {SPLITS_CSV}")
        return pd.read_csv(SPLITS_CSV)

    strata = clean["age"].map(age_group) + "_" + clean["gender"].astype(str)

    # Step 1: hold out the test set.  Step 2: split the rest into train and val.
    train_val, test = train_test_split(clean, test_size=TEST_FRAC,
                                       stratify=strata, random_state=SEED)
    val_share = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    _train, val = train_test_split(train_val, test_size=val_share,
                                  stratify=strata[train_val.index], random_state=SEED)

    splits = clean.copy()
    splits["split"] = "train"
    splits.loc[val.index, "split"] = "val"
    splits.loc[test.index, "split"] = "test"
    splits.to_csv(SPLITS_CSV, index=False)
    print(f"\nNew split created (seed={SEED}): {SPLITS_CSV}")
    return splits


def describe_splits(splits):
    """Shows the split sizes and how similar their age distributions are."""
    order = ["train", "val", "test"]
    print("\nSplit sizes and age statistics:")
    stats = splits.groupby("split")["age"].agg(["count", "mean", "std", "median"])
    print(stats.reindex(order).round(2).to_string())

    print("\nAge group shares (%):")
    shares = pd.crosstab(splits["age"].map(age_group), splits["split"], normalize="columns") * 100
    print(shares.reindex(index=AGE_GROUP_ORDER, columns=order).round(2).to_string())


# =====================================================================================
# [x] Use the test set only once for each final model, never for hyperparameter tuning.
# =====================================================================================
# This rule is enforced in code in data_utils.py:
# - load_split("train") / load_split("val") : training and hyperparameter tuning use only these.
# - load_split("test")                      : deliberately raises an error.
# - load_test_once("model_name")            : opens the test set once for that model and records it
#                                              in data/test_usage.json; a second call raises an error.


# =====================================================================================
# [x] Choose the loss function (MSE, L1 or Huber) and the metrics (MAE, RMSE, R²).
# =====================================================================================
# The three losses and three metrics are compared in tables, then the choice is printed with its
# rationale. The choice is fixed in data_utils.py (LOSS_NAME, HUBER_DELTA, regression_metrics) and all
# models use the same loss/metrics. Error definition: e = y - prediction (in years).

LOSS_TABLE = [
    ["Loss", "Formula", "Response to large errors", "Gradient", "When it fits"],
    ["MSE", "mean(e^2)", "Quadratic: very sensitive to outliers",
     "2e (grows with the error)", "Noise-free labels, few outliers"],
    ["L1", "mean(|e|)", "Linear: robust to outliers",
     "sign(e) (constant, kink at e=0)", "Noisy labels; estimates the median"],
    ["Huber", "|e|<=d: 0.5*e^2,  else d*(|e|-0.5*d)", "Quadratic for small, linear for large errors",
     "Smooth for small errors, at most d", "Balance between MSE and L1"],
]

METRIC_TABLE = [
    ["Metric", "Formula", "Unit", "Interpretation"],
    ["MAE", "mean(|e|)", "years", "How many years off on average; the easiest to interpret"],
    ["RMSE", "sqrt(mean(e^2))", "years", "Penalizes large errors heavily; RMSE >> MAE means there are large errors"],
    ["R²", "1 - SSE / SST", "unitless", "1 = perfect, 0 = as good as always predicting the mean, < 0 = worse"],
]

EXAMPLE_ERRORS = [1, 2, 5, 10, 20, 40]  # years


def mse_loss(error):
    return error ** 2


def l1_loss(error):
    return np.abs(error)


def huber_loss(error, delta=HUBER_DELTA):
    abs_error = np.abs(error)
    return np.where(abs_error <= delta, 0.5 * error ** 2, delta * (abs_error - 0.5 * delta))


def compare_losses_and_metrics():
    """Prints the loss and metric comparison tables and the choice made."""
    print_table("Loss functions:", LOSS_TABLE)
    print_table("Evaluation metrics:", METRIC_TABLE)

    errors = np.array(EXAMPLE_ERRORS, dtype=float)
    example = [["Error (years)", "MSE", "L1", f"Huber (d={HUBER_DELTA:g})"]]
    for error, mse, l1, huber in zip(errors, mse_loss(errors), l1_loss(errors), huber_loss(errors)):
        example.append([f"{error:g}", f"{mse:g}", f"{l1:g}", f"{huber:g}"])
    print_table("Contribution of a single sample to the loss (as the error grows):", example)

    print(f"""
Choice:
  Loss   : Huber, d = {HUBER_DELTA:g} years
           - UTKFace ages are labeled by appearance, are noisy and have a tail reaching 116.
             With MSE a single 40-year error adds a loss of {mse_loss(40):g} and locks the training onto
             these few samples; with Huber its contribution stays at {huber_loss(40):g}.
           - For errors below {HUBER_DELTA:g} years Huber behaves quadratically: instead of L1's constant
             gradient it gives a smooth gradient that shrinks as the error shrinks, and converges well.
  Metric : MAE is the primary metric (in years, directly interpretable). Hyperparameter and
           model selection are done on validation MAE.
           RMSE is reported in addition to see large errors, and R² to see how much better we are
           than the baseline that always predicts the mean.""")


# =====================================================================================
# [x] Trivial baseline: compute the model that always predicts the mean (or median)
#     of the training targets.
# =====================================================================================
# Two models that never look at the image and predict the same constant age for every sample:
#     mean baseline   : mean of the train ages    (the constant that minimizes MSE)
#     median baseline : median of the train ages  (the constant that minimizes MAE)
# The constant is computed from the train set only and measured on validation.
# The assignment's acceptance criterion is defined via "MAE_mean = MAE of the mean baseline", so
# the MEAN baseline is used for the threshold. The median baseline is reported only as an extra reference.
# The test set is not opened here; the baselines are also tested in the final evaluation (step 5).


def evaluate_baselines():
    """Prints the train and validation metrics of the mean and median baselines."""
    train_ages = load_split("train")["age"]
    val_ages = load_split("val")["age"]

    baselines = {"Mean": train_ages.mean(), "Median": train_ages.median()}

    rows = [["Baseline", "Prediction (years)", "Split", "MAE", "RMSE", "R²"]]
    for name, constant in baselines.items():
        for split_name, ages in [("train", train_ages), ("val", val_ages)]:
            predictions = np.full(len(ages), constant)
            metrics = regression_metrics(ages, predictions)
            rows.append([name, f"{constant:.2f}", split_name,
                         f"{metrics['MAE']:.2f}", f"{metrics['RMSE']:.2f}", f"{metrics['R2']:.4f}"])
    print_table("Trivial baseline results:", rows)

    mae_mean_val = regression_metrics(val_ages, np.full(len(val_ages), baselines["Mean"]))["MAE"]
    print(f"""
Acceptance criterion reference: MAE_mean (val) = {mae_mean_val:.2f} years  (mean baseline)

Interpretation:
  - Following the assignment's definition, the threshold is computed with the MAE of the mean baseline (MAE_mean).
  - The median baseline's MAE is slightly lower, because the median is the constant that minimizes MAE.
    It is reported as an extra reference: if a model also beats the median baseline, the gain comes
    from learning from the image, not just from the choice of constant.
  - R² of the mean baseline is ~0 by definition (exactly 0 on train). The models' R² shows how much
    they explain relative to this zero point.""")


# =====================================================================================
# [x] If the data has group labels (such as gender, age range), report the error by group
#     and interpret the bias.
# =====================================================================================
# UTKFace has three group labels: gender, ethnicity and age group (derived from age).
# The error is computed separately for each group with data_utils.group_errors(); the same function
# is also used for the CNNs in later steps.
#
# Here the validation errors of the mean baseline (the reference in the acceptance criterion) are
# reported. Since the baseline does not look at the image, its error in each group comes only from
# that group's age distribution. This table is a "bias from the data" reference for comparing the
# models' group differences: if a model's group difference is larger than the baseline's, the
# difference is a bias the model learned from the images.


def report_group_errors():
    """Prints and interprets the group-wise validation errors of the mean baseline."""
    val = load_split("val")
    mean_age = load_split("train")["age"].mean()
    predictions = np.full(len(val), mean_age)
    tables = group_errors(val, predictions)
    overall_mae = regression_metrics(val["age"], predictions)["MAE"]

    print(f"\nGroup-wise error — mean baseline (prediction = {mean_age:.2f} years), validation, "
          f"overall MAE = {overall_mae:.2f}")
    for group_name, table in tables.items():
        rows = [[group_name, "n", "Mean age", "MAE", "RMSE", "Bias"]]
        for group, values in table.iterrows():
            rows.append([group, int(values["n"]), f"{values['Mean age']:.1f}", f"{values['MAE']:.2f}",
                         f"{values['RMSE']:.2f}", f"{values['Bias']:+.2f}"])
        print_table(f"By {group_name}:", rows)

        worst, best = table["MAE"].idxmax(), table["MAE"].idxmin()
        print(f"  Highest MAE: {worst} ({table.loc[worst, 'MAE']:.2f}), "
              f"lowest: {best} ({table.loc[best, 'MAE']:.2f}), "
              f"difference: {table.loc[worst, 'MAE'] - table.loc[best, 'MAE']:.2f} years")

    print("""
Interpretation:
  - The baseline's error depends on how far the group's mean age is from the predicted constant.
    Bias is negative in groups with a high mean age (predicted younger than they are) and positive
    in groups with a low mean age. So the group differences here come entirely from the data distribution.
  - Age group creates the largest difference: the age group close to the constant (train mean) has a
    small error, children and the elderly get the largest error. Whether a model is useful shows most
    clearly in these extreme groups.
  - The groups are imbalanced (column n): in groups with few samples the model learns from less data
    and the MAE estimate is noisier. The CNNs' group tables should be interpreted by comparing them
    with this baseline.""")


if __name__ == "__main__":
    clean_dataset = load_clean_dataset(rebuild="--rebuild" in sys.argv)
    describe_dataset(clean_dataset)

    split_dataset = make_splits(clean_dataset, resplit="--resplit" in sys.argv)
    describe_splits(split_dataset)

    compare_losses_and_metrics()
    evaluate_baselines()
    report_group_errors()
