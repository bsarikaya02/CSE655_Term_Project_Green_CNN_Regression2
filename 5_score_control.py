"""
Step 5 — Score check: does our own CNN meet the acceptance criteria?

Usage:
    python 5_score_control.py

Test set rule: the test set can be opened ONLY ONCE for each final model (data_utils.load_test_once)
and is never used for hyperparameter tuning. Therefore:
    USE_TEST_SET = False  -> the checks are done on the validation set (pre-check). Model
                             improvements are evaluated by looking at this mode.
    USE_TEST_SET = True   -> final step: the final models are evaluated once on the test set; the results
                             and predictions are saved under models/test_results/. Later runs do not
                             reopen the test set, they read the saved results.
Any change to a model after the test set has been opened means tuning based on the test result.
"""

import json
import os
import time
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")  # draw to a file without opening a window
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd
import torch

from data_utils import (ROOT, group_errors, load_split, load_test_once, load_images, regression_metrics,
                        print_section, print_table, load_step_module, release_test_use)
from train_utils import INFERENCE_REPEATS, MODEL_DIR, measure_energy, predict, train_mean_age

own_cnn = load_step_module("2_own_cnn.py", "own_cnn")
standard = load_step_module("3_comparing_standard_models.py", "standard_models")
footprint = load_step_module("4_footprint.py", "footprint")

USE_TEST_SET = True
TEST_RESULTS_DIR = os.path.join(MODEL_DIR, "test_results")


class Config(NamedTuple):
    """A model configuration that can be evaluated: one or more trained models combined at prediction time.

    A trained model alone is a configuration with one part. Ensemble = several parts (their predictions are
    averaged); flip TTA = each part also predicts the mirrored image and the two predictions are averaged.
    """
    name: str
    short: str          # short label for wide tables (bias analysis)
    run_name: str       # name of the configuration's runs in emissions.csv and of its summary file
    parts: list         # trained models used: [(run name of the training, model builder, input size)]
    flip_tta: bool = False


# Trained models: (run name, model builder, input size)
OWN_64 = ("own_cnn_64", lambda: own_cnn.AgeCNN(), 64)
OWN_48 = ("own_cnn_48", lambda: own_cnn.AgeCNN(), 48)
RESNET = (standard.RESNET_RUN, standard.build_resnet18, 64)
MOBILENET = (standard.MOBILENET_RUN, standard.build_mobilenet_v3_large, 64)

OWN_TRAINED = [
    Config("Own CNN (64px)", "Own 64", "own_cnn_64", [OWN_64]),
    Config("Own CNN (48px)", "Own 48", "own_cnn_48", [OWN_48]),
]
# Improvements without retraining (analysis section below): the same trained models, combined at prediction
# time. The run names start with "improve_" so their measurements are not mixed up with the trained models.
OWN_IMPROVED = [
    Config("Own CNN (64px) + flip TTA", "Own 64+TTA", "improve_tta_own_cnn_64", [OWN_64], flip_tta=True),
    Config("Own CNN (48px) + flip TTA", "Own 48+TTA", "improve_tta_own_cnn_48", [OWN_48], flip_tta=True),
    Config("Own CNN ensemble (64+48px)", "Own ens", "improve_ensemble_own_cnn", [OWN_64, OWN_48]),
    Config("Own CNN ensemble (64+48px) + flip TTA", "Own ens+TTA", "improve_ensemble_tta_own_cnn",
           [OWN_64, OWN_48], flip_tta=True),
]
OWN_CANDIDATES = OWN_TRAINED + OWN_IMPROVED

STANDARD_TRAINED = [
    Config("ResNet-18", "ResNet", standard.RESNET_RUN, [RESNET]),
    Config("MobileNetV3-Large", "MobileNet", standard.MOBILENET_RUN, [MOBILENET]),
]
# Flip TTA is applied to the standard models too, so the comparison stays fair: our own model may not use a
# prediction-time technique that the reference models are not given. MAE_best is the best of ALL of these.
STANDARD_IMPROVED = [
    Config("ResNet-18 + flip TTA", "ResNet+TTA", "improve_tta_resnet18", [RESNET], flip_tta=True),
    Config("MobileNetV3-Large + flip TTA", "MobileNet+TTA", "improve_tta_mobilenetv3", [MOBILENET],
           flip_tta=True),
]
STANDARD_MODELS = STANDARD_TRAINED + STANDARD_IMPROVED
IMPROVED = OWN_IMPROVED + STANDARD_IMPROVED   # configurations measured in the analysis section
ALL_CONFIGS = OWN_CANDIDATES + STANDARD_MODELS


def load_summary(run_name):
    with open(os.path.join(MODEL_DIR, f"{run_name}_summary.json"), encoding="utf-8") as file:
        return json.load(file)


# Our own final model, chosen on VALIDATION results only and fixed BEFORE the test set was opened:
# the ensemble of the 64px and 48px models, without flip TTA. Of our configurations only the ensemble and the
# ensemble + flip TTA meet the threshold on validation (1.15 x MAE_best, MAE_best = ResNet-18 + flip TTA).
# Ensemble + TTA has a lower val MAE (5.31 vs 5.44), but needs twice the computation per image (4 network
# passes instead of 2): about twice the batch-1 latency and inference energy. In a green project the
# configuration that meets the requirement with less computation is chosen.
OWN_FINAL_RUN = "improve_ensemble_own_cnn"


def select_own_final():
    """Our own final model (OWN_FINAL_RUN; the choice and its reasons are given above)."""
    return next(config for config in OWN_CANDIDATES if config.run_name == OWN_FINAL_RUN)


def load_parts(config):
    """Loads the trained models of a configuration: [(model, checkpoint, input size)]."""
    parts = []
    for run_name, build_model, input_size in config.parts:
        saved = torch.load(os.path.join(MODEL_DIR, f"{run_name}.pt"), weights_only=False)
        model = build_model()
        model.load_state_dict(saved["model_state"])
        parts.append((model, saved, input_size))
    return parts


def config_images(config, samples, cache_name):
    """The images at every input size the configuration needs: {input size: uint8 tensor (N, H, W, 3)}."""
    return {size: torch.from_numpy(load_images(samples, size, cache_name)) for _, _, size in config.parts}


def predict_config(config, parts, images):
    """Prediction of a configuration: the mean over its models and, with flip TTA, over each image and its mirror."""
    predictions = []
    for model, saved, input_size in parts:
        x = images[input_size]
        predictions.append(predict(model, x, saved["mean"], saved["std"]))
        if config.flip_tta:
            # (N, H, W, 3): dimension 2 is the width -> horizontal mirror, the same flip as in training
            predictions.append(predict(model, x.flip(dims=[2]), saved["mean"], saved["std"]))
    return np.mean(predictions, axis=0)


def evaluate_on_test(config):
    """Evaluates the final configuration ONCE on the test set (energy measured); saves the result or reads it from the saved file."""
    result_path = os.path.join(TEST_RESULTS_DIR, f"{config.run_name}.json")
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as file:
            return json.load(file)

    parts = load_parts(config)  # models are loaded before the test set is opened (a missing file fails here)
    test = load_test_once(config.run_name)  # the usage is written to data/test_usage.json; it cannot be opened a second time
    try:
        images = config_images(config, test, "test")  # the cache is prepared before measuring
        predictions, energy = measure_energy(
            f"{config.run_name}_test", lambda: predict_config(config, parts, images))
        result = {"model": config.name, "run_name": config.run_name, "n_test": len(test),
                  **regression_metrics(test["age"], predictions), **energy}

        os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
        test.assign(prediction=predictions).to_csv(
            os.path.join(TEST_RESULTS_DIR, f"{config.run_name}_predictions.csv"), index=False)
        with open(result_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
    except BaseException:
        # Crashed before the result was saved: nothing was shown, so the usage record is taken back.
        release_test_use(config.run_name)
        raise
    return result


def model_metrics(config):
    """Test (once) or validation metrics depending on USE_TEST_SET: {MAE, RMSE, R2}."""
    if USE_TEST_SET:
        return evaluate_on_test(config)
    summary = load_summary(config.run_name)
    return {"MAE": summary["val_MAE"], "RMSE": summary["val_RMSE"], "R2": summary["val_R2"]}


def training_co2_g(config):
    """Training emissions of a configuration (g): sum of the final trainings of the models it uses."""
    return sum(load_summary(run_name)["train_co2_g"] for run_name, _, _ in config.parts)


def split_label():
    return "TEST" if USE_TEST_SET else "VALIDATION (pre-check, test set not opened)"


def split_path(path):
    """Adds the evaluation set to an output file name (_val / _test), so the test run keeps the validation files."""
    stem, extension = os.path.splitext(path)
    return f"{stem}_{'test' if USE_TEST_SET else 'val'}{extension}"


# =====================================================================================
# [x] MAE_own ≤ 1.15 × MAE_best (MAE of the best standard model)
# =====================================================================================
# - MAE_own : MAE of our own final CNN (select_own_final: the 64+48px ensemble, fixed on validation before
#             the test set was opened).
# - MAE_best: the lowest MAE among the standard models (ResNet-18, MobileNetV3-Large), with and without flip
#   TTA: a prediction-time technique used for our own model is also given to the reference models.
# - Both are measured on the same set (test or validation depending on USE_TEST_SET) and with the same
#   regression_metrics(). Condition: MAE_own / MAE_best ≤ 1.15.

MAX_RATIO_TO_BEST = 1.15


def check_ratio_to_best():
    """Checks the MAE ratio of our own CNN relative to the best standard model."""
    own = select_own_final()
    own_mae = model_metrics(own)["MAE"]
    standard_mae = {model.name: model_metrics(model)["MAE"] for model in STANDARD_MODELS}
    best_name = min(standard_mae, key=standard_mae.get)
    mae_best = standard_mae[best_name]

    rows = [["Model", "MAE", "MAE / MAE_best"]]
    rows.append([f"{own.name}  <- our own final model", f"{own_mae:.3f}", f"{own_mae / mae_best:.3f}"])
    for name, mae in standard_mae.items():
        rows.append([name + ("  <- MAE_best" if name == best_name else ""), f"{mae:.3f}", f"{mae / mae_best:.3f}"])
    print_table(f"MAE_own ≤ {MAX_RATIO_TO_BEST} × MAE_best  —  {split_label()}:", rows)

    passed = own_mae <= MAX_RATIO_TO_BEST * mae_best
    print(f"""
  Our own final model : {own.name} (fixed on validation before the test set was opened)
  Threshold           : {MAX_RATIO_TO_BEST} × {mae_best:.3f} = {MAX_RATIO_TO_BEST * mae_best:.3f}
  MAE_own             : {own_mae:.3f}  ->  ratio {own_mae / mae_best:.3f}
  Result              : {'MET' if passed else 'NOT MET'}"""
          + ("" if passed else f" (exceeds the threshold by {own_mae - MAX_RATIO_TO_BEST * mae_best:.3f} years)"))
    return passed


# =====================================================================================
# [x] MAE_own ≤ 0.6 × MAE_mean (MAE of the mean baseline)
# =====================================================================================
# - MAE_mean: MAE of the baseline that predicts the mean of the train ages for every image
#   (the assignment's definition; chosen in step 1). The constant is computed from the train set only.
# - Measured on the same set as MAE_own: validation in the pre-check, test in the final evaluation.
# - In test mode the baseline also opens the test set only once under its own name (baseline_mean);
#   the result and predictions are saved under models/test_results/ in the same way as the models'.
# - Condition: MAE_own / MAE_mean ≤ 0.6.

MAX_RATIO_TO_MEAN = 0.6
BASELINE_RUN = "baseline_mean"


def baseline_metrics():
    """Metrics of the mean baseline: computed on validation or evaluated once on test."""
    mean_age = train_mean_age()
    if not USE_TEST_SET:
        val = load_split("val")
        return regression_metrics(val["age"], np.full(len(val), mean_age))

    result_path = os.path.join(TEST_RESULTS_DIR, f"{BASELINE_RUN}.json")
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as file:
            return json.load(file)

    test = load_test_once(BASELINE_RUN)  # the usage is written to data/test_usage.json; it cannot be opened a second time
    predictions, energy = measure_energy(f"{BASELINE_RUN}_test", lambda: np.full(len(test), mean_age))
    result = {"model": "Mean baseline", "run_name": BASELINE_RUN, "prediction": mean_age,
              "n_test": len(test), **regression_metrics(test["age"], predictions), **energy}

    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    test.assign(prediction=predictions).to_csv(
        os.path.join(TEST_RESULTS_DIR, f"{BASELINE_RUN}_predictions.csv"), index=False)
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def check_ratio_to_mean():
    """Checks the MAE ratio of our own CNN relative to the mean baseline."""
    own = select_own_final()
    own_mae = model_metrics(own)["MAE"]
    mae_mean = baseline_metrics()["MAE"]

    rows = [["Model", "MAE", "MAE / MAE_mean"],
            [f"{own.name}  <- our own final model", f"{own_mae:.3f}", f"{own_mae / mae_mean:.3f}"],
            [f"Mean baseline (always {train_mean_age():.2f} years)  <- MAE_mean", f"{mae_mean:.3f}", "1.000"]]
    print_table(f"MAE_own ≤ {MAX_RATIO_TO_MEAN} × MAE_mean  —  {split_label()}:", rows)

    passed = own_mae <= MAX_RATIO_TO_MEAN * mae_mean
    print(f"""
  Threshold : {MAX_RATIO_TO_MEAN} × {mae_mean:.3f} = {MAX_RATIO_TO_MEAN * mae_mean:.3f}
  MAE_own   : {own_mae:.3f}  ->  ratio {own_mae / mae_mean:.3f}
  Result    : {'MET' if passed else 'NOT MET'}"""
          + (f" ({MAX_RATIO_TO_MEAN * mae_mean - own_mae:.3f} years below the threshold)" if passed
             else f" (exceeds the threshold by {own_mae - MAX_RATIO_TO_MEAN * mae_mean:.3f} years)"))
    return passed


# =====================================================================================
# [ ] If the model does not meet these conditions, report it honestly and analyze the reason.
# =====================================================================================
# 20 epochs are not enough. At epoch 20 our own CNN was still improving. The results will be reported
# after retrying with 40 epochs.
#
# Result of the 40-epoch retry (validation): the hypothesis did not hold for our own CNN. Both variants stopped
# early long before epoch 40 (64px: best epoch 14 of 22, 48px: best epoch 20 of 28) and their val MAE stayed
# at ~5.81 (20 epochs: 5.810 / 5.808). The extra epochs only cost energy. The standard models gained a little
# (ResNet-18 4.966 -> 4.937, MobileNetV3-Large 5.269 -> 5.188), so MAE_own / MAE_best = 1.177 > 1.15.
# Observation: the val MAE of our own CNN jumps strongly between epochs (64px: 7.45 at epoch 13, 5.81 at
# epoch 14), so a single checkpoint is a noisy estimate of the model.
#
# -------------------------------------------------------------------------------------
# Improvement without retraining: flip test-time augmentation (TTA) and ensemble (measured)
# -------------------------------------------------------------------------------------
# Both reduce the variance of the predictions without any new training energy; they only make inference
# more expensive (each image passes through the network 2 to 4 times):
#   - Flip TTA : the model predicts the image and its horizontal mirror; the two predictions are averaged.
#                The model was trained with random horizontal flips, so the mirror is a valid input.
#   - Ensemble : the already trained 64px and 48px models predict and their predictions are averaged.
# Flip TTA is applied to ResNet-18 and MobileNetV3-Large as well (STANDARD_IMPROVED), so MAE_best is taken
# under the same prediction-time conditions as our own model.
# The configurations are defined in IMPROVED (top of the file). Each one is evaluated on validation only
# (the test set is not touched): the val set is predicted INFERENCE_REPEATS times, measured with CodeCarbon
# (emissions.csv: improve_<...>_infer), the same way as the <run>_infer runs of the trained models, so the
# ms/image and mWh/1000 values are comparable. The result is saved to models/<run name>_summary.json; if the
# file exists the measurement is not repeated. The configurations appear in the checks, the results table
# and the plot; our own final model is chosen among them (OWN_FINAL_RUN, above select_own_final).


def measure_improvement(config):
    """Evaluates a no-retraining configuration on validation (energy measured) and saves its summary."""
    summary_path = os.path.join(MODEL_DIR, f"{config.run_name}_summary.json")
    if os.path.exists(summary_path):
        return load_summary(config.run_name)

    val = load_split("val")
    images = config_images(config, val, "val")  # images and models are prepared before measuring
    parts = load_parts(config)
    predictions, energy = measure_energy(
        f"{config.run_name}_infer",
        lambda: [predict_config(config, parts, images) for _ in range(INFERENCE_REPEATS)][0])
    metrics = regression_metrics(val["age"], predictions)
    n_predictions = len(val) * INFERENCE_REPEATS

    summary = {
        "run_name": config.run_name, "parts": [run_name for run_name, _, _ in config.parts],
        "flip_tta": config.flip_tta,
        "val_MAE": metrics["MAE"], "val_RMSE": metrics["RMSE"], "val_R2": metrics["R2"],
        "train_co2_g": training_co2_g(config),  # no new training: the trainings of the models it uses
        "infer_ms_per_image": 1000 * energy["seconds"] / n_predictions,
        "infer_mWh_per_1000": 1e6 * energy["energy_Wh"] / n_predictions,
        "measurement_energy_Wh": energy["energy_Wh"], "measurement_co2_g": energy["co2_g"],
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary


def report_improvements():
    """Measures the no-retraining configurations and compares them with the trained models (validation)."""
    for config in IMPROVED:  # measured first: MAE_best also depends on the standard models with flip TTA
        measure_improvement(config)
    mae_best = min(load_summary(model.run_name)["val_MAE"] for model in STANDARD_MODELS)
    threshold = MAX_RATIO_TO_BEST * mae_best

    rows = [["Configuration", "Val MAE", "Change vs base", "RMSE", "R²", "ms/img", "mWh / 1000 img",
             f"≤ {threshold:.2f}?"]]
    for config in ALL_CONFIGS:
        summary = load_summary(config.run_name)
        base = load_summary(config.parts[0][0])  # the trained model it is built on (ensemble: the 64px model)
        rows.append([config.name, f"{summary['val_MAE']:.3f}", f"{summary['val_MAE'] - base['val_MAE']:+.3f}",
                     f"{summary['val_RMSE']:.2f}", f"{summary['val_R2']:.3f}",
                     f"{summary['infer_ms_per_image']:.3f}", f"{summary['infer_mWh_per_1000']:.2f}",
                     "yes" if summary["val_MAE"] <= threshold else "no"])
    print_table(f"Improvement without retraining — validation (threshold {MAX_RATIO_TO_BEST} × MAE_best = "
                f"{MAX_RATIO_TO_BEST} × {mae_best:.3f} = {threshold:.3f}; base = the trained model the "
                f"configuration uses, for the ensemble the 64px model):", rows)

    measured = sum(load_summary(config.run_name)["measurement_energy_Wh"] for config in IMPROVED)
    print(f"\n  No new training energy. Energy of these validation measurements: {measured:.2f} Wh "
          f"(emissions.csv: improve_*_infer).\n  The price is paid at inference: see the ms/img and "
          f"mWh / 1000 img columns.")


# -------------------------------------------------------------------------------------
# Per-image latency (batch 1), measured
# -------------------------------------------------------------------------------------
# The ms/img column of the results table is the throughput of batch inference (val set, batch 512): it says how
# much time an image costs when many images are processed together. A deployed model often answers ONE image
# at a time, which is slower per image (less parallel work, fixed per-call overhead). Here each configuration
# predicts LATENCY_IMAGES validation images one by one (batch 1, the same predict_config as everywhere else:
# normalization + every model of the configuration + the mirror pass for flip TTA), and the wall-clock time of
# each image is recorded. Reported: median and 95th percentile (p95) latency, and energy per 1,000 images at
# batch 1. The first LATENCY_WARMUP images are not timed (first calls allocate memory), but they are inside the
# CodeCarbon measurement (emissions.csv: <run name>_latency), so no computation is left unmeasured.
# Results are saved to models/latency/<run name>.json; if the file exists the measurement is not repeated.

LATENCY_DIR = os.path.join(MODEL_DIR, "latency")
LATENCY_IMAGES = 300
LATENCY_WARMUP = 20


def measure_latency(config):
    """Measures the batch-1 latency of a configuration on validation images (energy measured)."""
    path = os.path.join(LATENCY_DIR, f"{config.run_name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    val = load_split("val").iloc[:LATENCY_WARMUP + LATENCY_IMAGES]
    images = config_images(config, val, "val")  # prepared before measuring
    parts = load_parts(config)

    def run():
        times = []
        for i in range(len(val)):
            single = {size: x[i:i + 1] for size, x in images.items()}
            start = time.perf_counter()
            predict_config(config, parts, single)
            times.append(1000 * (time.perf_counter() - start))
        return times[LATENCY_WARMUP:]

    times, energy = measure_energy(f"{config.run_name}_latency", run)
    result = {"run_name": config.run_name, "n_images": LATENCY_IMAGES, "warmup": LATENCY_WARMUP,
              "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
              "mWh_per_1000_batch1": 1e6 * energy["energy_Wh"] / len(val), **energy}
    os.makedirs(LATENCY_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def report_latency():
    """Measures and prints the batch-1 latency of every configuration next to its batch throughput."""
    rows = [["Configuration", "Batch 512: ms/img", "Batch 1: median ms", "Batch 1: p95 ms",
             "Batch 1: mWh / 1000 img"]]
    for config in ALL_CONFIGS:
        latency = measure_latency(config)
        rows.append([config.name, f"{load_summary(config.run_name)['infer_ms_per_image']:.3f}",
                     f"{latency['median_ms']:.2f}", f"{latency['p95_ms']:.2f}",
                     f"{latency['mWh_per_1000_batch1']:.1f}"])
    print_table(f"Per-image latency — {LATENCY_IMAGES} validation images one at a time "
                f"(CPU, {torch.get_num_threads()} threads):", rows)


# =====================================================================================
# [x] Prepare the results table: Model | Params | MFLOPs/img | Test MAE | RMSE | R² | Train kWh |
#     Train kg CO2 | Project kg CO2 | ms/img
# =====================================================================================
# Source of the columns (same code and same definitions as step 4):
#   Params, MFLOPs/img     : footprint.model_size() (FlopCounterMode; FLOPs = 2 x MACs)
#   MAE, RMSE, R²          : model_metrics() -> test (once) or validation depending on USE_TEST_SET
#   Train kWh, Train kg CO2: the emissions.csv row of the final training (footprint.used_row)
#   ms/img                 : the final evaluation run (val set x INFERENCE_REPEATS, batch 512)
#   Latency ms             : median per-image latency at batch 1 (latency section above; <run>_latency)
#   Project kg CO2         : sum of ALL runs belonging to that model (archive + current session; repeated /
#                            overlapping trainings, evaluations and the test included). Every row whose run
#                            name starts with the model's run name belongs to that model. Runs belonging to
#                            no model (footprint-reduction experiments) are given separately below the table,
#                            so the grand total equals the sum of the emissions.csv files.
# Configurations without retraining (flip TTA, ensemble) use already trained models:
#   Params, MFLOPs/img     : sum over the models they use; flip TTA doubles the MFLOPs (two passes per image)
#   Train kWh, Train kg CO2: sum of the final trainings of the models they use (no training of their own)
#   ms/img                 : their own measured run (improve_<...>_infer)
#   Project kg CO2         : only their own runs (improve_<...>); the trainings are counted in the rows of the
#                            trained models, so nothing is counted twice.
# The mean baseline is added as a reference row (no learned parameters, no training).

RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results_table.csv")
RESULTS_MD = os.path.join(RESULTS_DIR, "results_table.md")


def build_results_table():
    """Builds the results table: accuracy, size, energy and emissions per model."""
    emissions = footprint.read_emissions()
    all_runs = footprint.all_sessions(emissions)
    n_predictions = len(load_split("val")) * footprint.INFERENCE_REPEATS
    own_final = select_own_final().run_name
    mae_label = "Test" if USE_TEST_SET else "Val"

    records, attributed = [], pd.Series(False, index=all_runs.index)
    for config in ALL_CONFIGS:
        metrics = model_metrics(config)
        sizes = [footprint.model_size(build_model, input_size, os.path.join(MODEL_DIR, f"{run_name}.pt"))
                 for run_name, build_model, input_size in config.parts]
        trainings = [footprint.used_row(emissions, run_name) for run_name, _, _ in config.parts]
        infer = footprint.used_row(emissions, f"{config.run_name}_infer")
        # Rows whose run name is exactly this configuration or starts with "<run name>_" (own_cnn_64 and
        # own_cnn_48 do not get mixed up).
        own_rows = ((all_runs["project_name"] == config.run_name)
                    | all_runs["project_name"].str.startswith(f"{config.run_name}_"))
        attributed |= own_rows
        records.append({
            "Model": config.name + (" (final)" if config.run_name == own_final else ""),
            "Params": sum(size[0] for size in sizes),
            "MFLOPs/img": sum(size[1] for size in sizes) * (2 if config.flip_tta else 1),
            f"{mae_label} MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R²": metrics["R2"],
            "Train kWh": sum(row["energy_consumed"] for row in trainings),
            "Train kg CO2": sum(row["emissions"] for row in trainings),
            "Project kg CO2": all_runs.loc[own_rows, "emissions"].sum(),
            "ms/img": 1000 * infer["duration"] / n_predictions,
            "Latency ms": measure_latency(config)["median_ms"],
        })

    baseline = baseline_metrics()
    baseline_rows = all_runs["project_name"].str.startswith(BASELINE_RUN)
    attributed |= baseline_rows
    records.append({"Model": "Mean baseline", "Params": 0, "MFLOPs/img": 0.0,
                    f"{mae_label} MAE": baseline["MAE"], "RMSE": baseline["RMSE"], "R²": baseline["R2"],
                    "Train kWh": 0.0, "Train kg CO2": 0.0,
                    "Project kg CO2": all_runs.loc[baseline_rows, "emissions"].sum(), "ms/img": 0.0,
                    "Latency ms": 0.0})

    other = all_runs.loc[~attributed, "emissions"].sum()
    return pd.DataFrame(records), other, all_runs["emissions"].sum()


def report_results_table():
    """Prints the results table and saves it under results/ as CSV and Markdown."""
    table, other_kg, total_kg = build_results_table()
    formats = {"Params": "{:,.0f}", "MFLOPs/img": "{:.1f}", "RMSE": "{:.2f}", "R²": "{:.3f}",
               "Train kWh": "{:.4f}", "Train kg CO2": "{:.4f}", "Project kg CO2": "{:.4f}", "ms/img": "{:.3f}",
               "Latency ms": "{:.2f}"}
    mae_column = [column for column in table.columns if column.endswith("MAE")][0]
    formats[mae_column] = "{:.2f}"

    formatted = table.copy()
    for column, pattern in formats.items():
        formatted[column] = table[column].map(pattern.format)
    formatted.loc[formatted["Model"] == "Mean baseline", ["Params", "MFLOPs/img", "Train kWh",
                                                          "Train kg CO2", "ms/img", "Latency ms"]] = "-"

    print_table(f"Results table — {split_label()}:", [list(formatted.columns)] + formatted.values.tolist())
    note = (f"ms/img: batch 512 throughput; Latency ms: median at batch 1. "
            f"Runs belonging to no model (footprint-reduction experiments, bias-analysis predictions): "
            f"{other_kg:.4f} kg CO2eq. Project total (all sessions): {total_kg:.4f} kg CO2eq.")
    print(f"\n  {note}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(split_path(RESULTS_CSV), index=False)
    with open(split_path(RESULTS_MD), "w", encoding="utf-8") as file:
        file.write(f"Results table — {split_label()}\n\n")
        file.write("| " + " | ".join(formatted.columns) + " |\n")
        file.write("|" + "|".join("---" for _ in formatted.columns) + "|\n")
        for row in formatted.values.tolist():
            file.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
        file.write(f"\n{note}\n")
    print(f"Saved: {split_path(RESULTS_CSV)}\n       {split_path(RESULTS_MD)}")


# =====================================================================================
# [x] Draw the error–kg CO2eq plot: CO2 axis on a log scale, each model or configuration one point,
#     Pareto front marked.
# =====================================================================================
# - x: emissions of the final training (kg CO2eq, log scale), y: MAE (test or val depending on USE_TEST_SET).
# - Each model / configuration is one point: own CNN 64px and 48px, their no-retraining improvements (flip TTA,
#   ensemble), ResNet-18, MobileNetV3-Large. A configuration without retraining sits at the training
#   emissions of the models it uses (flip TTA: same x as its model); its extra inference cost is in the
#   ms/img column of the results table.
# - Labels are single lines; a label that would overlap a label above it is moved down and connected to its
#   point with a thin line (label_offsets).
# - Pareto front: points for which no other point has both a lower error and lower emissions
#   (both should be small). Pareto points are drawn filled blue with a step line, non-Pareto points as
#   hollow gray; the distinction is made not only by color but also by shape.
# - Dashed line: the 1.15 x MAE_best threshold. The mean baseline (MAE ≈ 15) is not drawn because it would
#   squash the y axis; it is mentioned in the caption.

FIGURE_DIR = os.path.join(ROOT, "figures")
FIGURE_PATH = os.path.join(FIGURE_DIR, "error_vs_co2.png")
PLOTTED_MODELS = OWN_CANDIDATES + STANDARD_MODELS

COLORS = {"pareto": "#2a78d6", "dominated": "#898781", "surface": "#fcfcfb", "ink": "#0b0b0b",
          "ink_secondary": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7"}
LABEL_FONTSIZE = 8
LABEL_DX = 7                              # labels start this far to the right of their point (points)
LABEL_DY_CANDIDATES = [5, -13, 17, -25, 29, -37]  # baseline height relative to the point (points), in order of
                                                  # preference: just above (the horizontal Pareto line drawn to
                                                  # the right of a point does not cross it), just below, ...
MARKER_RADIUS = 5                         # half the size of a marker (points)


def overlaps(a, b):
    """Do two boxes (left, right, bottom, top) overlap?"""
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def label_offsets(axis, xs, ys, labels):
    """Vertical offset of each label (points) so that no label overlaps a marker or another label.

    Labels are placed from the top point to the bottom one; each label takes the first position in
    LABEL_DY_CANDIDATES whose box overlaps no marker and no label placed before. The text width is
    estimated from its length.
    """
    char_width = 0.6 * LABEL_FONTSIZE
    points = axis.transData.transform(np.column_stack([xs, ys])) * 72 / axis.figure.dpi  # pixels -> points
    obstacles = [(x - MARKER_RADIUS, x + MARKER_RADIUS, y - MARKER_RADIUS, y + MARKER_RADIUS) for x, y in points]
    offsets = [0] * len(labels)
    for i in sorted(range(len(labels)), key=lambda i: -points[i][1]):
        x, y = points[i]
        for dy in LABEL_DY_CANDIDATES:
            # text box: from a little below the baseline (descenders) to the top of the capitals
            box = (x + LABEL_DX, x + LABEL_DX + char_width * len(labels[i]), y + dy - 2, y + dy + 7)
            if not any(overlaps(box, other) for other in obstacles):
                break
        offsets[i] = dy
        obstacles.append(box)
    return offsets


def pareto_front(points):
    """Indices of the Pareto points (both emissions and MAE should be small)."""
    front, best_mae = [], float("inf")
    for index in sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1])):
        if points[index][1] < best_mae:
            front.append(index)
            best_mae = points[index][1]
    return front


def plot_error_vs_co2():
    """Draws the error – kg CO2eq plot, marks the Pareto front and saves it as PNG."""
    names = [config.name + (" (final)" if config.run_name == OWN_FINAL_RUN else "") for config in PLOTTED_MODELS]
    co2 = [training_co2_g(config) / 1000 for config in PLOTTED_MODELS]
    mae = [model_metrics(config)["MAE"] for config in PLOTTED_MODELS]
    front = pareto_front(list(zip(co2, mae)))
    mae_best = min(model_metrics(config)["MAE"] for config in STANDARD_MODELS)

    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=150)
    figure.patch.set_facecolor(COLORS["surface"])
    axis.set_facecolor(COLORS["surface"])

    # Pareto front: step line between the points sorted by emissions
    front_points = sorted((co2[i], mae[i]) for i in front)
    axis.step([p[0] for p in front_points], [p[1] for p in front_points], where="post",
              color=COLORS["pareto"], linewidth=2, zorder=2, label="Pareto front")

    for i in range(len(names)):
        on_front = i in front
        axis.scatter(co2[i], mae[i], s=60, zorder=3, linewidths=2,
                     facecolors=COLORS["pareto"] if on_front else "none",
                     edgecolors=COLORS["pareto"] if on_front else COLORS["dominated"])

    axis.axhline(MAX_RATIO_TO_BEST * mae_best, color=COLORS["ink_secondary"], linestyle="--", linewidth=1,
                 zorder=1, label=f"{MAX_RATIO_TO_BEST} × MAE_best = {MAX_RATIO_TO_BEST * mae_best:.2f}")
    axis.scatter([], [], s=80, facecolors=COLORS["pareto"], edgecolors=COLORS["pareto"], label="On the Pareto front")
    axis.scatter([], [], s=80, facecolors="none", edgecolors=COLORS["dominated"], linewidths=2,
                 label="Not on the Pareto front")

    axis.set_xscale("log")
    axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))  # 0.001, 0.002, 0.005, 0.01 ...
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(FuncFormatter(lambda _value, _: ""))
    axis.set_xlim(min(co2) / 1.6, max(co2) * 4)  # room on the right for the labels
    axis.set_ylim(min(mae) - 0.2, max(max(mae), MAX_RATIO_TO_BEST * mae_best) + 0.15)
    axis.set_xlabel("Emissions of the final training (kg CO2eq, log scale)", color=COLORS["ink_secondary"])
    axis.set_ylabel(f"MAE (years), {'test' if USE_TEST_SET else 'validation'}", color=COLORS["ink_secondary"])
    axis.set_title("Error and carbon footprint: which models are on the Pareto front?", color=COLORS["ink"],
                   loc="left", fontsize=12)
    axis.grid(True, which="major", color=COLORS["grid"], linewidth=0.6)
    axis.tick_params(colors=COLORS["ink_secondary"], labelsize=9)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(COLORS["axis"])
    axis.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=COLORS["ink_secondary"])
    figure.text(0.01, 0.01, f"MAE of the mean baseline: {baseline_metrics()['MAE']:.2f} (outside the plot). "
                f"Error: {split_label().lower()}. Emissions: CodeCarbon, TUR.\n"
                f"Flip TTA / ensemble: no new training; x = training emissions of the models they use.",
                fontsize=7.5, color=COLORS["ink_secondary"])
    figure.tight_layout(rect=(0, 0.05, 1, 1))

    # Labels are placed after the layout is fixed, because their overlap is computed in screen coordinates.
    labels = [f"{name} · MAE {mae[i]:.2f} · {1000 * co2[i]:.1f} g" for i, name in enumerate(names)]
    for i, dy in enumerate(label_offsets(axis, co2, mae, labels)):
        far = dy not in LABEL_DY_CANDIDATES[:2]  # a label far from its point is connected to it with a thin line
        axis.annotate(labels[i], (co2[i], mae[i]), textcoords="offset points", xytext=(LABEL_DX, dy),
                      fontsize=LABEL_FONTSIZE, color=COLORS["ink"], zorder=4,
                      # background in the plot color: a Pareto line passing behind the label does not cross the text
                      bbox=dict(boxstyle="square,pad=0.1", facecolor=COLORS["surface"], edgecolor="none"),
                      arrowprops=dict(arrowstyle="-", color=COLORS["axis"], linewidth=0.7,
                                      shrinkA=0, shrinkB=4) if far else None)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    figure.savefig(split_path(FIGURE_PATH), facecolor=COLORS["surface"])
    plt.close(figure)

    print(f"Models on the Pareto front: {', '.join(names[i] for i in sorted(front, key=lambda i: co2[i]))}")
    print(f"Models not on the Pareto front: {', '.join(names[i] for i in range(len(names)) if i not in front) or '-'}")
    print(f"Saved: {split_path(FIGURE_PATH)}")


# -------------------------------------------------------------------------------------
# Bias analysis per model (group errors, as in step 1)
# -------------------------------------------------------------------------------------
# Step 1 reported the error of the mean baseline by gender, ethnicity and age group. Here the same
# group_errors() is applied to the models' predictions, on the same set as the other results (validation in the
# pre-check, test in the final evaluation). Compared: our own final configuration, our own 64px model without
# prediction-time techniques, and the standard models with and without flip TTA.
#   - Cells: "MAE / Bias" of the group. Bias = mean(prediction - true age): + the group is predicted older,
#     - younger than it is.
#   - "Gap" row: largest minus smallest group MAE (how unequal the error is across the groups).
# Predictions: validation -> one measured run (emissions.csv: bias_val_predictions) predicts the val set with
#   the compared configurations; saved to models/bias/val_predictions.csv (only missing models are predicted
#   again). Test -> the predictions saved by the one-time test evaluation (models/test_results/).
# Every group of every model is written to results/bias_by_group_val.csv (or _test.csv).

BIAS_DIR = os.path.join(MODEL_DIR, "bias")
BIAS_VAL_PREDICTIONS = os.path.join(BIAS_DIR, "val_predictions.csv")
BIAS_CSV = os.path.join(RESULTS_DIR, "bias_by_group.csv")
GROUPINGS = ["Gender", "Ethnicity", "Age group"]


def bias_models():
    """Configurations compared in the bias analysis: our own final first, without duplicates."""
    models = []
    for config in [select_own_final(), OWN_TRAINED[0]] + STANDARD_MODELS:
        if config.run_name not in [model.run_name for model in models]:
            models.append(config)
    return models


def bias_predictions(models):
    """The samples and each model's predictions on them: (samples, {run name: predictions})."""
    if USE_TEST_SET:
        tables = {}
        for config in models:
            model_metrics(config)  # makes sure the one-time test evaluation has been done and saved
            tables[config.run_name] = pd.read_csv(
                os.path.join(TEST_RESULTS_DIR, f"{config.run_name}_predictions.csv"))
        samples = tables[models[0].run_name]  # all files hold the same test samples in the same order
        return samples, {run_name: table["prediction"].to_numpy() for run_name, table in tables.items()}

    val = load_split("val")
    saved = pd.read_csv(BIAS_VAL_PREDICTIONS) if os.path.exists(BIAS_VAL_PREDICTIONS) else pd.DataFrame(index=val.index)
    missing = [config for config in models if config.run_name not in saved.columns]
    if missing:
        inputs = [(config, load_parts(config), config_images(config, val, "val")) for config in missing]
        predictions, _ = measure_energy("bias_val_predictions", lambda: {
            config.run_name: predict_config(config, parts, images) for config, parts, images in inputs})
        saved = saved.assign(**predictions)
        os.makedirs(BIAS_DIR, exist_ok=True)
        saved.to_csv(BIAS_VAL_PREDICTIONS, index=False)
    return val, {config.run_name: saved[config.run_name].to_numpy() for config in models}


def report_bias():
    """Prints the error and bias of each model by gender, ethnicity and age group; saves all groups to CSV."""
    models = bias_models()
    samples, predictions = bias_predictions(models)
    tables = {config.run_name: group_errors(samples, predictions[config.run_name]) for config in models}
    labels = [config.short + (" (final)" if index == 0 else "") for index, config in enumerate(models)]

    records = []
    for grouping in GROUPINGS:
        groups = tables[models[0].run_name][grouping]
        rows = [[grouping, "n"] + labels]
        for group in groups.index:
            rows.append([group, int(groups.loc[group, "n"])] + [
                f"{tables[config.run_name][grouping].loc[group, 'MAE']:.2f} / "
                f"{tables[config.run_name][grouping].loc[group, 'Bias']:+.2f}" for config in models])
        gaps = [tables[config.run_name][grouping]["MAE"] for config in models]
        rows.append(["Gap (max - min MAE)", ""] + [f"{mae.max() - mae.min():.2f}" for mae in gaps])
        print_table(f"{grouping}: MAE / Bias (years) — {split_label()}:", rows)

        for config in models:
            for group, row in tables[config.run_name][grouping].iterrows():
                records.append({"Model": config.name, "Grouping": grouping, "Group": group, **row.to_dict()})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(records).to_csv(split_path(BIAS_CSV), index=False)
    print(f"\n  Bias = mean(prediction - true age): + predicted older, - predicted younger.\n"
          f"Saved: {split_path(BIAS_CSV)}")


def main():
    print(f"Evaluation set: {split_label()}")

    # Runs first: its configurations are candidates for our own final model in all later sections.
    print_section("3) Analysis: improvement without retraining (flip TTA, ensemble; validation, measured)")
    report_improvements()

    print_section("1) MAE_own ≤ 1.15 × MAE_best")
    check_ratio_to_best()

    print_section("2) MAE_own ≤ 0.6 × MAE_mean")
    check_ratio_to_mean()

    print_section("4a) Per-image latency (batch 1, measured)")
    report_latency()

    print_section("4) Results table")
    report_results_table()

    print_section("5) Error – kg CO2eq plot (Pareto front)")
    plot_error_vs_co2()

    print_section("6) Bias analysis per model (gender, ethnicity, age group)")
    report_bias()


if __name__ == "__main__":
    main()
