"""
Step 4 — Carbon footprint report.

Usage:
    python 4_footprint.py

Does not do any new training; it only reads the saved results (the footprint-reduction
measurements below are done once, on the first run):
    emissions/emissions.csv   : all runs written by CodeCarbon (single source)
    emissions/run_notes.csv   : the measurement status of each run and which one is used in the summary
    models/<run>_summary.json, models/<run>.pt : error metrics and model weights
"""

import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from data_utils import (ROOT, SEED, load_split, load_images, decode_images, regression_metrics,
                        print_section, print_table, load_step_module)
from train_utils import (MODEL_DIR, EMISSIONS_CSV, INFERENCE_REPEATS, MAX_EPOCHS, EARLY_STOP_PATIENCE,
                         LEARNING_RATE, measure_energy, predict, load_train_val, train_model)

RUN_NOTES_CSV = os.path.join(ROOT, "emissions", "run_notes.csv")
FOOTPRINT_CSV = os.path.join(ROOT, "emissions", "footprint_report.csv")


own_cnn = load_step_module("2_own_cnn.py", "own_cnn")
standard = load_step_module("3_comparing_standard_models.py", "standard_models")

MODELS = [
    # (name, run name, model builder, input size)
    ("Own CNN", "own_cnn_64", lambda: own_cnn.AgeCNN(), 64),
    ("Own CNN (ablation 48px)", "own_cnn_48", lambda: own_cnn.AgeCNN(), 48),
    ("ResNet-18", standard.RESNET_RUN, standard.build_resnet18, 64),
    ("MobileNetV3-Large", standard.MOBILENET_RUN, standard.build_mobilenet_v3_large, 64),
]


# =====================================================================================
# [x] Measure every training and evaluation run with CodeCarbon (country_iso_code="TUR").
#     Hyperparameter searches and failed trials are included.
# =====================================================================================
# Measurement is done with train_utils.measure_energy() (country_iso_code="TUR"); every training and
# evaluation run is written to emissions/emissions.csv as one row, including runs that end with an error
# (tracker.stop() is inside finally). Repeated and overlapping runs also stay in the file.
# This report uses only the CodeCarbon records in emissions.csv. The first trials made before the
# measurement infrastructure was set up are not in this file.


ARCHIVE_DIR = os.path.join(ROOT, "archive")
NOTE_COLUMNS = ["measurement_status", "power_mode", "used_in_summary", "notes"]


def read_emissions(emissions_csv=EMISSIONS_CSV, notes_csv=RUN_NOTES_CSV):
    """Reads a session's emissions.csv, merged with the status information in run_notes.csv if present."""
    emissions = pd.read_csv(emissions_csv)
    if os.path.exists(notes_csv):
        return emissions.merge(pd.read_csv(notes_csv), on=["timestamp", "project_name"], how="left")
    # None: a run without a note. pd.NA is not used; comparing with NA returns NA instead of True/False.
    return emissions.assign(**{column: None for column in NOTE_COLUMNS})


def read_archived_sessions():
    """Reads the previous measurement sessions under archive/<session>/emissions as {session name: table}."""
    sessions = {}
    if os.path.isdir(ARCHIVE_DIR):
        for session in sorted(os.listdir(ARCHIVE_DIR)):
            folder = os.path.join(ARCHIVE_DIR, session, "emissions")
            if os.path.exists(os.path.join(folder, "emissions.csv")):
                sessions[session] = read_emissions(os.path.join(folder, "emissions.csv"),
                                                   os.path.join(folder, "run_notes.csv"))
    return sessions


def used_row(emissions, project_name):
    """The emissions.csv row of the run used in the summary.

    If it is marked in run_notes.csv (used_in_summary = yes) that row is used, otherwise the latest row
    with that name: the summary file is always written by the latest run with that name.
    """
    rows = emissions[emissions["project_name"] == project_name]
    marked = rows[rows["used_in_summary"] == "yes"]
    return (marked if len(marked) else rows).iloc[-1]


def measurement_note(row):
    """The measurement status note of a run; a short explanation for runs without a note (new session)."""
    if pd.isna(row["measurement_status"]):
        return "no note (ran alone)"
    return f"{row['measurement_status']}, {row['power_mode']}"


# =====================================================================================
# [x] Report the following for each model:
# =====================================================================================
# The five items below are collected into a single table per model and saved to
# emissions/footprint_report.csv. The energy values are always read from the run's emissions.csv row.


# -------------------------------------------------------------------------------------
# [x] Energy (kWh) and emissions (kg CO2eq) of the final training
# -------------------------------------------------------------------------------------
# "Final training": the training run used in the model's summary and checkpoint
# (used_in_summary = yes in run_notes.csv). The units are the same as CodeCarbon's:
# energy_consumed kWh, emissions kg CO2eq.


# -------------------------------------------------------------------------------------
# [x] Total energy and emissions of the project (sum of all runs)
# -------------------------------------------------------------------------------------
# The sum of all rows in emissions.csv: besides the final runs, repeated and overlapping runs are
# included because they really consumed energy.


CURRENT_SESSION = "current session"


def all_sessions(emissions):
    """Archived sessions + the current session; the same CodeCarbon run (run_id) is counted only once."""
    tables = [table.assign(session=name) for name, table in read_archived_sessions().items()]
    tables.append(emissions.assign(session=CURRENT_SESSION))
    return pd.concat(tables, ignore_index=True).drop_duplicates(subset="run_id", keep="first")


def report_project_total(emissions):
    """Prints the project's total energy, emissions and time, broken down by session and run type."""
    def run_group(row):
        if row["project_name"].startswith("reduce_"):
            kind = "footprint-reduction experiment"
        elif row["project_name"].endswith("_test"):
            kind = "test evaluation"
        elif row["project_name"].startswith("improve_"):
            kind = "no-retraining improvement (step 5)"
        elif row["project_name"].startswith("bias_"):
            kind = "bias-analysis predictions (step 5)"
        else:
            # A run without a note is an earlier trial if a newer run with the same name exists in the same session.
            superseded = pd.isna(row["used_in_summary"]) and row.name not in latest
            role = "repeated / earlier" if row["used_in_summary"] == "no" or superseded else "final"
            # <run>_infer: batch inference, <run>_latency: batch-1 latency (step 5); both are evaluations
            evaluation = row["project_name"].endswith(("_infer", "_latency"))
            kind = f"{role} {'evaluation' if evaluation else 'training'}"
        return f"{row['session']}: {kind}"

    runs = all_sessions(emissions)
    latest = set(runs.groupby(["session", "project_name"]).tail(1).index)
    runs = runs.assign(group=runs.apply(run_group, axis=1))
    rows = [["Group", "Runs", "Time (min)", "Energy (kWh)", "Emissions (kg CO2eq)"]]
    for group_name, group in runs.groupby("group", sort=False):
        rows.append([group_name, len(group), f"{group['duration'].sum() / 60:.1f}",
                     f"{group['energy_consumed'].sum():.4f}", f"{group['emissions'].sum():.4f}"])
    rows.append(["TOTAL (all sessions)", len(runs), f"{runs['duration'].sum() / 60:.1f}",
                 f"{runs['energy_consumed'].sum():.4f}", f"{runs['emissions'].sum():.4f}"])
    print_table("Total footprint of the project (all archived and current CodeCarbon runs):", rows)

    wasted = runs[runs["group"].str.contains("repeated")]["energy_consumed"].sum()
    print(f"\n  Share of repeated / overlapping runs: {100 * wasted / runs['energy_consumed'].sum():.1f}% "
          f"(energy that did not contribute to the final results)")
    print(f"  Carbon intensity (CodeCarbon, TUR): "
          f"{runs['emissions'].sum() / runs['energy_consumed'].sum():.3f} kg CO2eq / kWh")
    return runs


# -------------------------------------------------------------------------------------
# Comparison of the measurement sessions
# -------------------------------------------------------------------------------------
# In the first session the runs were measured in different power modes and some of them overlapped
# (archive/.../run_notes.csv). The current session was done after the Windows settings were applied
# (plugged in, fixed power mode, sleep off, one run at a time). Putting the measurements of the same run
# in the two sessions side by side shows how much the measurement conditions affect the energy.

def report_session_comparison(runs):
    """Compares the archived and current-session measurements of each final training run."""
    current = runs[runs["session"] == CURRENT_SESSION]
    previous = runs[runs["session"] != CURRENT_SESSION]
    if previous.empty:
        print("No archived session to compare with.")
        return

    def describe(row, label):
        return [row["project_name"], label, measurement_note(row), f"{row['duration'] / 60:.1f}",
                f"{1000 * row['energy_consumed']:.2f}", f"{3.6e6 * row['energy_consumed'] / row['duration']:.1f}"]

    rows = [["Run", "Session", "Measurement status", "Time (min)", "Energy (Wh)", "Avg. power (W)"]]
    for _, run_name, _, _ in MODELS:
        archived = previous[previous["project_name"] == run_name]
        rows.append(describe(used_row(archived, run_name), "session 1 (20 epochs)") if len(archived)
                    else [run_name, "session 1", "no run", "", "", ""])
        # The first run of the current session used 20 epochs, the last one the current MAX_EPOCHS.
        recent = current[current["project_name"] == run_name]
        if len(recent):
            rows.append(describe(recent.iloc[0], "current, first run (20 epochs)"))
        if len(recent) > 1:
            rows.append(describe(recent.iloc[-1], f"current, last run ({MAX_EPOCHS} epochs)"))
    print_table("Energy of the final trainings in the measurement sessions:", rows)
    print("  For the effect of the measurement conditions (Windows settings), compare the 'session 1' and\n"
          "  'current, first run' rows: both use 20 epochs. Read the 'last run' row separately, since the epoch limit also changed.")


# -------------------------------------------------------------------------------------
# [x] Training time, hardware (CPU/GPU, Colab or local) and location
# -------------------------------------------------------------------------------------
# The time is read from each model's final training row (duration). Hardware and location come from the
# columns recorded by CodeCarbon; since all runs were done on the same machine, they are given in one block.


def report_hardware(emissions):
    """Prints the hardware and location from emissions.csv; checks that they are the same in all runs."""
    columns = ["cpu_model", "cpu_count", "gpu_count", "ram_total_size", "os", "country_name",
               "country_iso_code", "on_cloud", "tracking_mode", "codecarbon_version"]
    unique = {column: emissions[column].fillna("").astype(str).unique() for column in columns}
    for column, values in unique.items():
        if len(values) > 1:
            print(f"  Warning: {column} differs between runs: {', '.join(values)}")

    first = emissions.iloc[0]
    gpu = "none (CPU only)" if int(first["gpu_count"]) == 0 else f"{first['gpu_count']} x {first['gpu_model']}"
    environment = "cloud" if first["on_cloud"] == "Y" else "local computer (not Colab / cloud)"
    print(f"""Hardware and location (from the CodeCarbon records, identical in all runs):
  Environment : {environment}
  CPU         : {first['cpu_model']} ({int(first['cpu_count'])} threads)
  GPU         : {gpu}
  RAM         : {first['ram_total_size']:.1f} GB
  OS          : {first['os']}
  Location    : {first['country_name']} ({first['country_iso_code']}); emissions with this country's grid carbon intensity
  Measurement : CodeCarbon {first['codecarbon_version']}, {first['tracking_mode']} mode; CPU energy is measured from the
                Windows RAPL counter, RAM energy is estimated (verification note in train_utils.py)""")


# -------------------------------------------------------------------------------------
# [x] Number of parameters, MFLOPs/image, model file size (MB)
# -------------------------------------------------------------------------------------
# - Parameters: number of trained weights (excluding BN's running mean/var buffers).
# - MFLOPs/image: counted with PyTorch FlopCounterMode for a single image at the model's own input size
#   (FLOPs = 2 x MACs; convolution and Linear layers).
# - Model file size: on-disk size of the weights in the final checkpoint (state_dict, float32).
#   The checkpoint file is slightly larger because it also contains the normalization values.


def model_size(build_model, input_size, checkpoint_path):
    """(parameters, MFLOPs / image, weights file MB, checkpoint file MB)."""
    model = build_model().eval()
    parameters = sum(p.numel() for p in model.parameters())
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size))

    state = torch.load(checkpoint_path, weights_only=False)["model_state"]
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return (parameters, counter.get_total_flops() / 1e6, buffer.getbuffer().nbytes / 1024 ** 2,
            os.path.getsize(checkpoint_path) / 1024 ** 2)


# -------------------------------------------------------------------------------------
# [x] Inference latency per image (ms) and energy per 1,000 test predictions
# -------------------------------------------------------------------------------------
# Computed from the evaluation run (<run>_infer): the best model predicts the val set INFERENCE_REPEATS
# times (batch 512).
#   latency (ms / image)             = time / number of predictions   (time per image in batch inference)
#   energy (Wh / 1000 predictions)   = energy / number of predictions x 1000
# Since, by the rule, the test set can only be opened once in step 5, the measurement uses the val images.
# The cost of a prediction depends only on the input size, not on the labels; since the val and test
# images have the same size (64x64 or 48x48), the cost of 1,000 test predictions is the same.
# Note: this is the average time in batch inference with batch 512; the single-image (batch 1) latency
# is higher.


def report_models(emissions):
    """Prints the energy, time, size and inference values of each model as a table and saves it."""
    n_predictions = len(load_split("val")) * INFERENCE_REPEATS
    records = []
    for name, run_name, build_model, input_size in MODELS:
        train = used_row(emissions, run_name)
        infer = used_row(emissions, f"{run_name}_infer")
        parameters, mflops, weights_mb, checkpoint_mb = model_size(
            build_model, input_size, os.path.join(MODEL_DIR, f"{run_name}.pt"))
        records.append({
            "Model": name, "Input": f"{input_size}px",
            "Training energy (kWh)": train["energy_consumed"],
            "Training emissions (kg CO2eq)": train["emissions"],
            "Training time (min)": train["duration"] / 60,
            "Params": parameters, "MFLOPs / image": mflops,
            "Model file (MB)": weights_mb, "Checkpoint (MB)": checkpoint_mb,
            "Latency (ms / image)": 1000 * infer["duration"] / n_predictions,
            "Energy (Wh / 1000 predictions)": 1000 * 1000 * infer["energy_consumed"] / n_predictions,
            "Training measurement": measurement_note(train),
        })
    table = pd.DataFrame(records)
    table.to_csv(FOOTPRINT_CSV, index=False)

    formats = {"Training energy (kWh)": "{:.4f}", "Training emissions (kg CO2eq)": "{:.4f}",
               "Training time (min)": "{:.1f}", "Params": "{:,}", "MFLOPs / image": "{:.1f}",
               "Model file (MB)": "{:.2f}", "Checkpoint (MB)": "{:.2f}",
               "Latency (ms / image)": "{:.3f}", "Energy (Wh / 1000 predictions)": "{:.4f}"}
    rows = [list(table.columns)]
    for _, record in table.iterrows():
        rows.append([formats[column].format(record[column]) if column in formats else record[column]
                     for column in table.columns])
    print_table("Footprint per model (final runs):", rows)
    if table["Training measurement"].str.contains("overlapping|sleep|low \\(").any():
        print(f"\n  Warning: some energy measurements were not taken under the same conditions (Training measurement"
              f"\n  column; details in {RUN_NOTES_CSV}). Params, MFLOPs and file size are not affected.")
    print(f"Saved: {FOOTPRINT_CSV}")


# =====================================================================================
# [x] Try at least a few techniques to reduce the footprint (small resolution, early stopping,
#     mixed precision, data caching, LR search on a subset, etc.) and report how much they
#     save.
# =====================================================================================
# Five techniques are tried. Existing runs are not touched; the new measurements are kept short, written
# to emissions.csv with the "reduce_" prefix and all done back to back in the same session (same power mode,
# no other load). The results are saved to models/reduction/reduction_results.json; if the file exists the
# measurements are not repeated.
#   1) Small resolution : the inference of our own CNN's 64px and 48px final models is measured back to back
#                         (the older measurements were in different power modes and not comparable).
#   2) Mixed precision  : the inference of the same 64px model is measured with bfloat16 (torch.autocast, CPU).
#                         This CPU (Alder Lake) has no hardware bf16 unit (AVX512-BF16 / AMX);
#                         the result may be negative and is reported as it is.
#   3) Caching          : decoding one epoch of train data from JPEG vs. reading it from the cache is measured.
#                         Without the cache this work would be repeated every epoch.
#   4) Subset LR search : 3 learning rates are tried for 3 epochs each on 20% of the train set. The energy of a
#                         MAX_EPOCHS-epoch search on the full data is estimated from the measurement in the same
#                         session (energy per image-epoch).
#   5) Early stopping   : no new runs. From the epoch history of the final runs, where training would stop with
#                         different patience values and how much energy would be left is computed retrospectively
#                         (the LR schedule is the same in the epochs before stopping, so the computation is exact).
#                         The 48px run is not included because of its sleep gap.

REDUCTION_DIR = os.path.join(MODEL_DIR, "reduction")
REDUCTION_JSON = os.path.join(REDUCTION_DIR, "reduction_results.json")
LR_CANDIDATES = [3e-4, 1e-3, 3e-3]
LR_SUBSET_FRACTION = 0.2
LR_SEARCH_EPOCHS = 3
PATIENCE_OPTIONS = [3, 5, EARLY_STOP_PATIENCE]
EARLY_STOP_RUNS = [("Own CNN", "own_cnn_64"), ("ResNet-18", standard.RESNET_RUN),
                   ("MobileNetV3-Large", standard.MOBILENET_RUN)]


def measure_own_cnn_inference(label, run_name, input_size, bfloat16=False):
    """Predicts the val set INFERENCE_REPEATS times with our own CNN's final model and measures it."""
    val = load_split("val")
    saved = torch.load(os.path.join(MODEL_DIR, f"{run_name}.pt"), weights_only=False)
    model = own_cnn.AgeCNN()
    model.load_state_dict(saved["model_state"])
    images = torch.from_numpy(load_images(val, input_size, "val"))

    def run():
        precision = torch.autocast("cpu", dtype=torch.bfloat16) if bfloat16 else contextlib.nullcontext()
        with precision:
            return [predict(model, images, saved["mean"], saved["std"]) for _ in range(INFERENCE_REPEATS)][0]

    predictions, energy = measure_energy(label, run)
    return {"energy_Wh": energy["energy_Wh"], "seconds": energy["seconds"],
            "val_MAE": regression_metrics(val["age"], predictions)["MAE"]}


def measure_caching():
    """Measures preparing one epoch of train data without the cache (JPEG decoding) and from the cache."""
    train = load_split("train")
    _, no_cache = measure_energy("reduce_cache_off_decode_jpeg", lambda: decode_images(train, 64))
    _, cache = measure_energy("reduce_cache_on_load_npz", lambda: load_images(train, 64, "train"))
    return {"no_cache": no_cache, "cache": cache}


def lr_search_on_subset():
    """Runs a short training of LR_SEARCH_EPOCHS epochs for each LR on 20% of the train set and measures it."""
    train, val, train_images, val_images, mean, std = load_train_val(64)
    positions = np.sort(np.random.default_rng(SEED).choice(
        len(train), size=int(LR_SUBSET_FRACTION * len(train)), replace=False))
    subset = (train.iloc[positions].reset_index(drop=True), val,
              train_images[torch.from_numpy(positions)], val_images, mean, std)

    results = []
    for lr in LR_CANDIDATES:
        model = own_cnn.build_own_cnn()
        name = f"reduce_lr_search_subset_lr{lr:g}"
        summary, energy = measure_energy(name, lambda: train_model(
            model, subset, os.path.join(REDUCTION_DIR, f"{name}.pt"),
            os.path.join(REDUCTION_DIR, f"{name}_history.csv"),
            learning_rate=lr, max_epochs=LR_SEARCH_EPOCHS))
        results.append({"lr": lr, "val_MAE": summary["val_MAE"], **energy})
    return {"subset_size": len(positions), "train_size": len(train), "runs": results}


def run_reduction_measurements():
    """Does the new measurements once, back to back; writes the results to JSON (reads it if present)."""
    if os.path.exists(REDUCTION_JSON):
        print(f"Using saved measurements: {REDUCTION_JSON}")
        with open(REDUCTION_JSON, encoding="utf-8") as file:
            return json.load(file)

    os.makedirs(REDUCTION_DIR, exist_ok=True)
    load_images(load_split("val"), 48, "val")  # make sure the cache is ready, so it does not leak into the measurement
    results = {
        "resolution_64": measure_own_cnn_inference("reduce_infer_own_cnn_64_fp32", "own_cnn_64", 64),
        "resolution_48": measure_own_cnn_inference("reduce_infer_own_cnn_48_fp32", "own_cnn_48", 48),
        "bfloat16_64": measure_own_cnn_inference("reduce_infer_own_cnn_64_bf16", "own_cnn_64", 64, True),
        "caching": measure_caching(),
        "lr_search": lr_search_on_subset(),
    }
    with open(REDUCTION_JSON, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return results


def simulate_early_stopping(history, patience):
    """Returns the epoch where training would stop with the given patience, and the best MAE up to it."""
    best, waited = float("inf"), 0
    for _, row in history.iterrows():
        if row["val_MAE"] < best:
            best, waited = row["val_MAE"], 0
        else:
            waited += 1
            if waited >= patience:
                return int(row["epoch"]), best
    return int(history["epoch"].iloc[-1]), best


def report_reductions(emissions):
    """Prints the savings of the five techniques."""
    results = run_reduction_measurements()

    def change(new, old):
        return f"{100 * (new - old) / old:+.1f}%"

    # 1-2) Resolution and mixed precision (inference, measured in the same session)
    base, small, bf16 = results["resolution_64"], results["resolution_48"], results["bfloat16_64"]
    n_predictions = len(load_split("val")) * INFERENCE_REPEATS
    rows = [["Variant", "Energy (Wh / 1000 predictions)", "Latency (ms / image)", "val MAE", "Energy change"]]
    for name, result in [("64px fp32 (baseline)", base), ("48px fp32", small), ("64px bfloat16", bf16)]:
        rows.append([name, f"{1e3 * result['energy_Wh'] / n_predictions:.4f}",
                     f"{1e3 * result['seconds'] / n_predictions:.3f}", f"{result['val_MAE']:.2f}",
                     "-" if result is base else change(result["energy_Wh"], base["energy_Wh"])])
    print_table("1-2) Small resolution and mixed precision (own CNN, inference, same session):", rows)

    # 3) Caching
    no_cache, cache = results["caching"]["no_cache"], results["caching"]["cache"]
    final_epochs = sum(len(pd.read_csv(os.path.join(MODEL_DIR, f"{run}_history.csv")))
                       for _, run in EARLY_STOP_RUNS)  # total number of epochs of the 64px final runs
    saved_wh = (no_cache["energy_Wh"] - cache["energy_Wh"]) * final_epochs
    rows = [["Data preparation (1 epoch, train)", "Time (s)", "Energy (Wh)"],
            ["JPEG decoding + downscaling (no cache)", f"{no_cache['seconds']:.1f}", f"{no_cache['energy_Wh']:.3f}"],
            ["Reading from the cache (.npz)", f"{cache['seconds']:.1f}", f"{cache['energy_Wh']:.3f}"]]
    print_table("3) Data caching:", rows)
    print(f"  Without the cache, over the {final_epochs} epochs of the 64px final runs an extra "
          f"~{saved_wh:.1f} Wh would have been spent ({no_cache['energy_Wh'] - cache['energy_Wh']:.3f} Wh per epoch).")

    # 4) LR search on a subset
    search = results["lr_search"]
    rows = [["LR", f"val MAE ({LR_SEARCH_EPOCHS} epochs, {100 * LR_SUBSET_FRACTION:.0f}% of data)",
             "Time (min)", "Energy (Wh)"]]
    for run in search["runs"]:
        rows.append([f"{run['lr']:g}", f"{run['val_MAE']:.2f}", f"{run['seconds'] / 60:.1f}", f"{run['energy_Wh']:.2f}"])
    subset_wh = sum(run["energy_Wh"] for run in search["runs"])
    wh_per_image_epoch = subset_wh / (search["subset_size"] * LR_SEARCH_EPOCHS * len(search["runs"]))
    full_wh = wh_per_image_epoch * search["train_size"] * MAX_EPOCHS * len(search["runs"])
    best_lr = min(search["runs"], key=lambda run: run["val_MAE"])["lr"]
    print_table("4) LR search on a subset (own CNN, 64px):", rows)
    print(f"""  Selected LR: {best_lr:g} (used in the final trainings: {LEARNING_RATE:g})
  Subset search: {subset_wh:.2f} Wh measured. The same search on the full data with {MAX_EPOCHS} epochs would
  cost ≈ {full_wh:.1f} Wh (estimated from the energy per image-epoch in the same session) -> saving {100 * (1 - subset_wh / full_wh):.1f}%.
  Limitation: a ranking made with {LR_SEARCH_EPOCHS} epochs and {100 * LR_SUBSET_FRACTION:.0f}% of the data is only an approximate
  indicator of the ranking in a long training.""")

    # 5) Early stopping (retrospective, from the existing histories)
    rows = [["Model", "Patience", "Stop epoch", "Remaining epochs", "Energy saving", "val MAE (best)", "MAE change"]]
    for name, run in EARLY_STOP_RUNS:
        history = pd.read_csv(os.path.join(MODEL_DIR, f"{run}_history.csv"))
        energy_wh = used_row(emissions, run)["energy_consumed"] * 1000
        full_best = history["val_MAE"].min()
        for patience in PATIENCE_OPTIONS:
            stop_epoch, best = simulate_early_stopping(history, patience)
            remaining = history[history["epoch"] > stop_epoch]["seconds"].sum() / history["seconds"].sum()
            rows.append([name, patience, stop_epoch, len(history) - stop_epoch,
                         f"{100 * remaining:.0f}% ({energy_wh * remaining:.1f} Wh)",
                         f"{best:.2f}", f"{best - full_best:+.2f}"])
    print_table(f"5) Early stopping (retrospective; current setting patience {EARLY_STOP_PATIENCE}, "
                f"at most {MAX_EPOCHS} epochs):", rows)
    print(f"""  The patience {EARLY_STOP_PATIENCE} rows (current setting) show the epoch where the runs actually stopped; if the
  stop epoch equals the last epoch run, the training was ended by the epoch limit, not by early stopping.
  A smaller patience cuts training earlier and saves energy, but risks missing the best epoch
  (MAE change column). The energy is prorated by the epoch durations.""")


def main():
    # The footprint-reduction measurements (only on the first run) are done at the very beginning, so the
    # total table below also includes these new runs.
    run_reduction_measurements()
    emissions = read_emissions()

    print_section("1) Energy, time, size and inference cost of the final trainings (per model)")
    report_models(emissions)

    print_section("2) Total energy and emissions of the project")
    runs = report_project_total(emissions)

    print_section("2b) Comparison of the measurement sessions (before / after the Windows settings)")
    report_session_comparison(runs)

    print_section("3) Hardware and location")
    report_hardware(emissions)

    print_section("4) Footprint-reduction techniques")
    report_reductions(emissions)


if __name__ == "__main__":
    main()
