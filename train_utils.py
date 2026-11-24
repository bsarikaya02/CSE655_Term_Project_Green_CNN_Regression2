"""
Shared training and energy-measurement code for all models (step 2: own CNN, step 3: standard models).

Because the same functions and the same settings are used, the comparison between models is
fair: data, split, loss, optimizer, schedule, augmentation and stopping criterion are identical
in code for every model.

    run_experiment(name, build_model, size) -> trains and measures the model once; reads the result from disk if it exists
    measure_energy(name, function)          -> measures the function with CodeCarbon, writes to emissions.csv

Every training and evaluation run is measured and appended as one row to emissions/emissions.csv
(including runs that end with an error). The project's total CO2 is the sum of this file.
"""

"""
Windows settings for carbon measurement:
1. Plugged in
2. Fixed power mode
3. Sleep mode off
4. Minimal background load
5. Antivirus and updates paused
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from codecarbon import OfflineEmissionsTracker

from data_utils import ROOT, SEED, HUBER_DELTA, load_split, load_images, regression_metrics

MODEL_DIR = os.path.join(ROOT, "models")
LOG_DIR = os.path.join(ROOT, "emissions")
EMISSIONS_CSV = os.path.join(LOG_DIR, "emissions.csv")

# Training settings: identical for all models (rationale in the design decisions section of 2_own_cnn.py)
BATCH_SIZE = 128
MAX_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2          # AdamW default; 1e-4 has almost no effect with AdamW
LR_PATIENCE, LR_FACTOR = 3, 0.5
EARLY_STOP_PATIENCE = 8
INFERENCE_REPEATS = 10       # inference energy is measured by predicting the val set this many times


# -------------------------------------------------------------------------------------
# Energy measurement
# -------------------------------------------------------------------------------------
# On this machine CodeCarbon reads CPU energy from the Windows Energy Meter Interface (RAPL) counter.
# Verified (output of a test run with CodeCarbon 3.3.2, log_level="info"):
#     [codecarbon INFO] Tracking CPU via the Windows Energy Meter Interface (RAPL)
#     [codecarbon INFO] CPU Tracking Method: Windows EMI
#     [codecarbon INFO] RAM Tracking Method: RAM power estimation model
#     [codecarbon INFO] CPU model: 12th Gen Intel(R) Core(TM) i7-12650H
# So CPU energy is measured, while RAM energy is estimated with a power model (10 W).
# There is no GPU. CO2 is computed with the carbon intensity of Turkey's electricity grid.

def measure_energy(run_name, function):
    """Measures function() with CodeCarbon. Returns (result, {seconds, energy Wh, CO2 g}).

    Each run is appended to emissions.csv as a row with project_name = run_name.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    tracker = OfflineEmissionsTracker(
        country_iso_code="TUR", project_name=run_name,
        output_dir=LOG_DIR, output_file="emissions.csv",
        measure_power_secs=1, allow_multiple_runs=True, log_level="error")
    tracker.start()
    try:
        result = function()
    finally:
        tracker.stop()  # the run is written to emissions.csv even if an error occurs
    data = tracker.final_emissions_data  # filled in during stop(), so it is read afterwards
    return result, {"seconds": data.duration,
                    "energy_Wh": data.energy_consumed * 1000,
                    "co2_g": data.emissions * 1000}


# -------------------------------------------------------------------------------------
# Data preparation and prediction
# -------------------------------------------------------------------------------------

def normalize(images, mean, std):
    """uint8 (N, H, W, 3) -> normalized float (N, 3, H, W)."""
    x = images.permute(0, 3, 1, 2).float().div(255)
    return (x - mean) / std


def random_horizontal_flip(x, generator):
    """Horizontally flips a random half of the images in the batch."""
    flip = torch.rand(x.shape[0], generator=generator) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(dims=[3]), x)


def predict(model, images, mean, std, batch_size=512):
    """Returns the age predictions (numpy) for the images."""
    model.eval()
    with torch.no_grad():
        outputs = [model(normalize(images[i:i + batch_size], mean, std))
                   for i in range(0, len(images), batch_size)]
    return torch.cat(outputs).float().numpy()  # float(): so bfloat16 inference can also be converted to numpy


def load_train_val(input_size):
    """Returns the train/val images (uint8 tensors), the ages and the normalization computed from train."""
    train, val = load_split("train"), load_split("val")
    train_images = torch.from_numpy(load_images(train, input_size, "train"))
    val_images = torch.from_numpy(load_images(val, input_size, "val"))

    pixels = train_images.permute(0, 3, 1, 2).float().div(255)
    mean = pixels.mean(dim=(0, 2, 3), keepdim=True)
    std = pixels.std(dim=(0, 2, 3), keepdim=True)
    return train, val, train_images, val_images, mean, std


def train_mean_age():
    """Mean of the train ages (the models' output bias is initialized with it)."""
    return float(load_split("train")["age"].mean())


# -------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------

def train_model(model, data, checkpoint_path, history_path,
                learning_rate=LEARNING_RATE, max_epochs=MAX_EPOCHS):
    """Trains the given model from scratch; saves the model with the best val MAE and the epoch history.

    Loss Huber, optimizer AdamW, schedule ReduceLROnPlateau, stopping by early stopping on val MAE.
    Trained only on train and measured on val; the test set is not touched.
    learning_rate and max_epochs are only changed in the subset LR search of step 4;
    all final models are trained with the default values.
    """
    train, val, train_images, val_images, mean, std = data
    train_ages = torch.tensor(train["age"].to_numpy(), dtype=torch.float32)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE)

    history, best_val_mae, epochs_without_improvement = [], float("inf"), 0
    print(f"Training: {len(train)} train / {len(val)} val images, input {train_images.shape[1]}px, "
          f"at most {max_epochs} epochs")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()

        # --- One training epoch ---
        model.train()
        total_loss, total_abs_error = 0.0, 0.0
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(train), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            x = random_horizontal_flip(normalize(train_images[batch], mean, std), generator)
            y = train_ages[batch]

            predictions = model(x)
            loss = loss_fn(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            total_abs_error += (predictions.detach() - y).abs().sum().item()

        # --- Validation and LR schedule ---
        val_metrics = regression_metrics(val["age"], predict(model, val_images, mean, std))
        scheduler.step(val_metrics["MAE"])

        history.append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / len(train), "train_MAE": total_abs_error / len(train),
            "val_MAE": val_metrics["MAE"], "val_RMSE": val_metrics["RMSE"], "val_R2": val_metrics["R2"],
            "seconds": time.perf_counter() - epoch_start,
        })
        row = history[-1]
        print(f"Epoch {epoch:2d} | lr {row['lr']:.1e} | train loss {row['train_loss']:6.2f} "
              f"MAE {row['train_MAE']:5.2f} | val MAE {row['val_MAE']:5.2f} RMSE {row['val_RMSE']:5.2f} "
              f"R² {row['val_R2']:.3f} | {row['seconds']:.0f} s", flush=True)

        # --- Keep the best model / early stopping ---
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae, epochs_without_improvement = val_metrics["MAE"], 0
            torch.save({"model_state": model.state_dict(), "mean": mean, "std": std,
                        "epoch": epoch, "val_metrics": val_metrics}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Val MAE did not improve for {EARLY_STOP_PATIENCE} epochs, training stopped.")
                break

    pd.DataFrame(history).to_csv(history_path, index=False)
    best = min(history, key=lambda r: r["val_MAE"])
    return {"best_epoch": best["epoch"], "epochs_run": len(history),
            "val_MAE": best["val_MAE"], "val_RMSE": best["val_RMSE"], "val_R2": best["val_R2"]}


# -------------------------------------------------------------------------------------
# Experiment: training + inference, both measured
# -------------------------------------------------------------------------------------

def run_experiment(run_name, build_model, input_size):
    """Trains and measures the model once; writes the summary to models/<run_name>_summary.json.

    The epoch limit used (MAX_EPOCHS) is written to the summary as "max_epochs", so models
    trained with different limits can be told apart later.

    If the summary file exists, the model is not retrained (repeating the same run wastes energy)
    and the saved summary is returned. To retrain, delete the summary file.
      - Training run  : project_name = <run_name> in emissions.csv
      - Inference run : project_name = <run_name>_infer in emissions.csv
    """
    summary_path = os.path.join(MODEL_DIR, f"{run_name}_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as file:
            summary = json.load(file)
        print(f"Using saved result: {summary_path}")
        print_summary(summary)
        return summary

    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, f"{run_name}.pt")
    history_path = os.path.join(MODEL_DIR, f"{run_name}_history.csv")

    # Data and model are prepared before measuring, so JPEG decoding / cache creation does not leak into the energy.
    data = load_train_val(input_size)
    model = build_model()

    # 1) Training (measured)
    training, train_energy = measure_energy(
        run_name, lambda: train_model(model, data, checkpoint_path, history_path))

    # 2) Inference (measured): the best model predicts the val set INFERENCE_REPEATS times.
    saved = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(saved["model_state"])
    val_images = data[3]
    _, infer_energy = measure_energy(
        f"{run_name}_infer",
        lambda: [predict(model, val_images, saved["mean"], saved["std"]) for _ in range(INFERENCE_REPEATS)])
    n_predictions = len(val_images) * INFERENCE_REPEATS

    summary = {
        "run_name": run_name, "input_size": input_size, "max_epochs": MAX_EPOCHS,
        "parameters": sum(p.numel() for p in model.parameters()),
        **training,
        "train_minutes": train_energy["seconds"] / 60,
        "train_energy_Wh": train_energy["energy_Wh"],
        "train_co2_g": train_energy["co2_g"],
        "infer_ms_per_image": 1000 * infer_energy["seconds"] / n_predictions,
        "infer_mWh_per_1000": 1e6 * infer_energy["energy_Wh"] / n_predictions,
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print_summary(summary)
    return summary


def print_summary(summary):
    """Prints the error and energy summary of an experiment."""
    mae_mean = regression_metrics(load_split("val")["age"],
                                  np.full(len(load_split("val")), train_mean_age()))["MAE"]
    print(f"""
{summary['run_name']}: {summary['epochs_run']} epochs, best epoch {summary['best_epoch']}
  val MAE {summary['val_MAE']:.2f}  RMSE {summary['val_RMSE']:.2f}  R² {summary['val_R2']:.3f}   (MAE / MAE_mean = {summary['val_MAE'] / mae_mean:.3f})
  Training : {summary['train_minutes']:.1f} min, {summary['train_energy_Wh']:.2f} Wh, {summary['train_co2_g']:.2f} g CO2
  Inference: {summary['infer_ms_per_image']:.3f} ms/image, {summary['infer_mWh_per_1000']:.2f} mWh / 1000 images""")
