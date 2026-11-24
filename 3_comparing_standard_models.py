"""
Step 3 — Comparison with standard models.

Usage:
    python 3_comparing_standard_models.py

Each model is trained once and measured with CodeCarbon (emissions/emissions.csv). The training and
measurement code is the same as for our own CNN (train_utils.run_experiment). If the result exists
on disk it is not retrained; to retrain, delete models/<run name>_summary.json.
"""

import hashlib
import json
import os
from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models import (resnet18, ResNet18_Weights,
                                mobilenet_v3_large, MobileNet_V3_Large_Weights)

from data_utils import ROOT, SPLITS_CSV, print_section, print_table
from train_utils import (MODEL_DIR, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                         EARLY_STOP_PATIENCE, run_experiment, train_mean_age)

# Image size policy: all models (including our own CNN) use the same 64x64 input.
# So the difference in error and energy between models comes only from the architecture and pretraining.
INPUT_SIZE = 64


class AgeRegressor(nn.Module):
    """Adapts a backbone to age regression: (N, 3, H, W) -> (N,) age (years)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # (N, 1) -> (N,)


def regression_head(in_features):
    """Single-output Linear layer; the bias starts at the mean train age (as in our own CNN)."""
    head = nn.Linear(in_features, 1)
    nn.init.constant_(head.bias, train_mean_age())
    return head


def complexity(model, input_size=INPUT_SIZE):
    """Returns the parameter count and the MACs per image (with PyTorch's FLOP counter)."""
    model.eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size))
    return sum(p.numel() for p in model.parameters()), counter.get_total_flops() // 2


# =====================================================================================
# [x] Train a large model (ResNet-18/50, ConvNeXt-Tiny, etc.).
# =====================================================================================
# Chosen: ResNet-18, fine-tuned from ImageNet weights (torchvision IMAGENET1K_V1).
# - Why ResNet-18: the smallest standard representative of the "large model" class that can be trained
#   on a CPU. With 11.2 M parameters it is about 38 times larger than our own CNN (~0.29 M); ResNet-50 (25.6 M)
#   or ConvNeXt-Tiny (28.6 M) would take hours on this machine (no GPU) and consume much more energy.
# - The architecture is not changed (so it stays a standard model): only the last layer (fc, 1000 classes)
#   is replaced with a single-output regression head.
# - With a 64x64 input, ResNet's stem (stride-2 conv + max pool) and the following stages shrink the map
#   to 2x2; this is a known cost of the shared image size policy.
# - The training settings (loss, optimizer, lr, schedule, augmentation, stopping) are the same as for our
#   own CNN. The input normalization is also the same (computed from the train set); using it instead of the
#   ImageNet statistics is acceptable because the BN layers adapt during fine-tuning.
# - The weights are downloaded while the model is built, before the energy measurement starts.

RESNET_RUN = "resnet18_imagenet_64"


def build_resnet18():
    """ResNet-18 with ImageNet weights + regression head."""
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = regression_head(backbone.fc.in_features)  # 512 -> 1
    return AgeRegressor(backbone)


def train_resnet18():
    """Reports the size of ResNet-18, then trains and measures the model (or reads the saved result)."""
    parameters, macs = complexity(build_resnet18())
    print(f"ResNet-18 (input {INPUT_SIZE}x{INPUT_SIZE}): {parameters:,} parameters, "
          f"{macs / 1e6:.1f} M MACs / image")
    return run_experiment(RESNET_RUN, build_resnet18, INPUT_SIZE)


# =====================================================================================
# [x] Train an efficiency-oriented model (MobileNetV3, EfficientNet-B0, ShuffleNetV2, etc.).
# =====================================================================================
# Chosen: MobileNetV3-Large, fine-tuned from ImageNet weights (torchvision IMAGENET1K_V1).
# Cost of the candidates at a 64x64 input (with the original 1000-class head, FlopCounterMode):
#     Model               Params      MACs / image
#     MobileNetV3-Small   2.54 M       6.5 M
#     MobileNetV3-Large   5.48 M      21.4 M   <- chosen
#     ShuffleNetV2 x1.0   2.28 M      12.8 M
#     EfficientNet-B0     5.29 M      33.2 M
#     (ResNet-18         11.69 M     148.6 M,  our own CNN 0.29 M, 53.7 M)
# - Why MobileNetV3-Large: its MACs are ~40% of our own CNN's and ~1/7 of ResNet-18's; so the three models
#   are at clearly different cost levels. Small is cheaper, but its capacity may be limited for the fine
#   facial details needed for age. EfficientNet-B0 is more expensive and uses the SiLU activation, which
#   runs slowly on a CPU.
# - The efficiency idea: depthwise separable convolutions (3x3 per channel + 1x1 mixing),
#   inverted residual blocks and squeeze-excitation; they do the same work with far fewer MACs.
# - Caveat: depthwise convolutions are memory-bound on a CPU; therefore the real time and energy may
#   not drop as much as the MACs. This difference will be seen in the measurement (CodeCarbon).
# - The architecture is not changed; only the last Linear layer of the classifier (1280 -> 1000) is
#   replaced with a single-output regression head. Training settings and input size are the same as the other models.

MOBILENET_RUN = "mobilenetv3_large_imagenet_64"


def build_mobilenet_v3_large():
    """MobileNetV3-Large with ImageNet weights + regression head."""
    backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    backbone.classifier[-1] = regression_head(backbone.classifier[-1].in_features)  # 1280 -> 1
    return AgeRegressor(backbone)


def train_mobilenet_v3_large():
    """Reports the size of MobileNetV3-Large, then trains and measures the model (or reads the saved result)."""
    parameters, macs = complexity(build_mobilenet_v3_large())
    print(f"MobileNetV3-Large (input {INPUT_SIZE}x{INPUT_SIZE}): {parameters:,} parameters, "
          f"{macs / 1e6:.1f} M MACs / image")
    return run_experiment(MOBILENET_RUN, build_mobilenet_v3_large, INPUT_SIZE)


# =====================================================================================
# [x] Put a regression head instead of the classifier head in both.
# =====================================================================================
# The same method in both models: the last classification layer (1000 outputs) is replaced with
# regression_head() (single-output Linear, bias = mean train age) and AgeRegressor makes the output (N,).
#     ResNet-18         : fc (512 -> 1000)              -> Linear(512 -> 1)
#     MobileNetV3-Large : classifier[-1] (1280 -> 1000) -> Linear(1280 -> 1)
# Check: both models return a single age prediction of shape (N,) for a sample input, and the initial
# prediction is close to the train mean (~33 years).


# =====================================================================================
# [x] Fine-tune at least one of them from ImageNet weights (transfer learning).
# =====================================================================================
# Both standard models are fine-tuned from ImageNet weights (torchvision IMAGENET1K_V1):
# build_resnet18() and build_mobilenet_v3_large() are built with weights=...IMAGENET1K_V1 and all
# layers (none frozen) are retrained for age regression. Since our own CNN is trained from scratch,
# the comparison also shows the effect of transfer learning.


# =====================================================================================
# [x] Discuss the carbon footprint that pretraining created in the past.
# =====================================================================================
# The ImageNet weights we use do not appear in our emissions.csv, but they are not free:
# someone trained them once and that energy was spent ("embodied carbon"). No energy / carbon value
# has been published for these weights, so it is estimated.
#
# Verified facts (sources):
#   - torchvision training recipe (references/classification/README.md):
#       "Except otherwise noted, all models have been trained on 8x V100 GPUs"
#       ResNet-18   : 90 epochs (general table and train.py default), batch 32 / GPU
#       MobileNetV3 : "--epochs 600 --opt rmsprop --batch-size 128 ..."
#     The README gives no training time, energy or carbon.
#   - MACs per image (224x224): torchvision weight metadata (_ops), ResNet-18 1.814 G,
#     MobileNetV3-Large 0.217 G. (Consistent with our 64x64 measurement: 148 M x (224/64)^2 ≈ 1.81 G.)
#   - The MobileNetV3 architecture was found by search (Howard et al., 2019, "Searching for MobileNetV3"):
#       "we use platform-aware NAS to search for the global network structures ...
#        We then use the NetAdapt algorithm to search per layer for the number of filters."
#     The paper does not give the search cost. MnasNet, which introduced the same search method (Tan et al., 2019):
#       "each architecture search takes 4.5 days on 64 TPUv2 devices ... our controller
#        samples about 8K models during architecture search"
#
# Estimation method (only the final published training run):
#   training FLOPs = 3 (forward + backward) x 2 (FLOP / MAC) x MACs x number of images x epochs
#   GPU-hours      = FLOPs / (GPU utilization x V100 FP32 peak throughput) / 3600
#   energy         = GPU-hours x GPU power x PUE (data-center overhead)
#   CO2            = energy x carbon intensity of the grid
# Since utilization, PUE and carbon intensity are unknown, three scenarios are used.

IMAGENET_TRAIN_IMAGES = 1_281_167
V100_FP32_TFLOPS = 15.7      # NVIDIA V100 peak FP32 throughput
V100_POWER_KW = 0.300        # V100 SXM2 TDP; the server's CPU/RAM power is not included (lower bound)

PRETRAINED_MODELS = [
    # (name, weights, number of ImageNet epochs, name of our fine-tuning run)
    ("ResNet-18", ResNet18_Weights.IMAGENET1K_V1, 90, RESNET_RUN),
    ("MobileNetV3-Large", MobileNet_V3_Large_Weights.IMAGENET1K_V1, 600, MOBILENET_RUN),
]

PRETRAINING_SCENARIOS = [
    # (name, GPU utilization, PUE, carbon intensity kg CO2 / kWh)
    ("Optimistic", 0.40, 1.1, 0.2),
    ("Central", 0.20, 1.3, 0.4),
    ("Pessimistic", 0.10, 1.6, 0.6),
]

MNASNET_SEARCH_DEVICE_HOURS = 4.5 * 24 * 64   # 6912 TPUv2 device-hours (MnasNet paper)


def pretraining_estimate(macs_224, epochs, utilization, pue, carbon_intensity):
    """Estimate of the FLOPs, GPU-hours, kWh and kg CO2 of one ImageNet pretraining."""
    flops = 3 * 2 * macs_224 * IMAGENET_TRAIN_IMAGES * epochs
    gpu_hours = flops / (utilization * V100_FP32_TFLOPS * 1e12) / 3600
    energy_kwh = gpu_hours * V100_POWER_KW * pue
    return flops, gpu_hours, energy_kwh, energy_kwh * carbon_intensity


def load_summary(run_name):
    """Returns the saved summary of a run; None if the run has not been done yet."""
    path = os.path.join(MODEL_DIR, f"{run_name}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def report_pretraining_footprint():
    """Prints the estimated energy and CO2 of pretraining and compares them with our measurements."""
    rows = [["Model", "Epochs", "Training FLOPs", "Scenario", "GPU-hours", "Energy (kWh)", "CO2 (kg)"]]
    middle = {}
    for name, weights, epochs, _ in PRETRAINED_MODELS:
        macs_224 = weights.meta["_ops"] * 1e9
        for scenario, utilization, pue, intensity in PRETRAINING_SCENARIOS:
            flops, gpu_hours, kwh, co2 = pretraining_estimate(macs_224, epochs, utilization, pue, intensity)
            rows.append([name, epochs, f"{flops:.2e}", scenario, f"{gpu_hours:.0f}", f"{kwh:.1f}", f"{co2:.1f}"])
            if scenario == "Central":
                middle[name] = (kwh, co2)
    print_table("Estimated footprint of ImageNet pretraining (only the final published run):", rows)
    print("\nScenario assumptions (GPU utilization / PUE / kg CO2 per kWh):")
    for scenario, utilization, pue, intensity in PRETRAINING_SCENARIOS:
        print(f"  {scenario:11s}: {utilization * 100:.0f}% / {pue} / {intensity}")

    # Comparison with the fine-tuning trainings we measured
    rows = [["Model", "Pretraining (Central, kWh)", "Our fine-tuning (Wh)", "Pretraining / fine-tuning"]]
    ratios = []
    for name, _, _, run_name in PRETRAINED_MODELS:
        summary = load_summary(run_name)
        if summary is None:
            rows.append([name, f"{middle[name][0]:.1f}", "not trained yet", "-"])
        else:
            ratios.append(middle[name][0] * 1000 / summary["train_energy_Wh"])
            rows.append([name, f"{middle[name][0]:.1f}", f"{summary['train_energy_Wh']:.2f}", f"~{ratios[-1]:,.0f}x"])
    print_table("Pretraining vs. the fine-tuning energy we measured:", rows)
    scale = f"~{min(ratios):,.0f}-{max(ratios):,.0f} times" if ratios else "(fine-tuning not measured yet)"

    print(f"""
Discussion:
  1) Scale: in the central scenario, a single ImageNet pretraining consumes {scale} more energy
     than one of our fine-tuning trainings. By downloading the weights we do not repeat this cost;
     we only pay a very small fine-tuning cost. This is the green side of transfer learning.
  2) This estimate is a lower bound. It counts only the final published training run; hyperparameter
     searches, failed trials and development runs are not included. Also only GPU power is counted
     (no server CPU, RAM and network power) and the carbon of hardware manufacturing is not included.
  3) An "efficient model" is efficient at inference, not at training. MobileNetV3-Large does ~8 times
     fewer MACs per image than ResNet-18, but since it was trained for 600 instead of 90 epochs the total
     training compute is almost the same. Because depthwise convolutions use the GPU less efficiently and
     600 epochs of data augmentation load the CPU, its real cost is most likely closer to the
     pessimistic scenario.
  4) The cost of the architecture search must be added separately. The MobileNetV3 architecture was found
     with a MnasNet-style search and NetAdapt. According to the MnasNet paper a single search takes 4.5 days
     on 64 TPUv2 devices: {MNASNET_SEARCH_DEVICE_HOURS:,.0f} device-hours, i.e. far more than the GPU-hours of
     the weight training above. The MobileNetV3 paper does not give its own search cost, so it can only be
     given as an order of magnitude. ResNet, on the other hand, was designed by hand.
  5) Sharing (amortization): pretraining is done once and the weights are reused by many people; the share
     per person is small. However, an honest carbon report of the project should state this embodied cost
     separately, as an estimate, next to the emissions.csv total.
  6) The uncertainty is large: there is a ~{max_min_ratio():.0f}x difference between the scenarios. Therefore
     the result should be reported not as a single number but as a range and an order of magnitude.""")


def max_min_ratio():
    """Ratio between the pessimistic and optimistic scenario CO2 estimates (independent of the model)."""
    estimates = [pretraining_estimate(1.0, 1, u, p, c)[3] for _, u, p, c in PRETRAINING_SCENARIOS]
    return max(estimates) / min(estimates)


# =====================================================================================
# [x] Use the same split, the same image size policy and the same evaluation code for all models.
# =====================================================================================
# How it is ensured in code:
#   - Same split : every model reads data/splits.csv with data_utils.load_split() (seed 42,
#                  generated once and preserved). The test set only in step 5, with load_test_once().
#   - Same images: every model gets the same 200x200 -> 64x64 bilinear-downscaled images from the same
#                  cache (data_utils.load_images) and the same normalization computed from train.
#   - Same code  : every model is trained and measured with train_utils.run_experiment(): same loss,
#                  optimizer, lr, schedule, augmentation, stopping, the same regression_metrics()
#                  and the same CodeCarbon measurement. The only model-specific part is the build_model function.
# check_fairness() verifies this on the runs that were executed and shows the results in one table.

COMPARED_RUNS = [
    # (name, run name, pretraining)
    ("Own CNN", "own_cnn_64", "none (from scratch)"),
    ("ResNet-18", RESNET_RUN, "ImageNet"),
    ("MobileNetV3-Large", MOBILENET_RUN, "ImageNet"),
]
OWN_CNN_LAYERS_CSV = os.path.join(MODEL_DIR, "own_cnn_layers.csv")
MODEL_BUILDERS = {RESNET_RUN: build_resnet18, MOBILENET_RUN: build_mobilenet_v3_large}


RUN_NOTES_CSV = os.path.join(ROOT, "emissions", "run_notes.csv")


def measurement_status(run_name):
    """Status of the training measurement used in the run's summary (emissions/run_notes.csv)."""
    if not os.path.exists(RUN_NOTES_CSV):
        return "no note (ran alone)"
    notes = pd.read_csv(RUN_NOTES_CSV)
    used = notes[(notes["project_name"] == run_name) & (notes["used_in_summary"] == "yes")]
    if used.empty:
        return "no note (ran alone)"
    note = used.iloc[-1]
    return f"{note['measurement_status']}, {note['power_mode']}"


def file_fingerprint(path):
    """First 12 characters of the file's SHA-256 digest."""
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()[:12]


def run_macs(run_name):
    """MACs per image at 64x64 of the run's model."""
    if run_name in MODEL_BUILDERS:
        return complexity(MODEL_BUILDERS[run_name]())[1]
    return pd.read_csv(OWN_CNN_LAYERS_CSV)["MACs"].sum()  # own CNN: the layer table of step 2


def check_fairness():
    """Verifies that all models were evaluated with the same split, the same input size and the same code."""
    split_time = os.path.getmtime(SPLITS_CSV)
    print(f"Split file    : {SPLITS_CSV}")
    print(f"  SHA-256     : {file_fingerprint(SPLITS_CSV)}   (all models read this file)")
    print(f"  Last change : {datetime.fromtimestamp(split_time):%Y-%m-%d %H:%M}")
    print(f"Training settings (train_utils, identical for all models): batch {BATCH_SIZE}, AdamW lr {LEARNING_RATE:g}, "
          f"weight decay {WEIGHT_DECAY:g}, early stopping {EARLY_STOP_PATIENCE}. The epoch limit per model is "
          f"in the table (current setting: {MAX_EPOCHS}).")

    rows = [["Model", "Pretraining", "Input", "Epoch limit", "Split before training?", "Params", "MACs (M)",
             "val MAE", "val RMSE", "val R²", "Training (Wh)", "Inference (mWh/1000)", "Energy measurement"]]
    problems, epoch_limits = [], set()
    for name, run_name, pretraining in COMPARED_RUNS:
        summary = load_summary(run_name)
        if summary is None:
            rows.append([name, pretraining, "-", "-", "-", "-", "-", "not trained yet", "", "", "", "", ""])
            problems.append(f"{name} not trained yet")
            continue

        summary_time = os.path.getmtime(os.path.join(MODEL_DIR, f"{run_name}_summary.json"))
        split_before = summary_time > split_time
        if summary["input_size"] != INPUT_SIZE:
            problems.append(f"{name}: input {summary['input_size']} (expected {INPUT_SIZE})")
        if not split_before:
            problems.append(f"{name}: the split file changed after this training")

        # The "max_epochs" field was added to the summary later; all summaries without it were trained with a 20-epoch limit.
        epoch_limit = summary.get("max_epochs", 20)
        epoch_limits.add(epoch_limit)
        rows.append([name, pretraining, f"{summary['input_size']}px", epoch_limit,
                     "yes" if split_before else "NO",
                     f"{summary['parameters']:,}", f"{run_macs(run_name) / 1e6:.1f}",
                     f"{summary['val_MAE']:.2f}", f"{summary['val_RMSE']:.2f}", f"{summary['val_R2']:.3f}",
                     f"{summary['train_energy_Wh']:.2f}", f"{summary['infer_mWh_per_1000']:.2f}",
                     measurement_status(run_name)])
    print_table("Results measured with the same split, the same input and the same code (validation):", rows)
    if any(word in row[-1] for row in rows[1:] for word in ("overlapping", "sleep", "low (")):
        print("\nWarning: the energy measurements were not taken under the same conditions (overlapping runs /\n"
              f"different power modes); details in {RUN_NOTES_CSV}. The error metrics (val MAE, RMSE, R²) are not affected.")

    if len(epoch_limits) > 1:
        problems.append(f"the models were trained with different epoch limits ({sorted(epoch_limits)}); "
                        "the training budget is not equal, this must be stated in the report")

    if problems:
        print("\nMissing / inconsistent:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nCheck passed: all models were measured with the same split, the same 64x64 input and the same "
              "training/evaluation code.")


def main():
    print_section("1) Large model: ResNet-18 (ImageNet fine-tuning)")
    train_resnet18()

    print_section("2) Efficiency-oriented model: MobileNetV3-Large (ImageNet fine-tuning)")
    train_mobilenet_v3_large()

    print_section("3) Carbon footprint of pretraining")
    report_pretraining_footprint()

    print_section("4) Same split, same image size and same evaluation code")
    check_fairness()


if __name__ == "__main__":
    main()
