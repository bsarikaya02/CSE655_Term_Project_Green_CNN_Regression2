"""
Step 2 — Our own CNN: an age regression network built from scratch and trained from scratch.

Usage:
    python 2_own_cnn.py

The steps run in order: model definition -> architecture figure -> training -> layer table ->
MACs/FLOPs -> design decisions -> ablation. Each training is done once and measured with CodeCarbon
(emissions/emissions.csv). If the result exists on disk it is not retrained; to retrain, delete
the corresponding summary file:
    models/own_cnn_64_summary.json  -> main model (64x64) is retrained
    models/own_cnn_48_summary.json  -> ablation variant (48x48) is retrained
The epoch limit is train_utils.MAX_EPOCHS; the limit used by each run is written to its summary file.
"""

import os
import textwrap

import matplotlib
matplotlib.use("Agg")  # draw to a file without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from data_utils import ROOT, HUBER_DELTA, load_split, print_table, print_section
from train_utils import (MODEL_DIR, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                         LR_PATIENCE, LR_FACTOR, EARLY_STOP_PATIENCE,
                         EMISSIONS_CSV, run_experiment, train_mean_age)


# =====================================================================================
# [x] Build the network yourself from layers (Conv2d, pooling, normalization, activation,
#     regression head). Adding residual or depthwise blocks is optional.
# =====================================================================================
# Input resolution: 64x64 (the 200x200 UTKFace images are downscaled).
#
# Depth — the receptive field (RF) of the last convolution should see the whole face (64 pixels):
#     Structure                 RF at last convolution   Meets the goal?
#     3 stages x 2 convolutions 32                       No, half of the face
#     4 stages x 1 convolution  38                       No
#     4 stages x 2 convolutions 68                       Yes  <- chosen
#     5 stages x 2 convolutions 140                      More than needed; the map also shrinks to 2x2
#
# Two consecutive 3x3 convolutions give the same RF as a single 5x5 convolution but are cheaper:
# 2 x 9 = 18 weights per channel pair instead of 25, plus an extra non-linearity (ReLU) in between.
#
# Width: the number of channels doubles at each stage (16 -> 32 -> 64 -> 128) while the resolution
# halves. This keeps the compute cost of each stage roughly balanced.
#
# Regression head: global average pooling -> a single Linear layer (128 -> 1).
# GAP reduces the 4x4x128 map to 128 numbers; instead of a large fully connected layer it works
# with very few parameters. The output bias is initialized to the mean train age:
# the model starts training from the prediction of the mean baseline (MAE_mean).

INPUT_SIZE = 64
STAGE_CHANNELS = [16, 32, 64, 128]


def conv_stage(c_in, c_out):
    """One stage: [Conv 3x3 -> BN -> ReLU] x 2 -> MaxPool. The resolution halves."""
    return nn.Sequential(
        # 1. Conv: 3x3, size preserved (stride=1, padding=1), channels c_in -> c_out.
        #    Since BN follows immediately, a bias is unnecessary (BN's β does this job).
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        # 2. BN: per-channel normalization (starts with γ=1, β=0)
        nn.BatchNorm2d(c_out),
        # 3. ReLU: non-linearity
        nn.ReLU(inplace=True),
        # 4. Conv: second layer at the same resolution, channels c_out -> c_out
        nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        # 5. BN
        nn.BatchNorm2d(c_out),
        # 6. ReLU
        nn.ReLU(inplace=True),
        # 7. MaxPool: halves H and W, no parameters
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class AgeCNN(nn.Module):
    """4 convolution stages + GAP + Linear: (N, 3, 64, 64) image -> (N,) age (years)."""

    def __init__(self, channels=STAGE_CHANNELS, initial_age=0.0):
        super().__init__()

        stages, c_in = [], 3  # RGB input
        for c_out in channels:
            stages.append(conv_stage(c_in, c_out))
            c_in = c_out
        self.features = nn.Sequential(*stages)   # 64x64 -> 32 -> 16 -> 8 -> 4x4

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (N, 128, 4, 4) -> (N, 128, 1, 1)
            nn.Flatten(),             # -> (N, 128)
            nn.Linear(c_in, 1),       # -> (N, 1)
        )
        nn.init.constant_(self.head[-1].bias, initial_age)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)  # (N, 1) -> (N,)


def check_model():
    """Builds the model and prints the output size of each stage and the parameter count for a sample input."""
    mean_age = load_split("train")["age"].mean()
    model = AgeCNN(initial_age=mean_age).eval()

    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    print(f"Input         : {tuple(x.shape)}")
    with torch.no_grad():
        for index, stage in enumerate(model.features, start=1):
            x = stage(x)
            print(f"Stage {index}       : {tuple(x.shape)}")
        output = model.head(x).squeeze(1)
    print(f"Output        : {tuple(output.shape)}  (initial prediction around ≈ {mean_age:.2f} years)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


# Architecture figure: input -> 4 stages -> GAP -> Linear -> age. The values are read from layer_table()
# (if the model changes, the figure changes too). Box height is proportional to the spatial size (H),
# so the resolution halving at each stage is visible. The stages go from light to dark on a blue ramp
# (depth order); the input and the head are neutral gray.

ARCHITECTURE_FIGURE = os.path.join(ROOT, "figures", "own_cnn_architecture.png")
STAGE_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]   # blue ramp: 250, 350, 450, 550
INK, INK_SECONDARY, NEUTRAL, SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#fcfcfb"


def draw_architecture(input_size=INPUT_SIZE):
    """Draws the architecture of our own CNN as a block diagram and saves it under figures/."""
    table = layer_table(AgeCNN(), input_size)
    table["Block"] = table["Layer"].str.split(" \\[").str[0]

    def shape_of(text):
        return [int(size) for size in text.split(" x ")]

    # (title, description, output size, parameters, MACs, RF, color)
    blocks = [("Input", "RGB image", [3, input_size, input_size], 0, 0, None, NEUTRAL)]
    for index, (name, rows) in enumerate(table[table["Block"] != "Head"].groupby("Block", sort=False)):
        blocks.append((name, "2 x [Conv 3x3 → BN → ReLU]\nMaxPool 2x2",
                       shape_of(rows["Output (C x H x W)"].iloc[-1]), rows["Params"].sum(),
                       rows["MACs"].sum(), rows[rows["Type"] == "Conv2d"]["RF"].iloc[-1], STAGE_COLORS[index]))
    head = table[table["Block"] == "Head"]
    blocks.append(("GAP", "Global average\npooling", shape_of(head["Output (C x H x W)"].iloc[0]), 0, 0,
                   None, NEUTRAL))
    linear = head[head["Type"] == "Linear"].iloc[0]
    blocks.append(("Linear", f"{STAGE_CHANNELS[-1]} → 1\nbias = train mean", [1], linear["Params"],
                   linear["MACs"], None, NEUTRAL))

    figure, axis = plt.subplots(figsize=(14, 5.2), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    box_width, gap, max_height = 1.5, 0.75, 3.0
    for position, (name, description, shape, params, macs, rf, color) in enumerate(blocks):
        x = position * (box_width + gap)
        spatial = shape[1] if len(shape) == 3 else 1
        height = 0.45 + (max_height - 0.45) * spatial / input_size
        axis.add_patch(plt.Rectangle((x, -height / 2), box_width, height, facecolor=color,
                                     edgecolor=SURFACE, linewidth=2))
        axis.text(x + box_width / 2, max_height / 2 + 0.55, name, ha="center", fontsize=11,
                  fontweight="bold", color=INK)
        axis.text(x + box_width / 2, max_height / 2 + 0.18, description, ha="center", va="center",
                  fontsize=7.5, color=INK_SECONDARY)
        details = [" x ".join(map(str, shape))]
        if params:
            details.append(f"{params:,} parameters")
        if macs:
            details.append(f"{macs / 1e6:.1f} M MACs" if macs >= 1e5 else f"{macs:,} MACs")
        if rf:
            details.append(f"RF {rf} px")
        axis.text(x + box_width / 2, -max_height / 2 - 0.25, "\n".join(details), ha="center", va="top",
                  fontsize=8.5, color=INK, linespacing=1.5)
        if position < len(blocks) - 1:
            axis.annotate("", xy=(x + box_width + gap - 0.08, 0), xytext=(x + box_width + 0.08, 0),
                          arrowprops={"arrowstyle": "-|>", "color": INK_SECONDARY, "linewidth": 1.4})

    x_end = len(blocks) * (box_width + gap) - gap
    axis.annotate("", xy=(x_end + 0.6, 0), xytext=(x_end + 0.08, 0),
                  arrowprops={"arrowstyle": "-|>", "color": INK_SECONDARY, "linewidth": 1.4})
    axis.text(x_end + 0.7, 0, "age\n(years)", va="center", fontsize=10, fontweight="bold", color=INK)

    axis.set_xlim(-0.3, x_end + 1.6)
    axis.set_ylim(-max_height / 2 - 1.6, max_height / 2 + 0.9)
    axis.axis("off")
    total_params, total_macs = table["Params"].sum(), table["MACs"].sum()
    axis.set_title(f"Our own CNN (AgeCNN): {total_params:,} parameters, {total_macs / 1e6:.1f} M MACs / "
                   f"image, input {input_size}x{input_size}", loc="left", fontsize=12, color=INK)
    figure.text(0.01, 0.02, "Box height is proportional to the spatial size (H). RF: receptive field at the "
                "last convolution of the stage. The convolutions have no bias (BN follows them).",
                fontsize=8, color=INK_SECONDARY)
    figure.tight_layout()

    os.makedirs(os.path.dirname(ARCHITECTURE_FIGURE), exist_ok=True)
    figure.savefig(ARCHITECTURE_FIGURE, facecolor=SURFACE)
    plt.close(figure)
    print(f"Saved: {ARCHITECTURE_FIGURE}")


# =====================================================================================
# [x] Train the model from scratch; do not use pretrained weights.
# =====================================================================================
# - Initialization: the weights start from PyTorch's default random initialization (kaiming_uniform_(a=√5)
#   for Conv2d and Linear); no pretrained weights are loaded. Rationale: design decisions section.
# - The training code is in train_utils.py; the standard models of step 3 are trained with the same
#   code and the same settings (fairness is guaranteed in code):
#     Huber loss (d = 5 years), AdamW (lr 1e-3, weight decay 1e-2), ReduceLROnPlateau,
#     early stopping on val MAE, only horizontal flips, normalization computed from train.
# - Trained only on train and measured on validation at the end of each epoch. The test set is not touched.
# - Training and inference are measured with CodeCarbon; each run is written to emissions/emissions.csv.
# - The model is trained once: if the result exists in models/own_cnn_64_summary.json it is not retrained.

MAIN_RUN = "own_cnn_64"


def build_own_cnn():
    """Model ready for training: the output bias starts at the mean train age."""
    return AgeCNN(initial_age=train_mean_age())


def train_own_cnn():
    """Trains and measures the main model (64x64), or reads its saved result."""
    return run_experiment(MAIN_RUN, build_own_cnn, INPUT_SIZE)


# =====================================================================================
# [x] Prepare a table for every layer: output size, parameter count, receptive field.
# =====================================================================================
# - Output size: a forward hook is attached to every layer and measured with a 1-image sample input
#   (not computed by hand, read from the real model).
# - Parameters: the layer's own weights. Conv: c_in*c_out*3*3 (no bias), BN: 2*c (γ, β),
#   Linear: 128 + 1. ReLU, pooling and Flatten have no parameters.
# - Receptive field (RF): the side length (pixels) of the input area seen by one output pixel.
#   Computed layer by layer with these two rules:
#       RF   <- RF + (kernel - 1) * jump
#       jump <- jump * stride        (jump: distance in the input between two neighboring output pixels)
#   BN and ReLU do not change the RF. GAP covers the whole 4x4 map like a single kernel.

LAYER_TABLE_CSV = os.path.join(MODEL_DIR, "own_cnn_layers.csv")


def layer_name(module_name):
    """'features.2.3' -> 'Stage 3 [3]',  'head.2' -> 'Head [2]'."""
    parts = module_name.split(".")
    if parts[0] == "features":
        return f"Stage {int(parts[1]) + 1} [{parts[2]}]"
    return f"Head [{parts[1]}]"


def layer_macs(module, output_shape):
    """Number of multiply-accumulates (MACs) of the layer per image. Only Conv2d and Linear are counted."""
    if isinstance(module, nn.Conv2d):
        kernel_h, kernel_w = module.kernel_size
        macs_per_output = (module.in_channels // module.groups) * kernel_h * kernel_w
        return int(np.prod(output_shape)) * macs_per_output
    if isinstance(module, nn.Linear):
        return module.in_features * module.out_features
    return 0


def layer_table(model, input_size=INPUT_SIZE):
    """Returns the output size, parameter count and receptive field of every layer."""
    layers = [(name, module) for name, module in model.named_modules()
              if len(list(module.children())) == 0]

    # Measure the output sizes with a real forward pass.
    output_shapes = {}
    hooks = [module.register_forward_hook(
                 lambda _module, _input, output, name=name: output_shapes.update({name: tuple(output.shape[1:])}))
             for name, module in layers]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size))
    for hook in hooks:
        hook.remove()

    rows, rf, jump, previous_shape = [], 1, 1, (3, input_size, input_size)
    for name, module in layers:
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
            kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            rf, jump = rf + (kernel - 1) * jump, jump * stride
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            kernel = previous_shape[1]  # the whole map
            rf, jump = rf + (kernel - 1) * jump, jump * kernel

        shape = output_shapes[name]
        rows.append({
            "Layer": layer_name(name),
            "Type": type(module).__name__,
            "Output (C x H x W)": " x ".join(str(size) for size in shape),
            "Params": sum(p.numel() for p in module.parameters()),
            "MACs": layer_macs(module, shape),
            "RF": rf,
        })
        previous_shape = shape
    return pd.DataFrame(rows)


def report_layers():
    """Prints the layer table to the console and saves it to models/own_cnn_layers.csv."""
    model = AgeCNN()
    table = layer_table(model)

    rows = [list(table.columns)]
    for _, layer in table.iterrows():
        rf_text = f"{layer['RF']}" + (" (whole image)" if layer["RF"] >= INPUT_SIZE else "")
        rows.append([layer["Layer"], layer["Type"], layer["Output (C x H x W)"],
                     f"{layer['Params']:,}", f"{layer['MACs']:,}", rf_text])
    print_table(f"Layer table (input 3 x {INPUT_SIZE} x {INPUT_SIZE}, MACs per image):", rows)

    os.makedirs(MODEL_DIR, exist_ok=True)
    table.to_csv(LAYER_TABLE_CSV, index=False)
    print(f"\nSaved: {LAYER_TABLE_CSV}")
    return model, table


# =====================================================================================
# [x] Report the total number of parameters and the MACs/FLOPs per image.
# =====================================================================================
# - MAC (multiply-accumulate) = one multiplication + one addition. Only Conv2d and Linear are counted:
#       Conv2d : (output C x H x W) * (c_in * 3 * 3)
#       Linear : input * output
#   BN is ignored because at inference it can be folded into the weights of the previous convolution,
#   ReLU and pooling because they are a single operation per element (common convention).
# - FLOPs = 2 x MACs (each MAC is one multiplication + one addition).
# - The hand-computed value is verified with PyTorch's own counter (torch.utils.flop_counter).
# - Model size: parameter count x 4 bytes (float32).


def report_complexity(model, table):
    """Prints the total parameters, model size and MACs/FLOPs per image."""
    total_params = int(table["Params"].sum())
    total_macs = int(table["MACs"].sum())
    assert total_params == sum(p.numel() for p in model.parameters())

    # Independent check: PyTorch counts the FLOPs in convolutions and matrix multiplications.
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model.eval()(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
    torch_flops = counter.get_total_flops()

    print(f"""
Model complexity (per image, input 3 x {INPUT_SIZE} x {INPUT_SIZE}):
  Total parameters : {total_params:,}
  Model size       : {total_params * 4 / 1024 ** 2:.2f} MB (float32)
  MACs             : {total_macs:,}  (≈ {total_macs / 1e6:.1f} M)
  FLOPs (2 x MACs) : {2 * total_macs:,}  (≈ {2 * total_macs / 1e6:.1f} M)
  PyTorch counter  : {torch_flops:,} FLOPs  ->  {'matches' if torch_flops == 2 * total_macs else 'DIFFERENT'}""")

    table["Block"] = table["Layer"].str.split(" \\[").str[0]
    summary = table.groupby("Block", sort=False).agg(
        Params=("Params", "sum"), MACs=("MACs", "sum"), RF=("RF", "last"))
    summary["Param. share (%)"] = (100 * summary["Params"] / total_params).round(1)
    summary["MAC share (%)"] = (100 * summary["MACs"] / total_macs).round(1)
    print("\nParameters, MACs and last RF per stage:")
    print(summary.to_string())


# =====================================================================================
# [x] Justify every design decision: depth/width, kernel size, input resolution, loss,
#     optimizer, LR schedule, augmentation, regularization, stopping criterion.
# =====================================================================================
# For each decision, the choice, the rationale and the alternative considered are kept in the list below.
# The architecture decisions (depth, width, resolution) are also backed by numbers: the parameters,
# MACs and RF of the alternative architectures are actually computed with layer_table().
# Main goal: a network that predicts age well (low val MAE) but does as little compute as possible (low energy).

DESIGN_DECISIONS = [
    ("Depth",
     "4 stages x 2 convolutions",
     "The RF of the last convolution is 68 pixels; it sees the whole 64x64 face. 3 stages see only "
     "half of it (RF 32); 5 stages enlarge the RF unnecessarily and shrink the map to 2x2.",
     "3 and 5 stages (cost table below)"),
    ("Width",
     "16 -> 32 -> 64 -> 128 channels",
     "The channels double while the resolution halves; stages 2-4 each do the same number of MACs, "
     "so no stage becomes a bottleneck. Starting with 16 channels keeps the first stage, at full "
     "resolution and the most expensive place, cheap.",
     "Half width (8..64) and double width (32..256)"),
    ("Kernel size",
     "3x3, two in a row",
     "Two 3x3 give the same RF as one 5x5; they use 18 weights per channel pair instead of 25 "
     "(~28% fewer parameters and MACs) and add an extra ReLU in between.",
     "A single 5x5 convolution"),
    ("Input resolution",
     f"{INPUT_SIZE}x{INPUT_SIZE} (downscaled from 200x200)",
     "MACs are proportional to the number of pixels: 128x128 is four times and 200x200 about ten times "
     "more expensive. At 64x64 the face shape and proportions are kept; some fine wrinkle detail is lost. "
     "For an energy-focused project a low resolution is the starting point.",
     "48, 96, 128 pixels (ablation candidate)"),
    ("Regression head",
     "GAP + Linear(128 -> 1), bias = train mean",
     "Thanks to GAP the head has only 129 parameters; with 4x4x128 -> Linear it would have 2,049 and "
     "become position-sensitive. Initializing the bias with the train mean makes training start from "
     "the MAE_mean point and avoids large gradients in the first epochs.",
     "Flatten + large fully connected layer"),
    ("Weight initialization",
     "PyTorch default: kaiming_uniform_(a=√5) for Conv2d/Linear; BN γ=1, β=0",
     "This is a Kaiming variant but does not use the optimal gain for ReLU: the weights come from "
     "U(-1/√fan_in, 1/√fan_in), std = 1/√(3·fan_in); √6 ≈ 2.45 times smaller than the √(2/fan_in) "
     "recommended by Kaiming for ReLU. This difference is not critical here: BN follows every "
     "convolution and rescales the output per channel, so it is independent of the scale of the "
     "convolution weights. Therefore the default initialization was kept.",
     "kaiming_normal_(nonlinearity='relu') (would be preferred in a network without BN)"),
    ("Loss",
     f"Huber, d = {HUBER_DELTA:g} years",
     "The age labels are noisy and long-tailed; Huber limits the effect of large errors (compared to MSE) "
     "and gives a smooth gradient for small errors (compared to L1). Detailed comparison in step 1.",
     "MSE, L1"),
    ("Optimizer",
     f"AdamW, lr = {LEARNING_RATE:g}, weight decay = {WEIGHT_DECAY:g}",
     "For a small CNN with BN trained from scratch, Adam's default lr is stable and needs little tuning; "
     "this matters because every trial is expensive on a CPU. AdamW decouples weight decay from the "
     "gradient: at every step the weights shrink by lr x wd. wd = 1e-2 (AdamW default) gives a shrink "
     "of 1e-5 per step; with 1e-4 this would be 1e-7 and practically have no effect.",
     "SGD + momentum (requires more lr tuning)"),
    ("LR schedule",
     f"ReduceLROnPlateau: if val MAE does not improve for {LR_PATIENCE} epochs, lr x {LR_FACTOR:g}",
     "Because early stopping is used, the total number of epochs is not known in advance; fixed-schedule "
     "schedules such as cosine/step do not fit this. The plateau schedule lowers the lr only when needed.",
     "Cosine annealing, step decay"),
    ("Augmentation",
     "Only random horizontal flips",
     "A mirrored face has the same age; the operation is free. Since UTKFace images are already "
     "aligned and cropped, large rotations/crops would produce unrealistic samples. Color jitter "
     "was not used because it could change skin tone and affect group-wise bias.",
     "Random crop, rotation, color jitter"),
    ("Regularization",
     f"Weight decay ({WEIGHT_DECAY:g}) + BN + horizontal flips + early stopping",
     "Weight decay limits overfitting by keeping the weights small. BN stabilizes training and has a "
     "mild regularization effect. The few parameters in the head (GAP) and early stopping limit "
     "memorization. Dropout was not added; it is the next candidate if the train/val MAE gap stays large.",
     "Dropout (in the head), stronger weight decay"),
    ("Stopping criterion",
     f"Stop if val MAE does not improve for {EARLY_STOP_PATIENCE} epochs, at most {MAX_EPOCHS} epochs",
     f"Since the primary metric is MAE, stopping also looks at val MAE and the model with the best val MAE "
     f"is kept. The patience ({EARLY_STOP_PATIENCE}) is larger than the lr patience ({LR_PATIENCE}): the lr "
     f"is lowered and tried at least once before stopping. The upper limit bounds the training time and energy on the CPU.",
     "Fixed number of epochs, stopping on val loss"),
    ("Batch size",
     f"{BATCH_SIZE}",
     "Large enough for stable BN statistics; balanced in terms of CPU memory and the number of "
     "updates per epoch (~127).",
     "64, 256"),
]

ARCHITECTURE_ALTERNATIVES = [
    # (name, channels, input resolution)
    ("Chosen model", STAGE_CHANNELS, INPUT_SIZE),
    ("Depth: 3 stages", [16, 32, 64], INPUT_SIZE),
    ("Depth: 5 stages", [16, 32, 64, 128, 256], INPUT_SIZE),
    ("Width: half", [8, 16, 32, 64], INPUT_SIZE),
    ("Width: double", [32, 64, 128, 256], INPUT_SIZE),
    ("Resolution: 48", STAGE_CHANNELS, 48),
    ("Resolution: 96", STAGE_CHANNELS, 96),
    ("Resolution: 128", STAGE_CHANNELS, 128),
]


def report_design_decisions():
    """Prints the design decisions and the cost of the alternative architectures."""
    print("Design decisions and rationale:")
    for topic, choice, reason, alternative in DESIGN_DECISIONS:
        print(f"\n{topic}")
        print(f"  Choice      : {choice}")
        print(textwrap.fill(reason, width=100, initial_indent="  Rationale   : ",
                            subsequent_indent=" " * 15))
        print(f"  Alternative : {alternative}")

    rows = [["Architecture", "Channels", "Input", "Params", "MACs (M)", "MAC ratio", "Last conv RF", "Last map"]]
    base_macs = None
    for name, channels, size in ARCHITECTURE_ALTERNATIVES:
        table = layer_table(AgeCNN(channels=channels), input_size=size)
        macs = table["MACs"].sum()
        base_macs = base_macs or macs
        last_conv = table[table["Type"] == "Conv2d"].iloc[-1]
        last_map = table[table["Type"] == "MaxPool2d"].iloc[-1]["Output (C x H x W)"]
        rows.append([name, "-".join(map(str, channels)), f"{size}x{size}", f"{table['Params'].sum():,}",
                     f"{macs / 1e6:.1f}", f"{macs / base_macs:.2f}x", f"{last_conv['RF']} / {size}", last_map])
    print_table("Cost of the alternative architectures (per image):", rows)
    print("""
  - Depth: the RF of 3 stages stays at half of the face; 5 stages add few MACs but roughly
    quadruple the parameters and shrink the map to 2x2.
  - Width: MACs and parameters grow with the square of the channel count (double width ~4x).
  - Resolution: the parameters do not change, MACs grow in proportion to the number of pixels.""")


# =====================================================================================
# [x] Do at least 1 ablation: change a single design choice and report its effect on
#     both the error and the energy.
# =====================================================================================
# The single decision changed: input resolution 64x64 -> 48x48.
#   - Architecture, parameter count, seed and all training settings are the same. Since the model
#     ends with GAP, it runs with the same number of weights at both resolutions.
#   - Because MACs drop in proportion to the pixels (0.56x), it is the single most effective variable for energy.
#   - Question: how much energy does lowering the resolution save, and how much does the MAE increase in return?
#
#   - The baseline row is the main training of item 2 itself (own_cnn_64); the same model is not
#     trained a second time for the ablation (retraining with the same settings + seed wastes energy).
#     Only the 48x48 variant (own_cnn_48) is trained for the ablation.
#
# Measurement (train_utils.run_experiment, the same code for both runs):
#   - Error : val MAE / RMSE / R² at the best epoch (the test set is not touched).
#   - Energy: CodeCarbon; training and inference are written to emissions/emissions.csv as separate runs.
#             Inference is measured by predicting the val set INFERENCE_REPEATS times.
#   - No other load should run on the machine during measurement and the computer should be plugged in.

ABLATION_RUN, ABLATION_SIZE = "own_cnn_48", 48
ABLATION_CSV = os.path.join(MODEL_DIR, "ablation_results.csv")


def run_ablation(base):
    """Trains and measures the 48x48 variant and compares it with the main model (base)."""
    ablation = run_experiment(ABLATION_RUN, build_own_cnn, ABLATION_SIZE)
    report_ablation(base, ablation)


def report_ablation(base, ablation):
    """Prints the error and energy of the two runs side by side and saves the percentage changes."""
    for summary in (base, ablation):
        summary["MACs_M"] = layer_table(AgeCNN(), input_size=summary["input_size"])["MACs"].sum() / 1e6

    metrics = [("MACs (M)", "MACs_M"), ("val MAE", "val_MAE"), ("val RMSE", "val_RMSE"), ("val R²", "val_R2"),
               ("Training (min)", "train_minutes"), ("Training energy (Wh)", "train_energy_Wh"),
               ("Training CO2 (g)", "train_co2_g"), ("Inference (ms/image)", "infer_ms_per_image"),
               ("Inference energy (mWh/1000 images)", "infer_mWh_per_1000")]

    rows = [["Measure", f"Baseline ({base['input_size']}px)", f"Ablation ({ablation['input_size']}px)", "Change"]]
    records = []
    for label, key in metrics:
        change = 100 * (ablation[key] - base[key]) / base[key]
        rows.append([label, f"{base[key]:.3f}", f"{ablation[key]:.3f}", f"{change:+.1f}%"])
        records.append({"Measure": label, "Baseline": base[key], "Ablation": ablation[key], "Change (%)": change})
    rows.append(["Epochs (best / total)", f"{base['best_epoch']} / {base['epochs_run']}",
                 f"{ablation['best_epoch']} / {ablation['epochs_run']}", ""])
    print_table("Ablation results (input resolution):", rows)
    pd.DataFrame(records).to_csv(ABLATION_CSV, index=False)

    def saving(key):
        return 100 * (1 - ablation[key] / base[key])

    print(f"""
Summary: lowering the resolution {base['input_size']} -> {ablation['input_size']} changes val MAE by \
{ablation['val_MAE'] - base['val_MAE']:+.2f} years while
         reducing training energy by {saving('train_energy_Wh'):.1f}% and inference energy by {saving('infer_mWh_per_1000'):.1f}% \
(theoretical reduction in MACs {saving('MACs_M'):.1f}%).
Note   : Training energy also depends on the number of epochs (early stopping); therefore inference energy
         is more directly comparable with MACs. Measured energy depends on the hardware and power mode;
         MACs are a hardware-independent indicator.
Saved: {ABLATION_CSV}
All measurements: {EMISSIONS_CSV}""")


def main():
    """Runs the items of step 2 in checklist order.

    Since the trainings (main model and ablation) take long and consume energy, each is done once;
    if their results exist on disk, the saved summaries are used (train_utils.run_experiment).
    """
    print_section("1) Model definition")
    check_model()

    print_section("1b) Architecture figure")
    draw_architecture()

    print_section("2) Training from scratch (energy measured)")
    base = train_own_cnn()

    print_section("3) Layer table")
    model, table = report_layers()

    print_section("4) Total parameters and MACs/FLOPs")
    report_complexity(model, table)

    print_section("5) Design decisions")
    report_design_decisions()

    print_section("6) Ablation: input resolution")
    run_ablation(base)


if __name__ == "__main__":
    main()
