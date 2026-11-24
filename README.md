# CSE655 Term Project — Green CNNs for Age Regression (UTKFace)

Burak Sarıkaya · CSE 655 Deep Learning and Applications, Fall 2026, Gebze Technical University

A small CNN, trained from scratch, predicts a person's age from a face image. It is compared with a fine-tuned
ResNet-18 and MobileNetV3-Large, and the energy use and CO2 of every run are measured with CodeCarbon.
The technical report explains the design and the results; this file explains how to reproduce every number.

## 1. Environment

- Python 3.15.2, Windows 11, CPU only (Intel Core i7-12650H, 16 GB RAM). No GPU is used.
- Install the exact versions:

```
pip install -r requirements.txt
```

`requirements.txt` pins torch 2.15.0+cpu, torchvision 0.29.2+cpu, codecarbon 3.3.2, numpy, pandas,
scikit-learn, matplotlib and pillow. The ImageNet weights of ResNet-18 and MobileNetV3-Large
(`IMAGENET1K_V1`) are downloaded by torchvision on first use.

## 2. Data

The images are **not included in this repository** (`UTKFace/` and the image cache `data/cache/` are in
`.gitignore`): UTKFace may only be shared by its authors, for non-commercial research use.

1. Open the official page https://susanqq.github.io/UTKFace/ and download the **"Aligned & Cropped Faces"**
   version (not "In-the-wild Faces" and not the landmarks file).
2. Extract the archive into the project folder, so that the images are **directly** in `UTKFace/`, next to the
   scripts. For a `.tar.gz` archive, from the project folder:

   ```
   tar -xzf UTKFace.tar.gz
   ```

   Expected layout (no extra sub-folder such as `UTKFace/UTKFace/`):

   ```
   CSE655_Term_Project_Green_CNN_Regression2/
   ├── 1_dataset_and_problem.py
   ├── ...
   └── UTKFace/
       ├── 100_0_0_20170112213500903.jpg.chip.jpg
       ├── ...
       └── 8_1_0_20170109203907694.jpg.chip.jpg
   ```

3. Check: `UTKFace/` should contain **23,279 files** named `[age]_[gender]_[race]_[date&time].jpg.chip.jpg`.
   On PowerShell: `(Get-ChildItem UTKFace).Count`; on bash: `ls UTKFace | wc -l`.

The folder name and location are fixed in the code (`IMAGE_DIR` in `1_dataset_and_problem.py`). The list of the
images actually used is `data/utkface_clean.csv` (23,213 files); `data/utkface_rejected.csv` lists the 66
removed files and why. Step 1 rescans the folder only with `--rebuild`. The image cache `data/cache/*.npz` is
built automatically from `UTKFace/` the first time a script needs the images.

## 3. Seeds and fixed choices

- `SEED = 42` (`data_utils.py`) fixes the split, the weight initialization (`torch.manual_seed`) and the
  batch shuffling and random flips (`torch.Generator().manual_seed`) in `train_utils.train_model`.
- The split is written once to `data/splits.csv` (70 / 15 / 15, stratified by age group × gender) and reused by
  every model.
- The test set can only be opened through `data_utils.load_test_once`, which records each use in
  `data/test_usage.json`; a configuration cannot be evaluated on the test set twice.

## 4. Run the steps

Run the five scripts in order from this folder. Each script prints its tables to the console.

| Step | Command | What it does | Report |
|---|---|---|---|
| 1 | `python 1_dataset_and_problem.py` | Cleaning (23,279 → 23,213 images), fixed split, loss/metric comparison, mean and median baselines, baseline group errors | Step 1 |
| 2 | `python 2_own_cnn.py` | AgeCNN architecture figure, training at 64 px, per-layer table, MACs, design costs, 48-px ablation | Step 2, Tables 1–3, Appendix A |
| 3 | `python 3_comparing_standard_models.py` | Fine-tuning ResNet-18 and MobileNetV3-Large, pretraining-footprint estimate, fairness check | Step 3, Tables 4–5 |
| 4 | `python 4_footprint.py` | Footprint-reduction experiments, per-model footprint, project total, hardware | Step 4, Tables 6–8 |
| 5 | `python 5_score_control.py` | Flip TTA and ensemble, acceptance check, test evaluation, results table, error–CO2 plot, bias tables | Step 5, Tables 9–11, Figure 1 |

Options of step 1: `--rebuild` rescans the `UTKFace/` folder, `--resplit` rebuilds the split (do not use it
after models have been trained, or the comparison is no longer on the same split).

In `5_score_control.py`, `USE_TEST_SET = False` runs every check on the validation set only (pre-check);
`USE_TEST_SET = True` (the submitted setting) evaluates the final configurations once on the test set.

## 5. Two ways to reproduce

**A. Re-print the reported numbers (no retraining, almost no energy).**
Everything except the model weights is included: run summaries, training histories, CodeCarbon logs (both
sessions), the one-time test results and predictions, and the latency, bias and footprint-reduction outputs.
Every training and measurement is skipped when its output already exists (`models/*_summary.json`,
`models/reduction/reduction_results.json`, `models/latency/*.json`, `models/test_results/*.json`), so running the
steps in order only re-prints the tables from the saved files.

The weight files (`*.pt`, about 200 MB in total) are **not in the repository** (`.gitignore`).
- Steps 1–3 re-print their tables without them.
- Steps 4 and 5 open the checkpoints to report the model file size (MB), so they need the weights in `models/`:
  <!-- TODO: link to the weights, or remove this line --> download them from [TODO: link] and copy them into
  `models/`, or retrain the models with option B.
- The test set cannot be evaluated again (`data/test_usage.json` records that every configuration has used it);
  step 5 reads the saved results in `models/test_results/` instead.

**B. Re-run everything from scratch.**
Move `models/`, `emissions/`, `data/test_usage.json` and `data/cache/` out of the folder, then run steps 1–5.
This also recreates all weight files. The archived first session (`archive/`) stays in the project total.
Errors will match closely (same seed and split), but energy, time and CO2 will differ: they depend on the
machine, its power mode and background load. The full run takes about 2 hours of CPU time on the laptop above.
Measurement protocol used for the reported runs: plugged in, fixed power mode, sleep off, minimal background
load, one run at a time.

## 6. Outputs

| Path | Content |
|---|---|
| `data/splits.csv` | The fixed train / val / test split |
| `data/utkface_clean.csv`, `data/utkface_rejected.csv` | Kept images and the 66 removed files with the reason |
| `data/test_usage.json` | When each configuration opened the test set |
| `models/own_cnn_64.pt`, `models/own_cnn_48.pt` | **Weights of the final model** (the ensemble averages these two, 1.14 MB each). *Not in the repository* |
| `models/resnet18_imagenet_64.pt`, `models/mobilenetv3_large_imagenet_64.pt` | Fine-tuned standard models. *Not in the repository* |
| `models/*_history.csv`, `models/*_summary.json` | Per-epoch training history and final metrics of each run (and of each TTA / ensemble configuration) |
| `models/test_results/` | One-time test metrics and predictions of each configuration |
| `models/latency/`, `models/bias/`, `models/reduction/` | Batch-1 latency, validation predictions for the bias analysis, footprint-reduction measurements |
| `emissions/emissions.csv` | CodeCarbon log of every run of the second (reported) session |
| `archive/session1_2026-11-22/emissions/emissions.csv` | CodeCarbon log of the first, archived session (included in the project total) |
| `results/results.csv` (copy of `results/results_table_test.csv`) | Main results table (Table 10) |
| `results/bias_by_group_test.csv` | Test error per age group, gender and ethnicity (Table 11) |
| `figures/error_vs_co2_test.png` | Error vs. CO2 plot with the Pareto front (Figure 1) |
| `figures/own_cnn_architecture.png` | Architecture figure (Step 2) |
| `figures/learning_curves.png` | Validation MAE per epoch (Appendix B), drawn from `models/*_history.csv` outside the five scripts |

## 7. Notes

- The final model is the ensemble of the 64-px and 48-px AgeCNN without TTA (`OWN_FINAL_RUN` in
  `5_score_control.py`). It was fixed on the validation set before the test set was opened.
- `archive/session2_20epoch/` holds the summaries and histories of the earlier 20-epoch trainings, which were
  replaced by the final 40-epoch runs. Their energy is counted in the project total. `archive/session1_2026-11-22/`
  holds the first measurement session, with per-run notes on its measurement conditions (`emissions/run_notes.csv`).
- Not in the repository (`.gitignore`): the UTKFace images, the image cache `data/cache/`, all weight files `*.pt`,
  `.venv/` and `__pycache__/`.
- AI tools were used in this project; see the separate AI Usage Report.
