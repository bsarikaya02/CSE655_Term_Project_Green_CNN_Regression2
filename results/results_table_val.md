Results table — VALIDATION (pre-check, test set not opened)

| Model | Params | MFLOPs/img | Val MAE | RMSE | R² | Train kWh | Train kg CO2 | Project kg CO2 | ms/img | Latency ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Own CNN (64px) | 294,129 | 107.3 | 5.81 | 7.99 | 0.829 | 0.0082 | 0.0038 | 0.0141 | 0.582 | 1.94 |
| Own CNN (48px) | 294,129 | 60.4 | 5.81 | 8.17 | 0.822 | 0.0058 | 0.0027 | 0.0066 | 0.310 | 1.37 |
| Own CNN (64px) + flip TTA | 294,129 | 214.7 | 5.60 | 7.71 | 0.841 | 0.0082 | 0.0038 | 0.0003 | 1.170 | 3.33 |
| Own CNN (48px) + flip TTA | 294,129 | 120.8 | 5.54 | 7.82 | 0.836 | 0.0058 | 0.0027 | 0.0001 | 0.607 | 2.52 |
| Own CNN ensemble (64+48px) (final) | 588,258 | 167.7 | 5.44 | 7.59 | 0.846 | 0.0140 | 0.0065 | 0.0002 | 0.883 | 3.18 |
| Own CNN ensemble (64+48px) + flip TTA | 588,258 | 335.5 | 5.31 | 7.41 | 0.853 | 0.0140 | 0.0065 | 0.0004 | 1.779 | 6.05 |
| ResNet-18 | 11,177,025 | 296.1 | 4.94 | 7.13 | 0.864 | 0.0369 | 0.0171 | 0.0622 | 1.575 | 6.28 |
| MobileNetV3-Large | 4,203,313 | 40.2 | 5.19 | 7.46 | 0.851 | 0.0246 | 0.0114 | 0.0397 | 0.654 | 5.85 |
| ResNet-18 + flip TTA | 11,177,025 | 592.2 | 4.76 | 6.87 | 0.874 | 0.0369 | 0.0171 | 0.0005 | 2.403 | 12.17 |
| MobileNetV3-Large + flip TTA | 4,203,313 | 80.4 | 5.09 | 7.29 | 0.858 | 0.0246 | 0.0114 | 0.0002 | 1.130 | 10.97 |
| Mean baseline | - | - | 14.99 | 19.34 | -0.000 | - | - | 0.0000 | - | - |

ms/img: batch 512 throughput; Latency ms: median at batch 1. Runs belonging to no model (footprint-reduction experiments, bias-analysis predictions): 0.0057 kg CO2eq. Project total (all sessions): 0.1300 kg CO2eq.
