Results table — TEST

| Model | Params | MFLOPs/img | Test MAE | RMSE | R² | Train kWh | Train kg CO2 | Project kg CO2 | ms/img | Latency ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Own CNN (64px) | 294,129 | 107.3 | 6.04 | 8.56 | 0.806 | 0.0082 | 0.0038 | 0.0141 | 0.582 | 1.94 |
| Own CNN (48px) | 294,129 | 60.4 | 5.99 | 8.87 | 0.792 | 0.0058 | 0.0027 | 0.0066 | 0.310 | 1.37 |
| Own CNN (64px) + flip TTA | 294,129 | 214.7 | 5.77 | 8.19 | 0.823 | 0.0082 | 0.0038 | 0.0003 | 1.170 | 3.33 |
| Own CNN (48px) + flip TTA | 294,129 | 120.8 | 5.74 | 8.50 | 0.809 | 0.0058 | 0.0027 | 0.0001 | 0.607 | 2.52 |
| Own CNN ensemble (64+48px) (final) | 588,258 | 167.7 | 5.65 | 8.23 | 0.821 | 0.0140 | 0.0065 | 0.0002 | 0.883 | 3.18 |
| Own CNN ensemble (64+48px) + flip TTA | 588,258 | 335.5 | 5.49 | 8.01 | 0.831 | 0.0140 | 0.0065 | 0.0004 | 1.779 | 6.05 |
| ResNet-18 | 11,177,025 | 296.1 | 5.12 | 7.61 | 0.847 | 0.0369 | 0.0171 | 0.0622 | 1.575 | 6.28 |
| MobileNetV3-Large | 4,203,313 | 40.2 | 5.29 | 7.93 | 0.834 | 0.0246 | 0.0114 | 0.0397 | 0.654 | 5.85 |
| ResNet-18 + flip TTA | 11,177,025 | 592.2 | 4.92 | 7.33 | 0.858 | 0.0369 | 0.0171 | 0.0005 | 2.403 | 12.17 |
| MobileNetV3-Large + flip TTA | 4,203,313 | 80.4 | 5.11 | 7.69 | 0.844 | 0.0246 | 0.0114 | 0.0002 | 1.130 | 10.97 |
| Mean baseline | - | - | 15.00 | 19.46 | -0.000 | - | - | 0.0000 | - | - |

ms/img: batch 512 throughput; Latency ms: median at batch 1. Runs belonging to no model (footprint-reduction experiments, bias-analysis predictions): 0.0057 kg CO2eq. Project total (all sessions): 0.1301 kg CO2eq.
