# Results

All values below are transcribed from saved notebook outputs.

## Held-out test forecasting results

| Model | MAE (kWh) | RMSE (kWh) |
|---|---:|---:|
| Mean of training + validation | 0.644 | 0.783 |
| Yesterday (24-hour lag) | 0.581 | 0.843 |
| Last week (168-hour lag) | 0.629 | 0.898 |
| 7-day same-hour mean | 0.481 | 0.673 |
| LightGBM, no weather | 0.462 | 0.644 |
| LightGBM, with weather | 0.462 | 0.642 |

The strict subset with real readings and fully real 168-hour history contains 1,835 test hours. On this subset, LightGBM with weather reports MAE 0.460 kWh and RMSE 0.637 kWh.

## Validation feature ablation

| Feature set | MAE (kWh) | RMSE (kWh) |
|---|---:|---:|
| Calendar only | 0.368 | 0.499 |
| Calendar + history | 0.340 | 0.471 |
| Calendar + history + weather | 0.344 | 0.470 |

Weather features did not improve overall validation MAE in this experiment. They reduced peak MAE slightly, but improved daily MAE on 45% of validation days and 49% of test days.

## Load-shifting simulation

The 50% flexibility laundry + heater/AC scenario reports:
- 1.035 kWh/day shifted
- €0.104 simulated savings per day
- 2.224% simulated bill savings
- 15.026% reduction in peak-window energy

Simulation assumptions: peak tariff €0.25/kWh, lower tariff €0.15/kWh, peak window 18:00–21:00, target hours 10:00–17:00 and 22:00–23:00, and a 1 kWh receiving-hour cap.
