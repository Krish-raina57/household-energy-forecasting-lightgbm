# Household Energy Consumption Forecasting & Load Optimization

A time-series machine-learning project that forecasts hourly household electricity demand using LightGBM and explores shifting flexible loads away from peak-tariff hours.

## Project highlights

- Processed **2,075,259 minute-level records** and aggregated them into **34,589 hourly records**.
- Handled **25,979 missing minute readings (1.25%)** and flagged **432 hourly records (1.25%)** as imputed.
- Built calendar, lag, rolling-window and historical-weather features.
- Evaluated LightGBM against simple historical baselines using chronological train/validation/test periods.
- Simulated flexible-load shifting over **88 complete days** using a two-level time-of-use tariff.

## Results recorded in the notebook

### Forecasting — held-out test period

| Model | MAE (kWh) | RMSE (kWh) |
|---|---:|---:|
| 7-day same-hour average | 0.481 | 0.673 |
| LightGBM (calendar + history, no weather) | 0.462 | 0.644 |
| LightGBM (calendar + history + weather) | 0.462 | 0.642 |

The notebook reports the weather-enabled model at 0.462 kWh MAE (rounded), compared with 0.481 kWh for the 7-day same-hour baseline — approximately **4% lower MAE**. The weather features made very little difference to aggregate test MAE.

### Load-shifting simulation

For the 50% flexibility scenario shifting laundry and heater/AC loads:
- Average energy shifted: **1.035 kWh/day**
- Simulated daily bill savings: **2.224%**
- Simulated peak-window energy reduction: **15.026%**

The simulation used peak hours 18:00–21:00 at €0.25/kWh and selected lower-price hours at €0.15/kWh. These are modelled results under the notebook's assumptions, not measured household savings.

## Workflow

1. Download the UCI Individual Household Electric Power Consumption dataset.
2. Parse timestamps, clean missing readings and aggregate minute-level measurements to hourly energy.
3. Join hourly historical weather observations from Open-Meteo for coordinates 48.78, 2.29.
4. Explore consumption patterns by hour, weekday, month and appliance sub-meter.
5. Engineer calendar, lag, rolling and weather features.
6. Train and evaluate LightGBM with chronological splits and compare against historical baselines.
7. Simulate shifting flexible loads from peak to lower-tariff hours.

## Dataset

The notebook downloads the source dataset directly from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Weather data is retrieved from the Open-Meteo Historical Weather API:

https://open-meteo.com/en/docs/historical-weather-api

Raw data files are intentionally not included in this repository. The notebook retrieves them when executed.

## Running the notebook

This notebook was originally developed in Google Colab. To run locally:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook Household_Energy_Forecasting.ipynb
```

The notebook includes Google Colab Drive mounting and save paths. Those cells are optional and should be skipped or adapted when running outside Colab. Internet access is required for the UCI and Open-Meteo downloads.

## Important interpretation notes

- The train/validation/test split is chronological; the final test period begins on 2010-08-26.
- Test scoring excludes hours flagged as imputed. The notebook also reports a stricter score for real readings whose preceding 168-hour history contains no imputed values.
- Historical weather observations were used. This is not a deployment-ready future-weather forecasting pipeline.
- Load shifting is a simplified tariff simulation with assumed flexibility and a 1 kWh hourly transfer cap. Savings should not be interpreted as real-world guaranteed savings.
- Reported metrics are the saved outputs in the supplied notebook; this repository does not claim an independent rerun.

## Repository structure

```text
household-energy-forecasting-lightgbm/
├── Household_Energy_Forecasting.ipynb
├── README.md
├── requirements.txt
├── docs/
│   ├── methodology.md
│   └── results.md
└── results/
    └── figures/
```

## Resume-ready summary

**Household Energy Consumption Forecasting Using LightGBM** | Python, LightGBM, Pandas, Scikit-learn
- Processed **2.07M minute-level readings** into **34,589 hourly records**, handling missing values and flagging imputed periods.
- Engineered calendar, lag, rolling-window and weather features to forecast hourly household electricity demand using LightGBM.
- Reduced test MAE from **0.481 to 0.462 kWh (~4%)** and simulated **2.224% daily bill savings** through flexible-load shifting.
