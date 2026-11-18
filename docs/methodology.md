# Methodology

## Data source and aggregation
The notebook downloads the UCI Individual Household Electric Power Consumption dataset. It parses the Date and Time columns, creates a timestamp index, and converts the minute-level active-power readings into energy per minute. Hourly records are formed using resampling; hours with fewer than 30 valid minute readings are marked missing.

## Missing-data treatment
For the target and appliance sub-meter series, gaps up to three hours are linearly interpolated. Remaining gaps are filled from the same hour one week earlier (168-hour lag). An `imputed` indicator preserves which hourly rows were originally incomplete, and imputed observations are excluded from model scoring.

## Weather enrichment
Historical hourly weather is retrieved from Open-Meteo for latitude 48.78 and longitude 2.29. Variables include temperature, relative humidity, wind speed, cloud cover and shortwave radiation. Weather observations are aligned to the household series and short internal gaps are interpolated.

## Features and model
The notebook builds calendar features, cyclical hour/day features, lagged demand values (including 24, 48 and 168 hours), rolling statistics and same-hour historical averages. LightGBM regression is trained with chronological train/validation/test partitions. Hyperparameters are selected on the validation period.

## Load-shifting experiment
A separate simulation transfers configurable fractions of laundry and heater/AC energy out of 18:00–21:00 peak hours into selected lower-price hours. It uses a simplified two-level tariff and limits each receiving hour to 1 kWh of additional load.
