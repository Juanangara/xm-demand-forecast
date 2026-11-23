# Model card: xm-demand-day-ahead

## What it is
A LightGBM regressor that forecasts the hourly electricity demand of the Colombian national grid (XM metric `DemaReal`, entity `Sistema`, in kWh per hour) for the 24 hours of the next day.

## Intended use
- Demonstrating a day-ahead forecasting pipeline: validation, comparison, serving and monitoring.
- **Not** intended for operational decisions, market bidding or grid dispatch. It has not been validated for that, and the service has no authentication or availability guarantees.

## Setting
At 00:00 of day D the model sees only values before D 00:00 and forecasts all 24 hours of D. Every feature uses values at least 24 h old (a test checks it by rewriting day D and later).

## Training data
- 2000-01-01 to 2026-10-31, hourly, 235,224 rows, no missing hours or duplicates.
- The deployed model uses the last 5 years only (43,825 rows), because the daily load curve changed over time (the evening ramp shrank from about +21 % to +5 %). The window was chosen on a 2021-2022 validation split.
- Two reviewed disturbance episodes (7 hours: 2002-03-16/17 and 2007-04-26) are masked in training only. Their cause is not verified.
- The data is not redistributed; see the README.

## Features and target
Target: `y / lag_168` (relative change versus the same hour last week). Features: hour, weekday, month, day type (weekday / near a holiday / Saturday / Sunday / holiday), holiday flags for the days 24, 168 and 336 h earlier, the ratios of the 24, 48 and 336 h lags to the 168 h lag, and the ratio of the previous-week mean. Colombian public holidays come from the `holidays` package.

## Performance
Walk-forward, 46 monthly folds, 2023-01 to 2026-10 (retrained at the start of each month):

| Model | MAPE | Ordinary days | Holiday-affected days | Worst day |
|---|---|---|---|---|
| Naive (same hour last week) | 3.37 % | 2.63 % | 10.91 % | 26.64 % |
| MSTL, no calendar | 2.25 % | 1.84 % | 6.38 % | 16.21 % |
| LightGBM, 5 years | 1.64 % | 1.58 % | 2.20 % | 8.02 % |

By day type (LightGBM, 5 years): weekday 1.50 %, Saturday 1.65 %, Sunday 1.83 %, near a holiday 2.04 %, holiday 2.31 %.

Paired tests (Diebold-Mariano, Newey-West, 7 lags; 95 % block-bootstrap intervals) put LightGBM ahead of the naive baseline and of MSTL with intervals far from zero. See `docs/DEVLOG.md` (Phase 7).

## Limitations and known issues
- **Optimistic estimate.** The holiday features were designed after seeing the baseline's errors on the test period; the validation split fixes the window choice, not the feature design.
- **Few distinct holidays** in the test (about 125 holiday-affected days), so the holiday numbers are the least firm.
- **No tuning** of LightGBM hyperparameters or of the MSTL reference.
- **Frozen vs. retrained:** a model trained once in 2022 scores 1.65 % on the same period, so frequent retraining is not what drives the result.
- **Demand shocks** (the two masked episodes, and any future event of that kind) are not forecastable by this model.
- **Calendar source:** the holiday list was checked against my own knowledge, not the official decree.
- **The API's `mape_pct` for past days is in-sample** (flagged by `in_sample: true`); use the table above for accuracy.

## Monitoring and maintenance
Level, daily-shape and error monitors were calibrated on 2021-2022 and tested by failure injection (see the README and DEVLOG Phase 10). Retraining at an alarm did not pay off in the simulation; the suggested policy is to investigate the data first and retrain periodically (for example quarterly). Nothing is scheduled or automated.
