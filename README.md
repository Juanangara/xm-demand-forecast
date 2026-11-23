# xm-demand-forecast

[![ci](https://github.com/Juanangara/xm-demand-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/Juanangara/xm-demand-forecast/actions/workflows/ci.yml)

Day-ahead hourly forecast of electricity demand in Colombia, built as an end-to-end ML exercise: real data from the national grid operator (XM), honest time-series validation, comparison against strong baselines with significance tests, experiment tracking, a REST API in a container, drift monitoring, and CI.

At 00:00 of day D, using only what happened before, the model forecasts the 24 hours of D. On 2023-2026 data it reaches **1.64 % MAPE**, against 3.37 % for "same hour last week" and 2.25 % for a classical MSTL model.

This repo is also a measurement log: the [development log](docs/DEVLOG.md) records what worked, what did not, and which claims were withdrawn.

## Features
- Data pipeline over the XM API (`pydataxm`): 26 years of hourly demand (2000-2026), cached month by month as Parquet, with a data-quality report and an anomaly audit
- Leak-proof setting: every feature uses values at least 24 h old, and a test proves it by rewriting everything from day D onwards
- Walk-forward validation: 46 monthly folds, retraining at the start of each month, scored by day type (ordinary / holiday)
- Colombian public-holiday calendar as features (holidays are 9 % of the hours but 28 % of the baseline's error)
- Comparison against a seasonal-naive baseline and a classical MSTL model, with Diebold-Mariano tests and block-bootstrap confidence intervals
- Training window chosen on a validation split that precedes the test period
- MLflow experiment tracking and model registry
- FastAPI service in Docker, with input validation and explicit errors
- Drift monitoring (level, daily shape, model error) calibrated on clean data and tested by injecting failures; Evidently report
- CI with GitHub Actions: lint, tests on synthetic data, Docker build

## How it works
```mermaid
flowchart LR
    XM[XM API] -->|download.py, monthly cache| RAW[(Parquet)]
    RAW --> AUDIT[Quality report + anomaly audit]
    AUDIT --> FEAT[Features: lags >= 24 h, holidays, calendar]
    FEAT --> WF[Walk-forward evaluation]
    WF --> CMP[Baselines, MSTL, Diebold-Mariano]
    WF -.-> MLF[(MLflow)]
    FEAT --> FINAL[train_final.py]
    FINAL --> MODEL[models/champion]
    MODEL --> API[FastAPI in Docker]
    RAW --> API
    RAW --> MON[Drift monitors]
```

1. `scripts/download.py` pulls `DemaReal` (real demand of the national system, kWh per hour) month by month and reports gaps, duplicates and zeros per year.
2. Reviewed anomalies are masked **only for training**; the raw series is never edited and the test period is never masked.
3. The model predicts `y / lag_168` (the change relative to the same hour last week) from scale-free features, so it does not have to extrapolate the 2x growth in demand since 2000.
4. Each model is scored on the same hours and compared with paired tests on daily errors.

## Results
Test period: 2023-01 to 2026-10 (46 monthly folds, 1,400 days, 33,600 hours), same hours for every model.

| Model | MAPE | Ordinary days | Holiday-affected days |
|---|---|---|---|
| Naive, same hour last week | 3.37 % | 2.63 % | 10.91 % |
| MSTL (classical, no calendar) | 2.25 % | 1.84 % | 6.38 % |
| **LightGBM, last 5 years** | **1.64 %** | **1.58 %** | **2.20 %** |

LightGBM beats the naive baseline in 46 of 46 months. Against MSTL the difference is 0.61 MAPE points (95 % block-bootstrap CI -0.77 to -0.47) and, on ordinary days alone, 0.26 points (CI -0.38 to -0.16), so the gain is not only holiday handling.

![Error by kind of day, by month, by day type, and the paired differences with their intervals](docs/results.png)

What the figure shows:
- **Holidays are where the models differ most.** On holiday-affected days the error is 10.9 % for the naive baseline, 6.4 % for MSTL and 2.2 % for LightGBM; on ordinary days the three are much closer (2.6 %, 1.8 %, 1.6 %).
- **Not a clean sweep.** LightGBM is below MSTL in 41 of 46 months, not all of them; the worst month is 2024-09 (2.47 % against 1.94 %). On Saturdays the two tie at 1.6 %, to one decimal.
- **The intervals are far from zero** for every comparison in the last panel, including ordinary days against MSTL (0.26 points, the smallest).

![Two weeks around Holy Week 2025: actual demand, the naive forecast and the LightGBM forecast, hour by hour](docs/holy_week.png)

A concrete case: Holy Thursday and Good Friday 2025 (17-18 April). The naive baseline copies the previous week and misses by 12.9 % on those two holidays; LightGBM misses by 3.9 %, and the figure shows it still overshoots the holidays themselves. One week later the naive baseline copies the holidays onto two ordinary days and misses by 13.9 %, while LightGBM, which knows about the holidays, misses by 1.6 %. Over the whole fortnight: 5.9 % against 1.7 %. It is one example chosen because it is clear, not a typical week.

What these numbers do **not** say:
- The test period was not untouched: the holiday features were designed after looking at the baseline's errors on it, so 1.64 % is probably a little optimistic.
- Neither MSTL nor LightGBM was tuned.
- Differences between training windows (5 years vs. all history) are real but tiny (about 0.06 points).

Details, tests and caveats: [docs/DEVLOG.md](docs/DEVLOG.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Quick start
Requirements: Python 3.12 and Docker. **The data is not in this repo**: it is rebuilt from the XM API (see [Data](#data)).

```
# 1. Environment
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Linux/Mac: .venv/bin/pip

# 2. Data (about 320 monthly requests; cached, so re-runs are fast)
python scripts/download.py

# 3. Baselines, model, comparison
python scripts/baselines.py
python scripts/train_lgbm.py
python scripts/stat_baseline.py        # slow: 1,400 daily refits (about 1 s each on one core)
python scripts/select_window.py
python scripts/compare.py
python scripts/make_figures.py         # redraws docs/results.png and docs/holy_week.png
python scripts/make_drift_figure.py    # docs/drift.png; about 2 minutes (trains two models)

# 4. Final model for the API (writes models/champion/)
python scripts/train_final.py

# 5. Serve
docker compose up --build
curl "http://localhost:8000/forecast?date=2026-10-12"   # any date inside your history, or the day after it ends
```
Experiments are in MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Tests and lint (no data needed):
```
pip install -r requirements-test.txt
ruff check src tests scripts && pytest -q
```

## API
| Method | Path | Description |
|---|---|---|
| GET | `/forecast?date=YYYY-MM-DD` | 24 hourly forecasts in kWh, day type, holiday flag, model version; the day's MAPE when real values exist |
| GET | `/health` | Status and the span of the loaded history |
| GET | `/model` | Model metadata: training end, features, code commit, reference metrics |

A day whose previous hour is not in the history returns `422` with a message saying what is missing. For days inside the training window the response carries `in_sample: true`, because the `mape_pct` it shows is then optimistic; the honest accuracy figure is the walk-forward result above.

## Drift monitoring
`scripts/simulate_drift.py` deploys a model frozen in 2023 and injects three failures (demand +10 %, a change in the evening profile, and a kWh-to-MWh pipeline bug). Main findings:
- No single monitor catches everything: the error-based monitor misses the profile change, because the model adapts through its lag features; only the shape monitor sees it.
- Retraining when the error alarm fires gained at most 0.05 MAPE points, and cannot fix a broken input. The frozen model scores 1.65 % against 1.64 % for the monthly-retrained one, so the policy is "an alarm opens an investigation", not "an alarm retrains".

![Each monitor's signal divided by its alarm threshold, with no failure and with three injected failures](docs/drift.png)

Each line is a monitor's signal divided by its own threshold (calibrated on 2021-2022, never on this period), so 1 is the alarm line; the failure starts on 3 June 2024 in the three panels where it is marked. In the figure:
- **Demand +10 %:** the level monitor crosses the line only briefly, and the error monitor barely touches it. A smaller shift would not have been caught.
- **Evening profile change:** only the shape monitor (green) alarms; the error monitor stays under the line.
- **Unit bug:** the level and error monitors go orders of magnitude above the threshold from the first days.
- **They fall back below the line after about two weeks** (both windows then hold the new regime), and in the clean panel the signals stay under it. Over the whole 1,400-day clean run, though, the monitors still raised 10 (level), 10 (shape) and 16 (error) alarm days: they are not free of false alarms (DEVLOG, Phase 10).

`scripts/drift_report.py` writes an Evidently report for the profile-change case.

## Repository layout
```
src/xmforecast/   data, quality, audit, calendar, features, evaluate, forecast,
                  stats, monitor, serving, api
scripts/          download, baselines, train_lgbm, stat_baseline, select_window,
                  compare, train_final, make_figures, make_drift_figure, error_analysis, simulate_drift,
                  drift_report
tests/            synthetic-data tests, including the leak and API tests
docs/             DEVLOG.md (measurements and decisions), MODEL_CARD.md, figures
```

## Data
Demand comes from [XM](https://www.xm.com.co) through the `pydataxm` package. The data and the trained model are **not** included in the repository: they are rebuilt with the scripts above. I did not review XM's terms of use for redistribution, which is why nothing derived from the data is shipped.

## Limitations
- It is not a live service: the history is a Parquet file that ends at the last downloaded complete month, and there is no daily refresh job.
- The API has no authentication, rate limiting or load tests.
- Drift monitoring is an offline simulation with three synthetic failures; it is not wired to the API.
- Only the national system total is modelled, and the Colombian holiday calendar comes from the `holidays` package, checked against my own knowledge but not against the official decree.
- Weather was deliberately left out: for a day-ahead forecast only *forecast* weather would exist, and using observed weather would leak information.
- The CI workflow has been verified in a clean local checkout; read the badge for its status on GitHub.
