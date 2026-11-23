# Development log

What was tried, what was measured, and what did not work. Newest entries last.

## Phase 1: data source (2026-11-22)

**Source:** XM (Colombian power-system operator) through `pydataxm` 0.3.19, metric `DemaReal`, entity `Sistema` (hourly real demand of the national system). No API key. Max 31 days per request, so history is fetched month by month and cached as Parquet.

**Result of the first full pull (2000-01-01 to 2026-10-31, 235,224 hourly rows):**
- 0 missing hours, 0 duplicates, 0 zeros in every year. The series is complete.
- Strong trend: yearly median goes from ~4.8 GWh/h (2000) to ~9.6 GWh/h (2025). A model trained on old years sees a different level, so the training window and drift monitoring matter.
- The lowest values are short isolated episodes, not scattered noise: 2007-04-26 10:00-12:00 (min 0.41 GWh) and 2002-03-16/17. They are kept in the raw data. (An earlier version of this note also listed 2000-01-17; that was wrong, it is just the minimum of that year. See Phase 4.)

**Two mistakes caught while building the report:**
1. XM returned some months with text values, so a median crashed. Values are now coerced to numeric (non-numeric becomes NaN, so it shows up as a gap, not as a crash).
2. The first report flagged 2026 as 16.7 % missing. It was comparing against the full year while the data ends on 31 October. The report now only counts hours up to the last available timestamp.

**Not checked yet:** XM terms of use for redistributing data (the repo does not ship data; it is rebuilt with `scripts/download.py`).

## Phase 3: validation and baselines (2026-11-22)

**Setting:** day-ahead. At 00:00 of day D we know everything before D and forecast its 24 hours. Folds are the 46 calendar months from 2023-01 to 2026-10, each scored on its own. `seasonal_naive` refuses lags under 24 h, since those would use values from the day being forecast.

**Baselines (test 2023-01 to 2026-10, logged in MLflow, experiment `day-ahead-demand`):**

| Model | MAE | MAPE | Worst month (MAPE) |
|---|---|---|---|
| naive 24 h (same hour yesterday) | 458 MWh | 4.90 % | 5.73 % |
| naive 168 h (same hour last week) | 315 MWh | 3.37 % | 5.63 % |

Any model has to beat **3.37 %** to be worth keeping. Last week's value wins because demand depends strongly on the day of the week; yesterday is wrong every Monday and every weekend.

**Known weakness of both baselines, not yet measured:** public holidays. Both copy a normal day onto a holiday (and the next week onto the day after one). The worst months are the first thing to inspect before adding holiday features, so that the improvement can be attributed instead of assumed.

**Limits:** 46 folds are not independent (consecutive months share a trend), so the spread across folds describes variability, not a confidence interval.

## Phase 4: data audit and cleaning rules (2026-11-22)

Phases 1-2 only audited gaps, duplicates and zeros; nothing had been cleaned or reviewed. This phase checks what was missing.

**Checks and results (235,224 hours, 2000-2026):**
- **Frozen values (3 or more identical hours in a row):** none.
- **Hour-to-hour jumps above 25 %:** 655, but they are structural, not errors. They sit at 05:00 and 18:00 (the morning and evening ramps) and shrink over time: the median jump at 18:00 goes from +21 % (2000-04) to +5 % (2025-26). The shape of the daily curve changed, which supports limiting the training window instead of using all 26 years.
- **Same hour vs. the median of +-4 weeks (ratio outside 0.6-1.4):** only 16 hours. They are two disturbances and two New Year's Days.
- **Daily totals 20 % away from the same weekday in nearby weeks:** 53 days, mostly public holidays (1 January, Good Friday). Expected, handled in Phase 5.

**Reviewed events (the only ones treated as anomalous):**
| Event | Observation |
|---|---|
| 2002-03-16 23:00 to 03-17 01:00 | demand falls to 49-72 % of the previous week and recovers by 02:00 |
| 2007-04-26 10:00 to 13:00 | demand falls to 6 % of the previous week at 10:00 and recovers by 14:00 |

A sharp drop followed by a gradual recovery over a few hours looks like a real grid event, not a corrupted reading. **The cause was not verified**, so this is an interpretation, not a finding.

**Cleaning rules (`src/xmforecast/audit.py`):**
1. The raw series is never edited.
2. A detector flags candidates; a person reviews them. Public holidays are flagged too (the detector does not know the calendar), so a flag is not a verdict.
3. Only reviewed events (7 hours in total) are masked, and only for training. The test period (2023 onwards) contains none of them and is never masked, so reported errors are not flattered.

**Still open:** the cause of the two events; whether XM changed how it measures demand over the years (the trend from 4.8 to 9.6 GWh/h looks real but was not checked against a second source).

## Phase 5: where the baseline fails, and the holiday calendar (2026-11-22)

**Calendar:** Colombian public holidays come from the `holidays` package (0.109), including the rule that moves many of them to Monday. It was checked against 2024 holidays written down by hand (18 of 18 match) and the yearly counts for 2023 (18) and 2025 (17) agree with what I know. This is a check against my own knowledge, **not against the official decree**; a wrong date would hurt the holiday features, so it is worth a second look before publishing.

**Error of the naive 168 h baseline by day type (test 2023-01 to 2026-10, 35,000 hours; `scripts/error_analysis.py`):**

| Target day | Hours | MAPE | Share of hours | Share of total error |
|---|---|---|---|---|
| holiday | 1,632 | 10.80 % | 4.9 % | 14.1 % |
| near a holiday (weekday) | 1,752 | 3.67 % | 5.2 % | 5.6 % |
| weekday | 20,712 | 3.01 % | 61.6 % | 57.8 % |
| saturday | 4,776 | 2.91 % | 14.2 % | 12.1 % |
| sunday | 4,728 | 2.72 % | 14.1 % | 10.3 % |

- The baseline also fails the week **after** a holiday, because it copies the holiday: 1,368 hours where the target is a normal day but last week's copy was a holiday, MAPE **11.05 %**.
- Hours touched by a holiday, as target or as reference, are **8.9 % of the hours but 28.4 % of the total error**.
- On ordinary days with an ordinary reference the baseline already reaches **2.57 % MAPE** (against 3.37 % overall). By hour it is flat, between 2.2 % and 2.8 %, so there is no bad hour to fix.

**What this means for the model:** a good holiday treatment is the biggest and cheapest gain available, so Phase 6 must add holiday features and then report the improvement separately on ordinary days and on holiday-affected days. If the model only wins on holidays, that is the honest summary. The 2.57 % is what the baseline does on ordinary days, not a floor for a model.

**Limits:** only 4 years of test data, so about 70 holidays; the holiday MAPE is based on few distinct days (each one contributes 24 correlated hours).

## Phase 6: LightGBM, walk-forward (2026-11-22)

**Design:** the model predicts `y / lag_168` (the change relative to the same hour last week) from scale-free features, so it does not have to extrapolate the 2x growth in demand since 2000. Features (`src/xmforecast/features.py`): hour, weekday, month, day type, holiday flags for the reference days (24, 168 and 336 h earlier), ratios of the 24, 48 and 336 h lags to the 168 h lag, and the ratio of the previous-week mean. Every feature uses values at least 24 h old; `tests/test_features.py` proves it by rewriting everything from day D onwards and checking that D's features do not move. The model is retrained at the start of every test month (46 folds) on hours strictly before that month, with the two reviewed disturbances masked in training only.

**Results (test 2023-01 to 2026-10; MLflow runs `lgbm_window_*`):**

| Model | MAPE | Worst month | Ordinary hours | Holiday-affected hours |
|---|---|---|---|---|
| naive 168 h (baseline) | 3.37 % | 5.63 % | 2.63 % | 10.91 % |
| LightGBM, all history | 1.70 % | 2.57 % | 1.64 % | 2.35 % |
| LightGBM, last 5 years | **1.64 %** | 2.47 % | 1.58 % | 2.20 % |
| LightGBM, last 3 years | 1.70 % | 2.51 % | 1.63 % | 2.47 % |

"Holiday-affected" = the target is a holiday, or last week's copy was one. (Ordinary baseline error is 2.63 % here and 2.57 % in Phase 5 because the Phase 5 figure also excluded days next to a holiday.)

**What holds up:**
- LightGBM beats the baseline in **46 of 46 months** and in every year (2023: 3.24 -> 1.67, 2024: 3.55 -> 1.80, 2025: 3.39 -> 1.51, 2026: 3.27 -> 1.56 for the 5-year window).
- The gain is not only holidays: ordinary hours go from 2.63 % to 1.58 %, and holiday-affected hours from 10.91 % to 2.20 %, so the holiday features did what Phase 5 predicted.
- Remaining weak spots (5-year window): holidays 2.31 %, near-holiday days 2.04 %, Sundays 1.83 %, versus 1.50 % on ordinary weekdays.

**What does not hold up yet:**
- **The window choice is weak.** 5 years beats all history in 37 of 46 months, but by 0.06 points of MAPE overall, and I picked it by looking at the test period among three options. Treat it as "no clear difference"; it needs a proper selection split or a significance test (Phase 7) before claiming it.
- Feature importances (split counts) were saved only for the last run; they are not causal and say little about the other windows. `r24` (yesterday relative to last week) ranks first, which suggests the model mostly corrects the weekly baseline with the most recent level.
- 46 months are not independent, and there is no confidence interval yet.

## Phase 7: reference model, window selection and significance (2026-11-22)

**Window chosen without the test period.** Each window was trained on 2021-2022 only (`scripts/select_window.py`): all history 1.527 %, last 5 years **1.456 %**, last 3 years 1.482 % MAPE. The 5-year window wins on validation too, so the Phase 6 choice stands, now for a reason that does not depend on the test numbers.

**Classical reference model.** MSTL (daily + weekly seasonality) with an ETS trend, fitted for every test day only on the 6 weeks before it (`scripts/stat_baseline.py`, 1,400 daily refits). It knows nothing about holidays, on purpose: it is what a good classical model does without calendar knowledge.

**Test 2023-01 to 2026-10, same 33,600 hours for every model (`scripts/compare.py`):**

| Model | MAPE | Ordinary days (1,275) | Holiday-affected days (125) | Worst day |
|---|---|---|---|---|
| naive 168 h | 3.37 % | 2.63 % | 10.91 % | 26.64 % |
| MSTL | 2.25 % | 1.84 % | 6.38 % | 16.21 % |
| LightGBM, all history | 1.70 % | 1.64 % | 2.35 % | 9.83 % |
| **LightGBM, 5 years** | **1.64 %** | **1.58 %** | **2.20 %** | 8.02 % |
| LightGBM, 3 years | 1.70 % | 1.63 % | 2.47 % | 11.48 % |

**Significance (Diebold-Mariano on daily MAPE, Newey-West with 7 lags; 95 % moving-block bootstrap CI, blocks of 14 days). Difference = LightGBM 5 y minus the other, negative is better:**

| Versus | Days | Difference (MAPE points) | 95 % CI | p |
|---|---|---|---|---|
| naive 168 h | all | -1.73 | [-1.99, -1.51] | < 1e-50 |
| MSTL | all | -0.61 | [-0.77, -0.47] | 5e-18 |
| MSTL | ordinary only | -0.26 | [-0.38, -0.16] | 1e-7 |
| MSTL | holiday-affected | -4.17 | [-4.72, -3.72] | < 1e-60 |
| LightGBM, all history | all | -0.06 | [-0.09, -0.03] | 2e-5 |
| LightGBM, all history | holiday-affected | -0.14 | [-0.42, 0.01] | 0.26 |
| LightGBM, 3 years | all | -0.07 | [-0.10, -0.04] | 2e-5 |

**What this supports:**
- LightGBM beats the naive baseline and the MSTL model with intervals far from zero, **also on ordinary days** (0.26 points better than MSTL), so the gain is not only holiday handling. Most of MSTL's remaining error is holidays (6.38 %), as expected.
- The window effect is real but small: about 0.06 points, with an interval that excludes zero overall but includes it on holiday-affected days (only 125 days). Practically, "5 years" and "all history" are close.

**Limits I want on record:**
- **The test period was not untouched.** The holiday features were designed after Phase 5 looked at baseline errors in this same period (2023-2026). The validation split fixes the window choice, not the feature design, so the final numbers are probably a little optimistic. The 2021-2022 validation MAPE of the selected setup (1.46 %) versus 1.64 % on test shows no sign of a collapse, but it is not an independent holdout either.
- MSTL has a single setting (6-week window, no holiday regressors) and LightGBM uses default-like hyperparameters; neither was tuned. A stronger classical model would add holiday regressors.
- The "holiday-affected" comparison uses 125 scattered days, so the Diebold-Mariano assumptions are looser there; read those p-values as indicative.
- Fifteen paired tests were run without a multiple-comparison correction; all the headline results are far below any reasonable threshold, the 0.26 p-value is the only one that matters for a negative claim.

## Phase 9: model export, API and container (2026-11-22)

**Model.** `scripts/train_final.py` trains LightGBM on the last 5 years of all available history (to 2026-10-31, 43,825 rows, reviewed disturbances masked) and writes `models/champion/model.txt` + `meta.json` (version, training end, features, code commit, reference metrics). The same run is logged to MLflow and registered as `xm-demand-day-ahead` (v1). The API loads the exported folder, **not** the MLflow registry: the registry stores absolute Windows paths that would not exist inside a container, and a self-contained folder keeps the service independent of the tracking server. (The first registration attempt failed because MLflow 3 registers *logged models*, not plain artifact folders; fixed with `mlflow.lightgbm.log_model`. That failed attempt left one extra empty run in the local MLflow database.)

**API (`src/xmforecast/api.py`, FastAPI).** `GET /forecast?date=YYYY-MM-DD` returns the 24 hourly forecasts (kWh), the day type and whether it is a holiday, the model version, and, when the real values are in the history, the day's MAPE. `GET /health`, `GET /model`. A forecast for day D only reads history before D 00:00, as in the evaluation; a test rewrites day D and later and checks the forecast does not move. Invalid dates and missing parameters give 422, and so does a day whose previous hour is not in the history (for example 2 days after the history ends), with a message saying what is missing.

**Container.** `docker compose up` builds a `python:3.12-slim` image (non-root user, pinned dependencies, health check) and mounts `models/` and `data/` read-only, since neither is in git. Verified here: the container answered `/health` and `/forecast?date=2026-10-12` with the same numbers as the local run (MAPE 2.10 %, a holiday), and returned 422 for a date outside the history. 34 tests pass.

**Honest limits:**
- **`mape_pct` for past days is in-sample.** The final model was trained on those days, so it flatters the model; the response carries `in_sample: true` for every day inside the training window. The honest accuracy figure is the walk-forward result (1.64 %), not what the endpoint shows for past dates.
- **It is not a live service yet.** The history is a Parquet file that stops at the last downloaded complete month (2026-10-31); real day-ahead use needs a daily refresh of the data (not implemented) and gaps in the latest hours are rejected, not filled.
- There is no authentication, rate limiting or load test; it is a demonstration of the serving path, not a hardened service.
- The container was tested on this machine only; there is no CI yet (Phase 11).

## Phase 10: drift monitoring (2026-11-22)

**Question:** if the model runs for years without anyone watching, would we notice when something breaks, and is retraining the right response? Everything below is an offline simulation (`scripts/simulate_drift.py`), not a live service.

**Setup.** A model trained up to 2022-12-31 is deployed **frozen** from 2023-01-01. Three signals (`src/xmforecast/monitor.py`), each with a threshold calibrated only on 2021-2022 using a model trained up to 2020 (never on the deployment period):
- **level:** log of (mean of the last 7 days / mean of the 7 days before). Comparing with the previous week, not a fixed reference, avoids alarming on the slow growth the model already handles.
- **shape:** the largest hourly change in the average daily profile (hour share of the day), same weekly comparison.
- **performance:** 14-day rolling mean of the model's daily MAPE.

Three failures were injected from 2024-06-03: demand +10 % (`level_shift`), hours 18-21 +12 % (`evening_shape`), and values divided by 1000 (`unit_bug`, a kWh to MWh pipeline bug).

**Detection delay in days (None = no alarm within 60 days):**

| Monitor | level +10 % | evening shape | unit bug |
|---|---|---|---|
| level (plain) | None | None | 0 |
| level (calendar-aware) | 5 | None | 1 |
| shape (plain) | None | 2 | None |
| performance | 5 | None | 0 |

- **No single monitor catches everything.** The performance monitor never sees the evening-shape change: the model adapts because its features are ratios to last week, so the error does not move. Only the shape monitor catches it. The level monitor catches the unit bug on day 0 and the performance monitor too.
- **They go quiet after about two weeks** by construction (both windows end up inside the new regime), so a permanent shift raises a transient alarm, not a permanent one.

**False alarms on the clean series** (days / episodes in the 520 clean days before the injection; alarm days over the whole 1,400-day run):

| Monitor | before injection | whole run |
|---|---|---|
| performance | 0 / 0 | 16 |
| level (plain) | 10 / 3 | 20 |
| level (calendar-aware) | 3 / 1 | 10 |
| shape (plain) | 4 / 3 | 10 |
| shape (calendar-aware) | 6 / 4 | 20 |

**Calendar-aware variants (public holidays left out of the weekly means).** For the level monitor this helped: it caught the +10 % shift that the plain version missed by a hair (peak 0.075 against a 0.080 threshold, because two holiday Mondays that week lowered the signal) and cut the false alarms. For the shape monitor it made things **worse** (fewer usable days per window, noisier average), so the plain shape monitor stays. The plain level monitor's false alarms were in Semana Santa and the end of the year, which is what led to the calendar-aware version; that variant was therefore designed after looking at this same deployment period.

**Retraining is not the answer to an alarm.** Over the whole 2023-2026 test the frozen model scores MAPE 1.65 % against 1.64 % for the monthly-retrained one of Phase 6 (by year 1.66/1.80/1.55/1.60 against 1.67/1.80/1.51/1.56). Retraining at the first performance alarm, scored on the following 28 days out of sample:

| Scenario | Alarm date | Frozen | Retrained at the alarm |
|---|---|---|---|
| clean series | 2024-09-22 | 2.12 % | 2.08 % |
| level +10 % | 2024-06-08 | 1.86 % | 1.82 % |
| evening shape | 2024-09-23 | 2.04 % | 2.01 % |
| unit bug | 2024-06-03 | about 19,200 % | about 19,300 % |

Gains of 0.03-0.05 points are within the noise of 28 days; the unit bug is a broken input that no retraining can fix. The performance alarms of Sep-Oct 2024 and Feb 2026 were real stretches of harder days (frozen 2.10 % against 2.08 % retrained), not a stale model. **Policy that follows:** an alarm opens an investigation (check the data pipeline first), and a challenger is promoted only if it beats the champion on held-out days; a periodic retrain (for example quarterly) is enough. For that reason `should_retrain` was renamed `sustained_alarm` (it only reports that an alarm has lasted 3 days).

**Evidently.** `scripts/drift_report.py` builds an Evidently data-drift report (`data/drift_report.html`, not committed) for the evening-shape case, 28 days before against 28 days after. It flags 21 of 25 columns, which is misleading on its face: hour shares are compositional (if the evening rises, every other hour's share falls). The place of the change shows in the p-values (hours 18-21: 1e-16 to 1e-14; the daily mean is untouched, p = 0.94). Evidently is used for the human-readable report; the alarms come from the fast, tested checks above.

**Limits:**
- Three synthetic injections at one date, with a handful of false-alarm episodes: the delays and rates are indicative, not estimates with intervals. A different start date (for example a week without holidays) would change them.
- The synthetic failures are not real incidents; real drift is usually slower and messier.
- The thresholds and the calendar-aware level monitor were not validated on an independent period.
- Nothing is connected to the API or scheduled; this phase shows the logic and measures it.

## Phase 11: continuous integration (2026-11-22)

**What the workflow does** (`.github/workflows/ci.yml`, on every push to `main` and every pull request):
- `test`: Python 3.12 on Ubuntu, installs `requirements-test.txt` (the exact serving dependencies plus pytest, httpx and ruff), runs `ruff check src tests scripts` and `pytest`. No data download, MLflow, Evidently or statsforecast: every test uses synthetic data, so CI needs no secrets and no network beyond PyPI.
- `docker`: builds the API image, so a broken Dockerfile or a missing dependency is caught.

**How it was checked without GitHub.** It has not run on GitHub yet (the repo is local only). Instead the committed files were exported with `git archive` into an empty folder, a fresh virtual environment was built from `requirements-test.txt` alone, and the same steps were run: lint passes and 41 tests pass; the workflow YAML parses; the Docker image builds from that clean export (875 MB). That check found two problems that only exist outside my machine:
1. The API tests trained their toy model with `LGBMRegressor`, which needs scikit-learn. The serving image does not install it (it only needs the native `Booster`), so 5 tests failed in the clean environment while passing on mine, where scikit-learn was there by accident. The tests now use the native `lgb.train`.
2. `requirements.txt` did not list scikit-learn either, although `forecast.py` and `train_final.py` use `LGBMRegressor`; it had been installed only as a side effect of other packages. It is listed now.
The same class of mistake had already happened with FastAPI in Phase 9; the clean-checkout test is what catches it.

Linting also found 13 small issues (unused imports, long lines, several statements on one line, an unused variable); fixed. The simulation of Phase 10 was rerun after those edits and gives identical numbers.

**Limits:**
- **Not yet run on GitHub.** Differences between this check (Windows host, a local virtual environment) and the real runner (Ubuntu, `actions/checkout@v4`, `actions/setup-python@v5`) are possible; the first real run may need a fix. I did not verify the action versions against a live run.
- The `docker` job only builds the image; it does not start the API, because the model and data are not in git, so there is nothing to serve in CI.
- The image is 875 MB (pandas, pyarrow, LightGBM); it has not been slimmed.
- Lint rules are deliberately minimal (errors, pyflakes, warnings, import order). There is no coverage report, no type checking, and no deployment step (CD).

## Publication (2026-11-22)

The repository was published as `Juanangara/xm-demand-forecast` (public, no license). The first real GitHub Actions run passed on both jobs (`test`: lint and 41 tests on Ubuntu; `docker`: image build), with no fix needed after the clean-checkout check of Phase 11. This closes the first limit listed there ("not yet run on GitHub"); the other limits stand (the Docker job only builds the image, no coverage, no CD).

## README figures (2026-11-22)

`scripts/make_figures.py` redraws `docs/results.png` (error by kind of day, by month and by day type, plus the paired differences with their 95 % intervals) and `docs/holy_week.png` (a fortnight around Holy Week 2025) from the saved predictions; every number in them is recomputed, none is copied. Two facts that the figures made visible and that are now written in the README instead of being left implicit:
- LightGBM beats MSTL in **41 of 46 months**, not all of them. The worst is 2024-09 (2.47 % against 1.94 %), inside the stretch of harder days seen in Phase 10. On Saturdays the two tie at 1.6 %.
- On Holy Thursday and Good Friday 2025 LightGBM misses by 3.9 % (the naive baseline by 12.9 %): it still overshoots the holidays themselves. The week after, the naive baseline copies the holidays onto two ordinary days (13.9 % against 1.6 %). That fortnight was picked because it is clear, not because it is typical.
