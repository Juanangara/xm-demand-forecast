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
