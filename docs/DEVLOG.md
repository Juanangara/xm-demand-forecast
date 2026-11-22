# Development log

What was tried, what was measured, and what did not work. Newest entries last.

## Phase 1: data source (2026-11-22)

**Source:** XM (Colombian power-system operator) through `pydataxm` 0.3.19, metric `DemaReal`, entity `Sistema` (hourly real demand of the national system). No API key. Max 31 days per request, so history is fetched month by month and cached as Parquet.

**Result of the first full pull (2000-01-01 to 2026-10-31, 235,224 hourly rows):**
- 0 missing hours, 0 duplicates, 0 zeros in every year. The series is complete.
- Strong trend: yearly median goes from ~4.8 GWh/h (2000) to ~9.6 GWh/h (2025). A model trained on old years sees a different level, so the training window and drift monitoring matter.
- The lowest values are short isolated episodes, not scattered noise: 2007-04-26 10:00-12:00 (min 0.41 GWh), 2002-03-16/17 and 2000-01-17. **Cause not investigated yet**; they are kept in the data until checked, not silently removed.

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
