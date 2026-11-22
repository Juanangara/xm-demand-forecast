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
