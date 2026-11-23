"""Draw docs/drift.png: the three monitors against their alarm thresholds, for each scenario.

Recomputes the simulation of scripts/simulate_drift.py (about two minutes: it trains two
models) and plots every signal divided by its own calibrated threshold, so 1.0 is the alarm
line for all of them. Log scale, because the unit bug is orders of magnitude off the others.
Usage: python scripts/make_drift_figure.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from make_figures import GRID, INK, INK_2, MUTED, SURFACE, style  # noqa: E402
from simulate_drift import CAL, DEPLOY, T, daily_errors, scenarios, train_frozen  # noqa: E402

from xmforecast.calendar import holiday_dates  # noqa: E402
from xmforecast.monitor import calibrate_threshold, level_signal, performance_signal, shape_signal  # noqa: E402

DOCS = HERE.parent / "docs"
WINDOW = ("2024-04-15", "2024-08-15")
GREEN, BLUE, ORANGE = "#1baf7a", "#2a78d6", "#eb6834"
MONITORS = {
    "level": ("Level", BLUE),
    "shape": ("Daily shape", GREEN),
    "performance": ("Model error", ORANGE),
}
TITLES = {
    "clean": "No failure injected",
    "level_shift": "Demand +10 %",
    "evening_shape": "Hours 18-21 +12 %",
    "unit_bug": "Values divided by 1000 (unit bug)",
}


def main() -> None:
    s = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    hol = holiday_dates(2019, 2027)
    cal = slice(*CAL)
    w = s.loc["2020-12-01":CAL[1]]
    model_a = train_frozen(s, "2020-12-31")
    thr = {
        "level": calibrate_threshold(level_signal(w, ignore_days=hol).loc[cal], q=0.995),
        "shape": calibrate_threshold(shape_signal(w).loc[cal], q=0.995),
        "performance": calibrate_threshold(
            performance_signal(daily_errors(model_a, w, "2020-12-01", CAL[1])).loc[cal], q=0.99, two_sided=False),
    }
    model_b = train_frozen(s, "2022-12-31")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE)
    for ax, (name, alt) in zip(axes.flat, scenarios(s).items()):
        a0 = alt.loc["2022-12-01":]
        sig = {
            "level": level_signal(a0, ignore_days=hol).loc[DEPLOY:],
            "shape": shape_signal(a0).loc[DEPLOY:],
            "performance": performance_signal(daily_errors(model_b, alt, "2022-12-01")).loc[DEPLOY:],
        }
        style(ax, TITLES[name], "signal / threshold" if name in ("clean", "evening_shape") else None)
        ax.set_yscale("log")
        for key, (label, color) in MONITORS.items():
            ratio = (sig[key].abs() / thr[key]).loc[WINDOW[0]:WINDOW[1]]
            ax.plot(ratio.index, np.clip(ratio.values, 0.03, None), color=color, linewidth=1.8, label=label)
        ax.axhline(1, color=INK, linewidth=1)
        ax.axvline(T, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        if name != "clean":
            ax.text(T - pd.Timedelta(days=1), 1500, "failure starts", fontsize=8, color=INK_2, ha="right", va="center")
        if name == "unit_bug":
            ax.text(T + pd.Timedelta(days=4), 700, "error is\noff the scale", fontsize=8, color=ORANGE,
                    ha="left", va="center")
        ax.set_ylim(0.03, 3000)
        ax.set_xlim(pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1]))
        ax.set_xticks(pd.to_datetime(["2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"]),
                      ["May", "Jun", "Jul", "Aug"])
        ax.set_yticks([0.1, 1, 10, 100, 1000], ["0.1", "1 (alarm)", "10", "100", "1000"])
        ax.minorticks_off()
        ax.spines["left"].set_color(GRID)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=10, labelcolor=INK,
               bbox_to_anchor=(0.98, 0.99))
    fig.suptitle("xm-demand-forecast: drift monitors", x=0.02, y=0.99, ha="left", va="top", fontsize=13,
                 color=INK)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.07, hspace=0.32, wspace=0.18)
    fig.savefig(DOCS / "drift.png", dpi=150, facecolor=SURFACE)
    print("wrote docs/drift.png; thresholds:", {k: round(v, 4) for k, v in thr.items()})


if __name__ == "__main__":
    main()
