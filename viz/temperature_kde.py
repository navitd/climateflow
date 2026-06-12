"""
KDE distributions of MAX and MIN temperature per city.

3×3 grid — one subplot per city — showing the actual probability density of
MAX_TEMPERATURE (orange) and MIN_TEMPERATURE (blue) with the ±2σ extreme
thresholds as vertical dashed lines and the tail zones shaded.  The asymmetric
left tail of MIN_TEMPERATURE (skewness < 0) is directly visible in the curve
shapes: more probability mass sits beyond the cold fence than beyond the warm
fence, explaining why cold extreme days outnumber warm ones.
"""
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

DB_PATH  = Path("data/processed/climateflow.db")
OUT_PATH = Path("outputs/plots/temperature_kde.png")
TRAIN_END = 2019

WARM_COLOR = "#e05c2b"
COLD_COLOR = "#3a7abf"


def load():
    conn = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT * FROM climate_features", conn)
    thr = pd.read_sql("SELECT * FROM city_thresholds",  conn)
    conn.close()
    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])
    return df.merge(thr, on="city")


def run():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load()
    train = df[df["LOCAL_YEAR"] <= TRAIN_END].copy()
    cities = sorted(train["city"].unique())

    # shared x-axis range across all cities
    x_min = train[["MAX_TEMPERATURE", "MIN_TEMPERATURE"]].min().min() - 3
    x_max = train[["MAX_TEMPERATURE", "MIN_TEMPERATURE"]].max().max() + 3
    x = np.linspace(x_min, x_max, 500)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    fig.suptitle(
        "Temperature distributions per city — MAX vs MIN (training 2004–2019)\n"
        "Shaded tails show days beyond the ±2σ extreme fence; "
        "cold tails (blue) are fatter than warm tails (orange)",
        fontsize=11, fontweight="bold",
    )

    for ax, city in zip(axes.flat, cities):
        city_df = train[train["city"] == city]
        upper = city_df["upper_fence"].iloc[0]
        lower = city_df["lower_fence"].iloc[0]

        # compute warm and cold % for annotation
        max_vals = city_df["MAX_TEMPERATURE"].dropna().values
        min_vals = city_df["MIN_TEMPERATURE"].dropna().values
        warm_pct = (max_vals > upper).mean() * 100
        cold_pct = (min_vals < lower).mean() * 100

        for vals, color, fence, shade_right in [
            (max_vals, WARM_COLOR, upper, True),
            (min_vals, COLD_COLOR,  lower, False),
        ]:
            kde = gaussian_kde(vals, bw_method=0.18)
            density = kde(x)

            ax.plot(x, density, color=color, linewidth=1.4)
            ax.fill_between(x, density, alpha=0.15, color=color)

            # shade the extreme tail
            tail_mask = (x >= fence) if shade_right else (x <= fence)
            ax.fill_between(x[tail_mask], density[tail_mask], alpha=0.55, color=color)
            ax.axvline(fence, color=color, linewidth=1.1, linestyle="--", alpha=0.85)

        # city title + tail % annotation
        ax.set_title(city, fontsize=9, fontweight="bold", pad=3)
        ax.annotate(
            f"warm: {warm_pct:.1f}%\ncold:  {cold_pct:.1f}%",
            xy=(0.03, 0.95), xycoords="axes fraction",
            va="top", ha="left", fontsize=7,
            color="#444444",
        )
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)

    # shared x-label
    fig.text(0.5, 0.02, "Temperature (°C)", ha="center", fontsize=10)

    # legend
    legend_handles = [
        mpatches.Patch(color=WARM_COLOR, alpha=0.7, label="MAX_TEMP  (warm fence = mean + 2σ)"),
        mpatches.Patch(color=COLD_COLOR,  alpha=0.7, label="MIN_TEMP  (cold fence = mean − 2σ)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.055))

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    run()
