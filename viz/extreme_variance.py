"""
Diagnostic: why are there more cold extreme days than warm?

Four-panel figure:
  1. Warm vs cold extreme day counts per city
  2. Std(MAX) vs Std(MIN) per city  — variance is similar, so variance alone
     does not explain the asymmetry
  3. Skewness of MAX vs MIN per city — MIN is more negatively skewed everywhere,
     meaning the cold tail is fatter → more days breach the -2σ fence
  4. Full temperature distributions (violin) for a representative prairie city
     (Winnipeg) with threshold lines, showing the asymmetric tails
"""
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

DB_PATH  = Path("data/processed/climateflow.db")
OUT_PATH = Path("outputs/plots/extreme_variance.png")
TRAIN_END = 2019

WARM_COLOR = "#e05c2b"
COLD_COLOR = "#3a7abf"
REP_CITY   = "Winnipeg"   # most extreme cold/warm asymmetry


def load():
    conn = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT * FROM climate_features",  conn)
    thr = pd.read_sql("SELECT * FROM city_thresholds",   conn)
    conn.close()
    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])
    df = df.merge(thr, on="city")
    df["warm_extreme"] = df["MAX_TEMPERATURE"] > df["upper_fence"]
    df["cold_extreme"] = df["MIN_TEMPERATURE"] < df["lower_fence"]
    return df


def panel_counts(ax, df):
    """Stacked bar: warm vs cold extreme days per city."""
    counts = df.groupby("city").agg(warm=("warm_extreme", "sum"), cold=("cold_extreme", "sum"))
    cities = counts.index.tolist()
    x = np.arange(len(cities))
    w = 0.35
    ax.bar(x - w/2, counts["warm"], w, color=WARM_COLOR, label="Warm extremes\n(MAX > +2σ)")
    ax.bar(x + w/2, counts["cold"], w, color=COLD_COLOR, label="Cold extremes\n(MIN < −2σ)")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Days (2004–2023)")
    ax.set_title("Extreme day counts by type", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def panel_variance(ax, df):
    """Grouped bar: std(MAX) vs std(MIN) per city (training period)."""
    train = df[df["LOCAL_YEAR"] <= TRAIN_END]
    stats = train.groupby("city").agg(
        max_std=("MAX_TEMPERATURE", "std"),
        min_std=("MIN_TEMPERATURE", "std"),
    ).round(2)
    cities = stats.index.tolist()
    x = np.arange(len(cities))
    w = 0.35
    ax.bar(x - w/2, stats["max_std"], w, color=WARM_COLOR, alpha=0.85, label="Std(MAX_TEMP)")
    ax.bar(x + w/2, stats["min_std"], w, color=COLD_COLOR, alpha=0.85, label="Std(MIN_TEMP)")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Std dev (°C)")
    ax.set_title("Temperature variance — MAX vs MIN\n(training period 2004–2019)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    # annotation: max_std ≥ min_std everywhere
    ax.annotate(
        "MAX std ≥ MIN std\nin most cities →\nvariance alone doesn't\nexplain the cold imbalance",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc"),
    )


def panel_skewness(ax, df):
    """Grouped bar: skewness of MAX vs MIN per city."""
    train = df[df["LOCAL_YEAR"] <= TRAIN_END]
    skew = train.groupby("city").agg(
        max_skew=("MAX_TEMPERATURE", lambda s: s.dropna().skew()),
        min_skew=("MIN_TEMPERATURE", lambda s: s.dropna().skew()),
    ).round(3)
    cities = skew.index.tolist()
    x = np.arange(len(cities))
    w = 0.35
    ax.bar(x - w/2, skew["max_skew"], w, color=WARM_COLOR, alpha=0.85, label="Skew(MAX_TEMP)")
    ax.bar(x + w/2, skew["min_skew"], w, color=COLD_COLOR, alpha=0.85, label="Skew(MIN_TEMP)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Skewness")
    ax.set_title("Distribution skewness — MAX vs MIN\n(training period 2004–2019)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(
        "MIN more negatively\nskewed everywhere\n→ fatter cold tail →\nmore −2σ breaches",
        xy=(0.98, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc"),
    )


def panel_violin(ax, df, city=REP_CITY):
    """Violin plot of MAX and MIN distributions for the representative city."""
    train = df[(df["LOCAL_YEAR"] <= TRAIN_END) & (df["city"] == city)]
    upper = train["upper_fence"].iloc[0]
    lower = train["lower_fence"].iloc[0]

    max_vals = train["MAX_TEMPERATURE"].dropna().values
    min_vals = train["MIN_TEMPERATURE"].dropna().values

    parts = ax.violinplot(
        [max_vals, min_vals],
        positions=[1, 2],
        showmedians=True,
        showextrema=True,
    )
    for pc, color in zip(parts["bodies"], [WARM_COLOR, COLD_COLOR]):
        pc.set_facecolor(color)
        pc.set_alpha(0.45)
    for key in ("cmedians", "cmaxes", "cmins", "cbars"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.8)

    # threshold lines
    ax.axhline(upper, color=WARM_COLOR, linewidth=1.5, linestyle="--",
               label=f"+2σ warm fence = {upper:.1f}°C")
    ax.axhline(lower, color=COLD_COLOR, linewidth=1.5, linestyle="--",
               label=f"−2σ cold fence = {lower:.1f}°C")

    # shade extreme zones
    ax.axhspan(upper, max_vals.max() + 2, alpha=0.08, color=WARM_COLOR)
    ax.axhspan(min_vals.min() - 2, lower,  alpha=0.08, color=COLD_COLOR)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MAX_TEMP", "MIN_TEMP"], fontsize=9)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"{city}: temperature distributions\nwith ±2σ extreme thresholds", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # annotate % beyond each fence
    warm_pct = (max_vals > upper).mean() * 100
    cold_pct = (min_vals < lower).mean() * 100
    ax.text(1, upper + 1.5, f"{warm_pct:.1f}% above\nwarm fence",
            ha="center", va="bottom", fontsize=7.5, color=WARM_COLOR)
    ax.text(2, lower - 1.5, f"{cold_pct:.1f}% below\ncold fence",
            ha="center", va="top", fontsize=7.5, color=COLD_COLOR)


def run():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Why are there more cold extreme days than warm?\n"
        "Left-skewed temperature distributions create fatter cold tails",
        fontsize=13, fontweight="bold", y=1.01,
    )

    panel_counts(axes[0, 0], df)
    panel_variance(axes[0, 1], df)
    panel_skewness(axes[1, 0], df)
    panel_violin(axes[1, 1], df)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    run()
