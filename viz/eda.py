"""
EDA plots — data quality and distribution overview.
Saves three PNGs to outputs/plots/ for the README.

  1. missingness_heatmap.png  — null pattern per column × city
  2. distributions.png        — violin plots of key measurements by city
  3. temperature_normals.png  — observed daily temp vs monthly normal (sample cities)
"""
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

DB_PATH = Path("data/processed/climateflow.db")
OUT_DIR = Path("outputs/plots")

CITY_ORDER = [
    "Vancouver", "Calgary", "Edmonton", "Saskatoon", "Winnipeg",
    "Toronto", "Ottawa", "Montreal", "Quebec", "Halifax",
]

PALETTE = sns.color_palette("tab10", n_colors=len(CITY_ORDER))
CITY_COLORS = dict(zip(CITY_ORDER, PALETTE))


def load() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM climate_daily_clean", conn)
    conn.close()
    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])
    return df


# ---------------------------------------------------------------------------
# Plot 1 — Missingness heatmap
# ---------------------------------------------------------------------------

def plot_missingness(df: pd.DataFrame) -> None:
    # Columns with any nulls, ordered by null rate descending
    null_pct = df.isnull().mean() * 100
    cols = null_pct[null_pct > 0].sort_values(ascending=False).index.tolist()

    # Build city × column null-rate matrix
    matrix = (
        df.groupby("city")[cols]
        .apply(lambda g: g.isnull().mean() * 100)
        .reindex(CITY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 7},
        cbar_kws={"label": "% missing", "shrink": 0.7},
    )
    ax.set_title("Missing data (%) by city and column", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=40, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    fig.tight_layout()

    out = OUT_DIR / "missingness_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Distribution violin plots
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame) -> None:
    measurements = {
        "MEAN_TEMPERATURE":    "Mean Temperature (°C)",
        "TOTAL_PRECIPITATION": "Total Precipitation (mm)",
        "SNOW_ON_GROUND":      "Snow on Ground (cm)",
        "SPEED_MAX_GUST":      "Max Gust Speed (km/h)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, measurements.items()):
        data = df[["city", col]].dropna()
        # Cap heavy-tailed precip/snow for readability
        if col in ("TOTAL_PRECIPITATION", "SNOW_ON_GROUND"):
            cap = data[col].quantile(0.99)
            data = data[data[col] <= cap]

        sns.violinplot(
            data=data,
            x="city", y=col,
            order=CITY_ORDER,
            hue="city",
            hue_order=CITY_ORDER,
            palette=PALETTE,
            inner="quartile",
            linewidth=0.8,
            legend=False,
            ax=ax,
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(6))

    fig.suptitle("Distribution of key weather variables by city  (2004–2023)",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    out = OUT_DIR / "distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Observed temperature vs monthly normal
# ---------------------------------------------------------------------------

def plot_temperature_normals(df: pd.DataFrame) -> None:
    sample_cities = ["Vancouver", "Toronto", "Winnipeg", "Calgary"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax, city in zip(axes, sample_cities):
        city_df = df[df["city"] == city].copy()

        # Daily observed — thin, low-alpha line
        ax.plot(
            city_df["LOCAL_DATE"],
            city_df["MEAN_TEMPERATURE"],
            color=CITY_COLORS[city],
            alpha=0.25,
            linewidth=0.4,
            label="Daily observed",
        )

        # Annual rolling mean — smoothed signal
        smoothed = (
            city_df.set_index("LOCAL_DATE")["MEAN_TEMPERATURE"]
            .rolling(30, center=True, min_periods=15)
            .mean()
        )
        ax.plot(smoothed.index, smoothed.values,
                color=CITY_COLORS[city], linewidth=1.6, label="30-day rolling mean")

        # Monthly normals — repeated across years as a step line
        norm = (
            city_df.groupby("LOCAL_MONTH")["NORMAL_MEAN_TEMP"]
            .first()
            .reindex(range(1, 13))
        )
        # Tile normals across all years for comparison
        years = sorted(city_df["LOCAL_DATE"].dt.year.unique())
        for yr in years:
            month_starts = pd.date_range(f"{yr}-01-01", periods=12, freq="MS")
            ax.step(month_starts, norm.values, where="post",
                    color="black", linewidth=0.6, alpha=0.35,
                    label="Monthly normal" if yr == years[0] else "")

        ax.set_title(city, fontsize=11)
        ax.set_ylabel("°C", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(
            plt.matplotlib.dates.YearLocator(4)
        )
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%Y")
        )

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Daily mean temperature vs 1981–2010 monthly normals",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    out = OUT_DIR / "temperature_normals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading cleaned data...")
    df = load()
    print(f"  {len(df):,} rows, {df.shape[1]} columns\n")

    print("Generating plots...")
    plot_missingness(df)
    plot_distributions(df)
    plot_temperature_normals(df)
    print("\nDone.")


if __name__ == "__main__":
    run()
