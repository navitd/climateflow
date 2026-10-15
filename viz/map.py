"""
Step 5 — Visualize
Loads the feature matrix and saved model, generates predictions, and produces:
  1. outputs/maps/climateflow_map.html  — interactive Folium map
  2. outputs/plots/time_series.png      — predicted vs actual per city per year
  3. outputs/plots/city_f1.png          — F1 per city bar chart
"""
import sqlite3
from pathlib import Path

import folium
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

DB_PATH   = Path("data/processed/climateflow.db")
MODEL_PATH = Path("outputs/model.joblib")
OUT_DIR   = Path("outputs")
MAPS_DIR  = OUT_DIR / "maps"
PLOTS_DIR = OUT_DIR / "plots"

TEST_START = 2020
TEST_END   = 2023

CITY_ORDER = [
    "Vancouver", "Calgary", "Edmonton", "Saskatoon", "Winnipeg",
    "Toronto", "Ottawa", "Montreal", "Quebec",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load() -> tuple:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM climate_features", conn)
    conn.close()
    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])

    bundle  = joblib.load(MODEL_PATH)
    model   = bundle["model"]
    features = bundle["features"]
    winner  = bundle["winner"]
    threshold = bundle["threshold"]
    print(f"Loaded {len(df):,} rows  |  model: {winner}  |  {len(features)} features  "
          f"|  threshold {threshold}")
    return df, model, features, threshold


# ---------------------------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------------------------

def generate_predictions(df: pd.DataFrame, model, features: list,
                         threshold: float) -> pd.DataFrame:
    valid = df[df["is_extreme_day"].notna()].copy()
    X = valid[features]
    valid["y_prob"] = model.predict_proba(X)[:, 1]
    valid["y_pred"] = (valid["y_prob"] >= threshold).astype(int)
    print(f"Predictions generated for {len(valid):,} rows")
    return valid


# ---------------------------------------------------------------------------
# Helper: city-level summary
# ---------------------------------------------------------------------------

def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in CITY_ORDER:
        sub = df[df["city"] == city]
        if sub.empty:
            continue
        lat = sub["lat"].iloc[0]
        lon = sub["lon"].iloc[0]
        freq_actual = sub["is_extreme_day"].mean() * 100

        test = sub[(sub["LOCAL_YEAR"] >= TEST_START) & (sub["LOCAL_YEAR"] <= TEST_END)]
        n_actual_test    = int(test["is_extreme_day"].sum())
        n_predicted_test = int(test["y_pred"].sum()) if "y_pred" in test.columns else None

        rows.append({
            "city":              city,
            "lat":               lat,
            "lon":               lon,
            "freq_pct":          round(freq_actual, 2),
            "n_actual_test":     n_actual_test,
            "n_predicted_test":  n_predicted_test,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1 — Folium interactive map
# ---------------------------------------------------------------------------

def plot_folium_map(summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    summary = summary.merge(
        metrics[["city", "f1"]].rename(columns={"f1": "test_f1"}),
        on="city", how="left",
    )

    cmap   = plt.get_cmap("RdYlGn_r")
    f_min  = summary["freq_pct"].min()
    f_max  = summary["freq_pct"].max()

    def freq_to_color(freq):
        norm = (freq - f_min) / (f_max - f_min + 1e-9)
        r, g, b, _ = cmap(norm)
        return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    m = folium.Map(location=[56, -96], zoom_start=4, tiles="CartoDB positron")

    for _, row in summary.iterrows():
        color  = freq_to_color(row["freq_pct"])
        radius = 6 + row["freq_pct"] * 2.5

        popup_html = f"""
        <div style='font-family:sans-serif; font-size:13px; min-width:180px'>
          <b>{row['city']}</b><br>
          <hr style='margin:4px 0'>
          Extreme day frequency (2004–2023): <b>{row['freq_pct']:.1f}%</b><br>
          Test period actual extreme days: <b>{row['n_actual_test']}</b><br>
          Test period predicted extreme days: <b>{row['n_predicted_test']}</b><br>
          Model F1 (test set): <b>{row['test_f1']:.3f}</b>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['city']}  ({row['freq_pct']:.1f}% extreme)",
        ).add_to(m)

    # Colour scale legend
    legend_html = """
    <div style='position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:8px;
                border:1px solid #ccc; font-family:sans-serif; font-size:12px'>
      <b>Extreme day frequency</b><br>
      <span style='color:#1a9850'>&#9632;</span> Low &nbsp;&nbsp;
      <span style='color:#fee08b'>&#9632;</span> Medium &nbsp;&nbsp;
      <span style='color:#d73027'>&#9632;</span> High<br>
      <i style='font-size:11px'>Marker size also reflects frequency</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    path = MAPS_DIR / "climateflow_map.html"
    m.save(str(path))
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2 — Predicted vs actual per year per city
# ---------------------------------------------------------------------------

def plot_time_series(df: pd.DataFrame) -> None:
    test = df[(df["LOCAL_YEAR"] >= TEST_START) & (df["LOCAL_YEAR"] <= TEST_END)].copy()
    years = list(range(TEST_START, TEST_END + 1))
    cities = [c for c in CITY_ORDER if c in test["city"].values]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=False)
    axes = axes.flatten()

    for ax, city in zip(axes, cities):
        sub = test[test["city"] == city]
        actual    = sub.groupby("LOCAL_YEAR")["is_extreme_day"].sum().reindex(years, fill_value=0)
        predicted = sub.groupby("LOCAL_YEAR")["y_pred"].sum().reindex(years, fill_value=0)

        x = np.arange(len(years))
        w = 0.35
        ax.bar(x - w/2, actual.values,    w, label="Actual",    color="steelblue",  alpha=0.85)
        ax.bar(x + w/2, predicted.values, w, label="Predicted", color="darkorange", alpha=0.85)

        ax.set_title(city, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=8)
        ax.set_ylabel("Extreme days", fontsize=8)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Next-day forecast vs actual extreme days per year — test period 2020–2023",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    path = PLOTS_DIR / "time_series.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3 — Per-city F1
# ---------------------------------------------------------------------------

def plot_city_f1(metrics: pd.DataFrame) -> None:
    data = (
        metrics[["city", "f1"]]
        .dropna()
        .sort_values("f1", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(data["city"], data["f1"], color="steelblue", alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, data["f1"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 score", fontsize=10)
    ax.set_title("Next-day forecast F1 by city — test period 2020–2023", fontsize=12)
    ax.axvline(data["f1"].mean(), color="tomato", linestyle="--",
               linewidth=1.2, label=f"Mean F1 = {data['f1'].mean():.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = PLOTS_DIR / "city_f1.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 4 — Seasonal risk rose
# ---------------------------------------------------------------------------

def plot_seasonal_risk(df: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_PATH)
    thresholds = pd.read_sql("SELECT * FROM city_thresholds", conn)
    conn.close()

    df = df.merge(thresholds, on="city", how="left")
    df["hot_extreme"]  = (df["MAX_TEMPERATURE"] > df["upper_fence"]).astype(float)
    df["cold_extreme"] = (df["MIN_TEMPERATURE"] < df["lower_fence"]).astype(float)

    months       = np.arange(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    width  = 2 * np.pi / 12 * 0.85

    fig, axes = plt.subplots(3, 3, figsize=(14, 14),
                             subplot_kw=dict(projection="polar"))
    axes = axes.flatten()

    for ax, city in zip(axes, CITY_ORDER):
        sub = df[(df["city"] == city) & df["MAX_TEMPERATURE"].notna()]

        hot  = sub.groupby("LOCAL_MONTH")["hot_extreme"].mean().reindex(months, fill_value=0)
        cold = sub.groupby("LOCAL_MONTH")["cold_extreme"].mean().reindex(months, fill_value=0)

        ax.bar(angles, cold.values, width=width,
               color="steelblue", alpha=0.85, label="Cold extreme")
        ax.bar(angles, hot.values,  width=width, bottom=cold.values,
               color="tomato", alpha=0.85, label="Heat extreme")

        # Clockwise from top so January is at 12 o'clock
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(month_labels, fontsize=8)

        # Radial grid: one reference ring at 10%
        r_max = (hot + cold).max()
        ax.set_ylim(0, max(r_max * 1.25, 0.01))
        ax.set_yticklabels([])
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        ax.set_title(city, fontsize=11, fontweight="bold", pad=15)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="steelblue", alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color="tomato",    alpha=0.85),
    ]
    fig.legend(handles, ["Cold extreme", "Heat extreme"],
               loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Monthly extreme-day risk profile — 2004–2023\n"
        "Bar height = fraction of days classified as extreme",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    path = PLOTS_DIR / "seasonal_risk.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run() -> None:
    print("=== Step 5 — Visualize ===\n")
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df, model, features, threshold = load()
    df = generate_predictions(df, model, features, threshold)

    summary = city_summary(df)
    metrics = pd.read_csv(OUT_DIR / "metrics_by_city.csv")

    print("\nGenerating visualisations...")
    plot_folium_map(summary, metrics)
    plot_time_series(df)
    plot_city_f1(metrics)
    plot_seasonal_risk(df)

    print("\n=== Step 5 complete ===")


if __name__ == "__main__":
    run()
