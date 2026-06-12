"""Build climateflow_analysis.ipynb programmatically via nbformat."""
import nbformat as nbf
from pathlib import Path

nb  = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "geoflow", "language": "python", "name": "geoflow"},
    "language_info": {"name": "python", "version": "3.11.0"},
}

md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

# ── Title ──────────────────────────────────────────────────────────────────

cells.append(md("""\
# ClimateFlow: Predicting Extreme Climate Days Across Canada

**An end-to-end data pipeline and ML project using publicly available Canadian weather records.**

Canada spans six climate zones — from Vancouver's oceanic mild winters to the continental
deep-freeze of the prairies. That diversity makes a simple question surprisingly hard to answer:
*what counts as an extreme weather day for a given city?*

ClimateFlow builds a reproducible pipeline that:
1. Fetches 20 years of daily climate observations and 30-year normals from the MSC GeoMet OGC API
2. Cleans and merges the data into SQLite
3. Engineers 25 geospatial and temporal features, including anomalies relative to 30-year baselines
4. Trains a city-specific XGBoost classifier to flag extreme temperature days
5. Visualises results on an interactive Folium map

**Key result:** F1 = 0.833 · Recall = 0.961 · ROC-AUC = 0.998 across 9 cities, 2020–2023 test period.
"""))

# ── Setup ──────────────────────────────────────────────────────────────────

cells.append(md("---\n## 0. Setup"))

cells.append(code("""\
import sqlite3, warnings, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Image, IFrame, display

warnings.filterwarnings("ignore")

# Locate project root regardless of where the kernel was launched from.
# Walks up from CWD until it finds the directory that contains 'pipeline/'.
PROJ = Path.cwd()
while not (PROJ / "pipeline").exists() and PROJ != PROJ.parent:
    PROJ = PROJ.parent

DB    = PROJ / "data/processed/climateflow.db"
PLOTS = PROJ / "outputs/plots"
MAPS  = PROJ / "outputs/maps"

print(f"Project root: {PROJ}")
print(f"Database    : {DB}  (exists: {DB.exists()})")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120})

def load(table):
    conn = sqlite3.connect(DB)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df
"""))

# ── SECTION 1: THE DATA ────────────────────────────────────────────────────

cells.append(md("""\
---
## 1. The Data

All data is fetched programmatically from Canada's
[MSC GeoMet OGC API](https://api.weather.gc.ca/collections) — no manual downloads required.

| Dataset | API endpoint | What it contains |
|---|---|---|
| Daily observations | `climate-daily` | MAX/MIN/MEAN temperature, precipitation, snow depth, humidity, wind gust |
| Climate normals | `climate-normals` | 30-year monthly averages (1981–2010) — the climatological baseline |
| Station metadata | `climate-stations` | Coordinates and elevation for 8,614 stations across Canada |

**Nine cities were selected** to span Canada's main climate regions:
Vancouver (oceanic), Calgary and Edmonton (semi-arid continental), Saskatoon and Winnipeg
(cold continental), Toronto and Ottawa (humid continental), Montréal and Québec City
(humid continental / maritime).

Halifax was excluded: its primary daily station had 100% null temperatures throughout the
2020–2023 test period, making it impossible to evaluate the model there.
"""))

cells.append(code("""\
df = load("climate_daily_clean")
df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])

print(f"Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range : {df['LOCAL_DATE'].min().date()} → {df['LOCAL_DATE'].max().date()}")
print(f"Cities     : {sorted(df['city'].unique())}")
"""))

cells.append(code("""\
coverage = (
    df.groupby(["city", "province"])
    .agg(
        from_year=("LOCAL_DATE", lambda s: s.min().year),
        to_year  =("LOCAL_DATE", lambda s: s.max().year),
        n_days   =("LOCAL_DATE", "count"),
        lat      =("lat",  "first"),
        lon      =("lon",  "first"),
    )
    .reset_index()
    .sort_values("lon")
    .rename(columns={"from_year": "from", "to_year": "to"})
)
coverage.style.hide(axis="index").format({"lat": "{:.2f}", "lon": "{:.2f}"})
"""))

# ── SECTION 2: EDA ─────────────────────────────────────────────────────────

cells.append(md("""\
---
## 2. Exploratory Analysis

### 2.1 Temperature range across cities

The box plots below show the full distribution of daily mean temperature per city.
The spread immediately shows why a fixed national threshold (e.g., flag any day where
MAX > 30°C or MIN < −20°C) would produce inconsistent results: a −30°C day in Vancouver
is a once-in-a-generation event; in Winnipeg, it happens most winters.
"""))

cells.append(code("""\
city_order = ["Vancouver", "Calgary", "Edmonton", "Saskatoon", "Winnipeg",
              "Toronto", "Ottawa", "Montreal", "Quebec"]

fig, ax = plt.subplots(figsize=(11, 4))
data = [df[df["city"] == c]["MEAN_TEMPERATURE"].dropna().values for c in city_order]

bp = ax.boxplot(
    data, patch_artist=True,
    medianprops =dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=0.8),
    capprops    =dict(linewidth=0.8),
    flierprops  =dict(marker=".", markersize=2, alpha=0.25),
)
palette = ["#3a7abf"] + ["#5a9fd4"] * 4 + ["#e8956d"] * 4
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)

ax.set_xticks(range(1, len(city_order) + 1))
ax.set_xticklabels(city_order, rotation=20, ha="right")
ax.set_ylabel("Mean Temperature (°C)")
ax.set_title("Daily mean temperature distribution per city (2004–2023)", fontweight="bold")
ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.35, label="0°C")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
### 2.2 Data quality and missingness

Two columns required specific attention before modelling.

**SNOW_ON_GROUND (64% null)** — nulls here are not missing data in the conventional sense.
Environment Canada stations only report snow depth when snow is present; no report means no snow.
Imputing with 0 is the correct domain decision, not a statistical workaround.

**Temperature columns (~6% null)** — short gaps caused by station maintenance.
These are filled by forward-propagating the last valid reading within each city, capped at 7 days.
Gaps longer than 7 days remain null and are excluded from model training.
"""))

cells.append(code("""\
display(Image(PLOTS / "missingness_heatmap.png", width=820))
"""))

cells.append(md("""\
### 2.3 Climate normals as a baseline

The 30-year climate normals (1981–2010) provide the climatological expected value for each
city and month. Comparing observed temperatures to these baselines is more informative than
looking at raw values: a mean of 5°C in January signals a warm spell; in July, a cold snap.

This framing — *how anomalous is today relative to the long-run normal?* — is the foundation
of the feature engineering in Section 4.
"""))

cells.append(code("""\
display(Image(PLOTS / "temperature_normals.png", width=820))
"""))

# ── SECTION 3: CLEANING ────────────────────────────────────────────────────

cells.append(md("""\
---
## 3. Cleaning Decisions

Each decision in `pipeline/clean.py` has a domain rationale, not just a statistical one.

| Decision | Rationale |
|---|---|
| Impute `SNOW_ON_GROUND` nulls → 0 | Null means no snow reported (see §2.2) |
| Forward-fill temperature gaps ≤ 7 days per city | Short maintenance gaps; longer gaps left null |
| Drop 14 `_FLAG` columns | Internal EC quality codes, not useful for modelling |
| Nullify rows where MAX_TEMP < MIN_TEMP | Data entry errors (14 rows); safer to exclude |
| Exclude Halifax | 100% null temperatures in 2020–2023 — no test-period ground truth |

The forward-fill is applied **within each city separately** and uses a 7-day limit.
Without the per-city grouping, a gap at a city boundary could fill across cities.
Without the 7-day cap, extended outages (e.g., station decommissioning) would silently
propagate stale values rather than producing visible nulls.
"""))

cells.append(code("""\
# Confirm null rates after cleaning
temp_cols = ["MEAN_TEMPERATURE", "MIN_TEMPERATURE", "MAX_TEMPERATURE"]
null_after = df[temp_cols].isnull().mean().mul(100).rename("null % after cleaning")
null_before = pd.Series({"MEAN_TEMPERATURE": 6.0, "MIN_TEMPERATURE": 6.0,
                          "MAX_TEMPERATURE": 6.0}, name="null % before cleaning (approx)")
pd.concat([null_before, null_after], axis=1).round(2)
"""))

# ── SECTION 4: FEATURE ENGINEERING ────────────────────────────────────────

cells.append(md("""\
---
## 4. Feature Engineering

`pipeline/features.py` engineers 25 model-ready features across four families.

**Anomaly features** — how far is today from what is historically expected?
`temp_relative = MEAN_TEMPERATURE − NORMAL_MEAN_TEMP` measures the departure from
the 30-year normal for that city and month. Separate anomalies are computed for MAX and MIN.

**Memory features** — weather has momentum.
A cold snap doesn't appear from nowhere. `temp_lag_1/2/3` (yesterday, two days ago, three days ago)
and `temp_roll_5/7` (5- and 7-day rolling means) give the model the recent temperature trajectory,
which is often as predictive as today's reading.

**Calendar features** — timing shapes meaning.
`day_of_year`, `month`, and `season` let the model learn that a −10°C reading in January
is unremarkable but the same value in May is a genuine anomaly.

**Rainy season** — a city-specific binary flag derived from each city's own climate normals.
Months where normal precipitation exceeds the city's annual monthly average are marked 1.
This is computed from normals only — no leakage from observed daily values.
"""))

cells.append(code("""\
feat = load("climate_features")
feat["LOCAL_DATE"] = pd.to_datetime(feat["LOCAL_DATE"])

print(f"Feature table: {len(feat):,} rows × {feat.shape[1]} columns")
print()

engineered = [
    "temp_relative", "temp_max_relative", "temp_min_relative", "precip_relative",
    "temp_lag_1", "temp_lag_2", "temp_lag_3",
    "temp_roll_5", "temp_roll_7", "precip_roll_7",
    "temp_range", "snow_present",
    "day_of_year", "month", "season", "rainy_season",
]
null_rates = feat[engineered].isnull().mean().mul(100).round(1).rename("null %")
print(null_rates.to_string())
"""))

cells.append(code("""\
# Do anomaly features actually separate extreme from normal days?
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, col, title in zip(axes,
    ["temp_relative", "temp_min_relative"],
    ["Mean temperature anomaly (°C)", "Min temperature anomaly (°C)"],
):
    for val, color, label in [
        (0.0, "#aaaaaa", "Normal day"),
        (1.0, "#e05c2b", "Extreme day"),
    ]:
        vals = feat[feat["is_extreme_day"] == val][col].dropna()
        ax.hist(vals, bins=60, alpha=0.55, color=color, label=label, density=True)
    ax.set_xlabel(title)
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Anomaly features cleanly separate extreme from normal days", fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
"""))

# ── SECTION 5: DEFINING EXTREME ────────────────────────────────────────────

cells.append(md("""\
---
## 5. Defining "Extreme"

### Why not a fixed national threshold?

The intuitive approach — flag any day where MAX > 30°C or MIN < −20°C — fails for Canada
because its climate zones are too diverse:

- MAX = 28°C in Vancouver (July normal: 22°C) is hotter than almost any day on record there
- MIN = −22°C in Winnipeg (January normal: −16°C) is cold but unremarkable

Applying the same numbers across all cities would over-flag prairie winters and under-flag
coastal heat events.

### City-specific 2σ thresholds

Extreme days are instead defined **relative to each city's own training-period distribution**:

$$\\text{warm fence} = \\overline{\\text{MAX}} + 2\\sigma_{\\text{MAX}} \\qquad
  \\text{cold fence} = \\overline{\\text{MIN}} - 2\\sigma_{\\text{MIN}}$$

Both statistics are computed **from the training period only** (2004–2019) to prevent the
target variable from being contaminated by test-period data.

A day is labelled extreme if `MAX_TEMPERATURE > warm_fence` **or** `MIN_TEMPERATURE < cold_fence`.
"""))

cells.append(code("""\
thr = load("city_thresholds").sort_values("upper_fence", ascending=False)
thr = thr.rename(columns={"upper_fence": "warm fence (°C)", "lower_fence": "cold fence (°C)"})
thr.style.hide(axis="index").format(precision=1)
"""))

cells.append(code("""\
rates = (
    feat.groupby("city")["is_extreme_day"]
    .agg(extreme="sum", total="count")
    .assign(pct=lambda d: d["extreme"] / d["total"] * 100)
    .sort_values("pct", ascending=True)
)

fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.barh(rates.index, rates["pct"], color="#3a7abf", alpha=0.8)
ax.axvline(2.3, color="#e05c2b", linewidth=1.3, linestyle="--",
           label="2.3%  expected if distribution were normal (±2σ of a Gaussian)")
ax.set_xlabel("Extreme days (%)")
ax.set_title("Extreme day rate per city (all years)", fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

print(rates[["extreme", "total", "pct"]].rename(
    columns={"extreme": "extreme days", "total": "total days", "pct": "%"}).round(1).to_string())
"""))

# ── SECTION 6: MODELLING ───────────────────────────────────────────────────

cells.append(md("""\
---
## 6. Modelling

### Algorithm: XGBoost

Three properties made XGBoost the right choice for this problem:

1. **Class imbalance tolerance.** Extreme days are ~3% of the dataset. Gradient boosted trees
   handle imbalanced targets without resampling — they iteratively focus on misclassified
   minority examples.

2. **Non-linear interactions.** Whether a temperature anomaly constitutes an extreme day
   depends on city, season, and recent trajectory simultaneously. Trees capture these
   multi-way interactions naturally; a linear model would need them specified manually.

3. **Interpretable feature importance.** Gain-based importance gives a clear post-hoc
   explanation of which features drove decisions — essential for communicating results
   beyond a technical audience.

### The train/test split

The split is **temporal**, not random:

| Set | Years | Days |
|---|---|---|
| Train | 2004–2019 | ~46,000 |
| Test | 2020–2023 | ~12,000 |

A random split would leak future data into training — a model that saw 2022 observations during
training would appear to generalise but would fail in deployment. The forward split simulates
real operational use: learn from history, predict the future.
"""))

cells.append(code("""\
train = feat[feat["LOCAL_YEAR"] <= 2019]
test  = feat[feat["LOCAL_YEAR"] >= 2020]

print(f"Train: {len(train):,} days  |  extreme rate: {train['is_extreme_day'].mean():.2%}")
print(f"Test : {len(test):,} days  |  extreme rate: {test['is_extreme_day'].mean():.2%}")
"""))

cells.append(code("""\
metrics = pd.read_csv(PROJ / "outputs/metrics_by_city.csv")

overall = pd.DataFrame([{
    "city": "Overall (macro avg)", "n_total": metrics["n_total"].sum(),
    "n_extreme": metrics["n_extreme"].sum(),
    "f1": 0.833, "precision": 0.782, "recall": 0.961, "roc_auc": 0.998,
}])
display_df = pd.concat([metrics, overall], ignore_index=True)
display_df = display_df.rename(columns={
    "city": "City", "n_total": "Test days", "n_extreme": "Extreme days",
    "f1": "F1", "precision": "Precision", "recall": "Recall", "roc_auc": "ROC-AUC",
})

(display_df.style
    .hide(axis="index")
    .format({
        "F1": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "ROC-AUC": "{:.4f}",
        "Test days": "{:,}", "Extreme days": "{:,}",
    })
    .apply(lambda row: [
        "font-weight: bold; background-color: #f0f4ff" if row["City"].startswith("Overall") else ""
        for _ in row
    ], axis=1)
)
"""))

cells.append(md("""\
**Interpreting the results:**

- **Recall = 0.961** — the model catches 96% of all true extreme days. Missing only 4% of
  real events is the right trade-off for a climate-risk application where false negatives
  are more costly than false alarms.
- **ROC-AUC ≈ 1.0** across all cities — the model has near-perfect rank-ordering ability,
  consistently scoring extreme days higher than normal days regardless of the exact threshold.
- **Toronto's lower F1 (0.691)** reflects a smaller extreme-day sample (20 events in 4 years),
  making precision volatile — one extra false positive materially shifts the score.

An ablation test confirmed that including the `season` feature improved all four metrics
simultaneously — it was retained.
"""))

cells.append(code("""\
display(Image(PLOTS / "feature_importance.png", width=760))
"""))

cells.append(md("""\
`temp_min_relative` — the departure of today's minimum temperature from its 30-year normal —
is the single most important feature. This makes physical sense: cold extremes carry a
stronger and more consistent signal than warm extremes (see Section 7).

The rolling and lag features (`temp_roll_5`, `temp_lag_1`) confirm that weather trajectory
matters: knowing where temperatures have been over the past week is nearly as informative
as the current reading. Calendar features (`day_of_year`, `season`) contribute substantially,
reflecting high-risk months specific to each city.
"""))

# ── SECTION 7: WARM/COLD ASYMMETRY ─────────────────────────────────────────

cells.append(md("""\
---
## 7. Finding: Why Are There More Cold Extreme Days Than Warm?

Examining the model's predictions revealed a striking asymmetry: cold extreme days
outnumber warm extreme days by 10–100× in most Canadian cities.

The climate-change intuition — that temperatures have been trending warmer since 2000,
so more warm extremes should appear — predicts the opposite. Three checks were run:

1. **Is it the variance?** No. `std(MAX_TEMPERATURE)` is actually *larger* than `std(MIN_TEMPERATURE)`
   in every city. If distributions were symmetric, larger variance would produce *more* warm
   extremes, not fewer.

2. **Is it the thresholds?** The thresholds are ±2σ for both sides, so a symmetric Gaussian
   would produce the same ~2.3% exceedance rate on each tail.

3. **Is it the skewness?** Yes. Canadian temperature distributions are left-skewed — the cold
   tails are fatter than a Gaussian would predict. Arctic air masses produce sharp, episodic
   cold snaps that push `MIN_TEMPERATURE` well beyond the −2σ fence. Summer heat has a
   physical ceiling and dissipates more gradually, leaving the warm tail thin.

**Vancouver is the exception.** Its oceanic climate moderates both extremes, producing
near-symmetric distributions and roughly equal warm/cold extreme counts (131 vs 121 days).
"""))

cells.append(code("""\
display(Image(PLOTS / "temperature_kde.png", width=900))
"""))

cells.append(code("""\
display(Image(PLOTS / "extreme_variance.png", width=900))
"""))

# ── SECTION 8: INTERACTIVE MAP ─────────────────────────────────────────────

cells.append(md("""\
---
## 8. Interactive Map

Each marker is placed at the station coordinates, sized and coloured by the share of
test-period days flagged as extreme. Click any marker for city-level metrics.

Prairie cities (Saskatoon, Calgary, Winnipeg) anchor the high end of the scale;
Vancouver sits at the low end — consistent with the asymmetry finding above.

*If the map does not render inline, open `outputs/maps/climateflow_map.html` in a browser.*
"""))

cells.append(code("""\
IFrame(src=str(MAPS / "climateflow_map.html"), width="100%", height=520)
"""))

# ── CONCLUSION ─────────────────────────────────────────────────────────────

cells.append(md("""\
---
## 9. Conclusions

ClimateFlow shows that city-specific, data-driven thresholds outperform fixed national
cutoffs for defining climate extremes across a geographically diverse country like Canada.
An XGBoost classifier trained on anomaly, memory, and calendar features achieves strong
predictive performance (F1 = 0.833, ROC-AUC = 0.998) on a genuinely forward-looking
test set — no 2020–2023 data was seen during training.

The most interesting finding emerged from diagnosing the output: the dominance of cold
extreme days over warm is a structural property of Canadian temperature distributions,
not a modelling artefact. Left-skewed MIN temperatures, driven by episodic Arctic
intrusions, cause more days to breach the cold −2σ fence than summer heat breaches the
warm +2σ fence. Vancouver's symmetric oceanic climate produces the only city with
near-equal warm and cold counts — a clean natural experiment.

---

**To reproduce from scratch:**
```bash
python -m pipeline.ingest    # fetch data from MSC GeoMet API
python -m pipeline.clean     # clean, impute, merge normals → SQLite
python -m pipeline.features  # engineer 25 features
python -m models.forecast    # train XGBoost, save outputs
python -m viz.map            # interactive map + plots
```

**Repository:** https://github.com/navitd/climateflow
**Data:** Environment and Climate Change Canada — MSC GeoMet OGC API (open licence)
"""))

# ── Write ──────────────────────────────────────────────────────────────────

nb["cells"] = cells

out = Path("notebooks/climateflow_analysis.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, open(out, "w"))
print(f"Saved {len(cells)} cells → {out}")
