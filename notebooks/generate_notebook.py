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
3. Engineers temporal features, including anomalies relative to 30-year baselines
4. Trains an XGBoost classifier to **forecast whether tomorrow will be an extreme temperature day**,
   using only information available today, and compares it with two simple baselines
5. Visualises results on an interactive Folium map

**Key result (2020–2023 test period, 9 cities):** F1 = 0.617 and PR-AUC = 0.688 for the forecast,
vs F1 = 0.606 and PR-AUC = 0.380 for persistence ("tomorrow is extreme if today was").
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

This framing — *how anomalous was yesterday relative to the long-run normal?* — is the
foundation of the feature engineering in Section 4.
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

### The forecasting setup

The target (Section 5) is defined from day $t$'s own MAX and MIN temperatures. A model given
those same-day temperatures would simply re-apply the threshold rule and score almost perfectly
while forecasting nothing. ClimateFlow therefore poses a **next-day forecast**: predict whether
day $t$ is extreme using only information available at the end of day $t-1$.

`pipeline/features.py` builds three families of model inputs. Every observation-based input
ends on day $t-1$.

**Previous-day anomalies.** How far was yesterday from what is historically expected?
For any observed variable $T$, the anomaly is

$$\\Delta_{c,t-1} = T_{c,t-1} - N_{c,m(t-1)}$$

where $T_{c,t-1}$ is the value observed in city $c$ on day $t-1$, $N_{c,m}$ is the 1981–2010
climate normal for city $c$ and month $m$, and $m(t-1)$ is the month of day $t-1$.
This gives `prev_temp_relative`, `prev_temp_max_relative`, `prev_temp_min_relative` and
`prev_precip_relative`. `prev_temp_range` and `prev_snow_present` complete the set.

**Memory features.** Weather has momentum. `temp_lag_1/2/3` are the mean temperatures of days
$t-1$, $t-2$ and $t-3$. `temp_roll_5/7` and `precip_roll_7` are means and totals over windows
that end on day $t-1$.

**Calendar features.** These are known in advance. `day_of_year`, `month`, `season` and
`rainy_season` describe day $t$ itself. `rainy_season` is computed from climate normals only.

Two safeguards are built in:
- Lags, rolling windows and previous-day values are computed on a full daily calendar per city.
  A day after a station gap therefore gets NaN for "yesterday", not an older day's value.
- `tests/test_features.py::TestForecastInputsExcludeDayT` changes one day's observed weather
  and checks that none of that day's model inputs change.
"""))

cells.append(code("""\
feat = load("climate_features")
feat["LOCAL_DATE"] = pd.to_datetime(feat["LOCAL_DATE"])

print(f"Feature table: {len(feat):,} rows × {feat.shape[1]} columns")
print()

model_inputs = [
    "prev_temp_relative", "prev_temp_max_relative", "prev_temp_min_relative",
    "prev_precip_relative", "prev_temp_range", "prev_snow_present",
    "temp_lag_1", "temp_lag_2", "temp_lag_3",
    "temp_roll_5", "temp_roll_7", "precip_roll_7",
    "day_of_year", "month", "season", "rainy_season",
]
null_rates = feat[model_inputs].isnull().mean().mul(100).round(1).rename("null %")
print(null_rates.to_string())
"""))

cells.append(md("""\
### Why same-day features would be leakage

The left panel shows **today's** minimum-temperature anomaly. Extreme and normal days separate
almost perfectly, but only because the extreme label is defined from that same minimum
temperature. The right panel shows **yesterday's** anomaly, which is what a forecaster
actually has. The two groups still differ, but they overlap. That overlap is the real
forecasting problem.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

for ax, col, title in zip(axes,
    ["temp_min_relative", "prev_temp_min_relative"],
    ["Same day: MIN anomaly on day t (leaks the label)",
     "Forecast input: MIN anomaly on day t−1"],
):
    for val, color, label in [
        (0.0, "#aaaaaa", "Normal day t"),
        (1.0, "#e05c2b", "Extreme day t"),
    ]:
        vals = feat[feat["is_extreme_day"] == val][col].dropna()
        ax.hist(vals, bins=60, alpha=0.55, color=color, label=label, density=True)
    ax.set_xlabel("Min temperature anomaly (°C)")
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Density")

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

The split is **temporal**, not random, with a validation window inside the training years:

| Set | Years | Used for |
|---|---|---|
| Fit | 2004–2016 | Fitting candidate models |
| Validation | 2017–2019 | Choosing the decision threshold and the feature set |
| Train (refit) | 2004–2019 | Fitting the final model |
| Test | 2020–2023 | Final scores only |

A random split would leak future data into training. The test years are never used to
choose anything.

### Decision threshold

The model outputs a probability $\\hat p_{c,t}$ and flags day $t$ when $\\hat p_{c,t} \\ge \\tau$.
The threshold $\\tau$ is the value in $\\{0.05, 0.06, \\dots, 0.95\\}$ that maximises F1 on the
validation years. Training uses `scale_pos_weight` ≈ 29 to offset the ~3% extreme-day rate.
That pushes probabilities up, so the default $\\tau = 0.5$ would flood the forecast with false alarms.

### Baselines

- **Persistence:** $\\hat y_{c,t} = y_{c,t-1}$, i.e. tomorrow is extreme if today was. A missing
  $y_{c,t-1}$ counts as 0.
- **Seasonal base rate:** the score is $r_{c,m(t)}$, the share of extreme days in city $c$ and
  month $m(t)$ during 2004–2019. A day is flagged when $r_{c,m(t)} \\ge \\kappa$, where $\\kappa$
  is the cut-off that maximises F1 on the training years.

### Metrics

With $TP$, $FP$ and $FN$ the counts of true positives, false positives and false negatives:

$$\\text{Precision} = \\frac{TP}{TP+FP}, \\qquad \\text{Recall} = \\frac{TP}{TP+FN}, \\qquad
F_1 = \\frac{2\\,\\text{Precision}\\cdot\\text{Recall}}{\\text{Precision}+\\text{Recall}}$$

**PR-AUC** (average precision) is $\\sum_n (R_n - R_{n-1})\\,P_n$, where $P_n$ and $R_n$ are the
precision and recall when days are flagged down to the $n$-th highest score. **ROC-AUC** is the
probability that a randomly chosen extreme day scores higher than a randomly chosen normal day.
With ~97% normal days, ROC-AUC is high for almost any method. PR-AUC is the more telling score.
"""))

cells.append(code("""\
train = feat[feat["LOCAL_YEAR"] <= 2019]
test  = feat[feat["LOCAL_YEAR"] >= 2020]

print(f"Train: {len(train):,} days  |  extreme rate: {train['is_extreme_day'].mean():.2%}")
print(f"Test : {len(test):,} days  |  extreme rate: {test['is_extreme_day'].mean():.2%}")
"""))

cells.append(code("""\
results = pd.read_csv(PROJ / "outputs/ablation_results.csv")
labels = {
    "baseline_climatology":  "Baseline: seasonal base rate",
    "baseline_persistence":  "Baseline: persistence",
    "with_season":           "XGBoost: with season",
    "without_season":        "XGBoost: without season",
    "with_season_and_rainy": "XGBoost: with season + rainy season",
}
results["method"] = results["feature_set"].map(labels)
cols = ["method", "f1", "precision", "recall", "roc_auc", "pr_auc", "val_f1", "threshold", "winner"]
(results[cols]
    .rename(columns={"f1": "F1", "precision": "Precision", "recall": "Recall",
                     "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
                     "val_f1": "Validation F1", "threshold": "Threshold"})
    .style.hide(axis="index")
    .format(precision=3, na_rep="—")
    .apply(lambda row: ["font-weight: bold; background-color: #f0f4ff" if row["winner"] else ""
                        for _ in row], axis=1)
)
"""))

cells.append(code("""\
display(Image(PLOTS / "baseline_comparison.png", width=700))
"""))

cells.append(md("""\
**Interpreting the results:**

- **Persistence is a strong baseline.** Extreme cold and heat come in multi-day spells, so
  "same as today" alone reaches F1 = 0.61. At its validation-chosen threshold, the forecast
  only slightly improves on that (F1 = 0.617 on the test years).
- **The forecast's advantage is ranking.** Its PR-AUC is 0.69, against 0.38 for persistence and
  0.13 for the seasonal base rate. It gives a graded probability that can be traded between
  missed events and false alarms. Persistence gives only a yes/no answer.
- **Feature-set differences are noise.** `season` and `rainy_season` change validation F1 by
  less than 0.01. The saved model is the set with the highest validation F1 (`without_season`).
"""))

cells.append(code("""\
metrics = pd.read_csv(PROJ / "outputs/metrics_by_city.csv")
(metrics.rename(columns={
    "city": "City", "n_total": "Test days", "n_extreme": "Extreme days",
    "f1": "F1", "precision": "Precision", "recall": "Recall",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
  })
  .style.hide(axis="index")
  .format({"F1": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
           "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
           "Test days": "{:,}", "Extreme days": "{:,}"})
)
"""))

cells.append(md("""\
The Prairie cities (Calgary, Edmonton, Saskatoon) score highest, with F1 = 0.72–0.77. They have the
most extreme days, and those come mostly in long Arctic cold spells that are visible a day early.
Toronto, Ottawa and Montréal have only 20–25 extreme days in four years. Their scores are lower
and noisy: a couple of hits or misses moves F1 by several points.
"""))

cells.append(code("""\
display(Image(PLOTS / "feature_importance.png", width=760))
"""))

cells.append(md("""\
`temp_lag_1`, yesterday's mean temperature, is by far the most important input. The second is
`prev_temp_min_relative`, yesterday's minimum-temperature anomaly. In a next-day forecast, the
most recent observation carries most of the signal. Calendar features (`day_of_year`, `month`)
add the seasonal context: the same anomaly matters more in some months than in others.
"""))

# ── SECTION 7: WARM/COLD ASYMMETRY ─────────────────────────────────────────

cells.append(md("""\
---
## 7. Finding: Why Are There More Cold Extreme Days Than Warm?

Examining the extreme-day labels revealed a striking asymmetry: cold extreme days
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

Each marker is placed at the station coordinates, sized and coloured by the city's share of
extreme days over 2004–2023. Click any marker to see the actual and forecast extreme-day
counts for 2020–2023 and the city's forecast F1.

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

ClimateFlow shows that city-specific, data-driven thresholds are a better way than fixed
national cutoffs to define climate extremes across a country as varied as Canada.

Framed honestly as a next-day forecast, the problem is hard. Persistence ("tomorrow is
extreme if today was") is already a strong baseline (F1 = 0.61). On the 2020–2023 test years,
the XGBoost forecast matches and slightly beats it on F1 (0.617). It nearly doubles
persistence's PR-AUC (0.69 vs 0.38), which makes it a usable risk score rather than a
yes/no rule. No 2020–2023 data was used for training, threshold choice or feature-set selection.

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
python -m pipeline.features  # engineer next-day forecast features
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
