# ClimateFlow — StoryMap Content Script

Use this as a copy-paste guide when building the StoryMap at storymaps.arcgis.com.
Each section is labelled with the StoryMap block type to use.

---

## COVER

**Block type:** Cover (full-width, with title + subtitle)

**Title:**
ClimateFlow: Forecasting Extreme Climate Days Across Canada

**Subtitle:**
A data pipeline and machine learning project built on 20 years of Environment Canada weather records

**Background image:** `outputs/plots/time_series.png`
*(Upload this image as the cover background — it shows forecast vs actual extreme days per city and year in 2020–2023, and sets the geographic scope immediately)*

---

## SECTION 1 — The Question

**Block type:** Text block

What counts as an extreme weather day in Canada?

A day where temperatures hit −25°C is routine in Winnipeg in January. The same temperature in Vancouver would shut down the city. A fixed national threshold — flag any day below −20°C, or above 30°C — treats these two events the same. It shouldn't.

ClimateFlow is a full end-to-end data pipeline and machine learning project that answers this question city by city, using 20 years of daily observations from Environment and Climate Change Canada's open data API. It then asks a harder question: using only what is known today, can we tell whether tomorrow will be extreme? The resulting next-day forecast was trained on 2004–2019 and tested on 2020–2023. It reaches F1 = 0.62, and it ranks extreme-day risk far better than two simple baselines.

---

## SECTION 2 — The Data

**Block type:** Text block + image (side by side)

**Text:**
All data comes from Canada's MSC GeoMet OGC API — a public, open-licence endpoint maintained by Environment and Climate Change Canada. No manual downloads. The pipeline fetches everything programmatically:

- **Daily climate observations** — 70,500 rows of daily MAX, MIN, and MEAN temperature, precipitation, snow depth, humidity, and wind gust across 9 cities from 2004 to 2023
- **30-year climate normals** — monthly averages from 1981–2010, serving as the historical baseline for each city
- **Station metadata** — coordinates and elevation for over 8,600 stations across Canada

Nine cities were selected to represent Canada's main climate zones: Vancouver (oceanic), Calgary and Edmonton (semi-arid continental), Saskatoon and Winnipeg (cold continental), Toronto and Ottawa (humid continental), and Montréal and Québec City (humid continental with maritime influence).

**Image:** `outputs/plots/temperature_normals.png`
**Caption:** 30-year climate normals vs observed monthly means per city. The gap between the lines is the temperature anomaly — one of the forecast model's main inputs.

---

## SECTION 3 — The Interactive Map

**Block type:** Embed block (full width)

**Embed URL:**
https://navitd.github.io/climateflow/outputs/maps/climateflow_map.html

**Text above the embed:**
Each marker shows a weather station location. Size and colour show each city's share of extreme days over 2004–2023, from green (lower) to red (higher). Click a marker to see the actual and forecast number of extreme days in 2020–2023 and the city's forecast F1.

Prairie cities dominate the high end of the scale. Vancouver sits at the low end — a pattern explained in Section 6.

---

## SECTION 4 — Defining "Extreme" City by City

**Block type:** Text block + image (side by side)

**Text:**
Rather than applying a fixed national threshold, ClimateFlow defines extreme days relative to each city's own historical distribution:

**Warm fence** = mean(MAX temperature) + 2 standard deviations
**Cold fence** = mean(MIN temperature) − 2 standard deviations

In symbols: $U_c = \mu^{\max}_c + 2\sigma^{\max}_c$ and $L_c = \mu^{\min}_c - 2\sigma^{\min}_c$, where $\mu^{\max}_c$ and $\sigma^{\max}_c$ are the mean and standard deviation of city $c$'s daily maximum temperature over 2004–2019, and $\mu^{\min}_c$, $\sigma^{\min}_c$ the same for its daily minimum temperature.

Both are computed from training data only (2004–2019) — the test years (2020–2023) never touch the threshold calculation. A day is labelled extreme if MAX exceeds the warm fence or MIN falls below the cold fence.

This produces city-specific thresholds that reflect local climate reality:

| City | Warm fence | Cold fence |
|---|---|---|
| Vancouver | 27.7°C | −1.9°C |
| Toronto | 35.4°C | −13.9°C |
| Calgary | 33.9°C | −21.1°C |
| Winnipeg | 38.5°C | −27.0°C |
| Saskatoon | 37.2°C | −26.3°C |

Winnipeg's cold fence is 25°C colder than Vancouver's — which is exactly what you would expect from the climatology.

**Image:** `outputs/plots/distributions.png`
**Caption:** Temperature distributions per city. The spread illustrates why a single national threshold would be climatologically indefensible.

---

## SECTION 5 — The Model

**Block type:** Text block + image (side by side)

**Text:**
Whether a day is extreme is decided by that day's own MAX and MIN temperatures. A model that sees those temperatures has nothing left to predict. ClimateFlow therefore poses the question a forecaster would face: **at the end of today, will tomorrow be extreme?** Every input built from observed weather stops at today.

The classifier is an XGBoost gradient-boosted tree with three families of inputs:

**Recent weather.** Today's temperature anomaly compared with the 30-year normal for the city and month, the mean temperatures of the last three days, and 5- and 7-day averages ending today. Weather has momentum, so a cold snap underway today is the best single clue for tomorrow.

**Calendar features.** Tomorrow's day of year and month. These are known in advance, and they tell the model which part of the annual cycle it is in.

**Rainy season.** A binary flag derived from each city's own climate normals. It marks the months when precipitation is climatologically above average.

The split is strictly temporal. The model learns from 2004–2016. The decision threshold is chosen on 2017–2019, the model is refitted on 2004–2019, and it is scored once on 2020–2023.

**Image:** `outputs/plots/feature_importance.png`
**Caption:** XGBoost feature importance (gain). Yesterday's mean temperature (`temp_lag_1`) is the dominant input, followed by yesterday's minimum-temperature anomaly. In a next-day forecast, the latest observation carries the most information.

---

## SECTION 6 — Results

**Block type:** Text block + image (side by side)

**Text:**
A forecast only means something when it's compared with simpler alternatives. Across all nine cities, 2020–2023 (12,743 days, 411 of them extreme):

| Method | F1 | PR-AUC |
|---|---|---|
| Seasonal base rate (how often this month is extreme in this city) | 0.22 | 0.13 |
| Persistence (tomorrow is extreme if today was) | 0.61 | 0.38 |
| **XGBoost forecast** | **0.62** | **0.69** |

Persistence is hard to beat on F1. Extreme cold and heat arrive in multi-day spells, so "same as today" is often right. The model's real advantage is ranking. Its PR-AUC is nearly twice that of persistence, so it gives a reliable probability that can be tuned toward fewer misses or fewer false alarms. Persistence offers only a yes/no answer.

The scores are defined as follows, where $TP$, $FP$ and $FN$ are the numbers of correctly flagged extreme days, false alarms and missed extreme days:

$$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}, \quad F_1 = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

PR-AUC (average precision) is $\sum_n (R_n - R_{n-1})\,P_n$, where $P_n$ and $R_n$ are the precision and recall when the $n$ highest-scoring days are flagged. It summarises how well the forecast ranks rare extreme days across all thresholds.

**Image:** `outputs/plots/baseline_comparison.png`
**Caption:** Next-day forecast vs the persistence and seasonal base-rate baselines on the 2020–2023 test period.

**Second image (optional):** `outputs/plots/city_f1.png`
**Caption:** Forecast F1 per city. The Prairie cities (Calgary, Edmonton, Saskatoon) score highest, with F1 = 0.72–0.77. Their many extreme days mostly come in long Arctic cold spells that show up a day early. Toronto, Ottawa and Montréal have only 20–25 extreme days in four years, so their scores are lower and noisier.

---

## SECTION 7 — An Unexpected Finding

**Block type:** Text block + image (full width)

**Text:**
Examining the extreme-day labels revealed a striking asymmetry: cold extreme days outnumber warm extreme days by 10 to 100 times in most cities.

The climate-change intuition — that temperatures have been trending warmer since 2000, so more warm extremes should appear — predicts the opposite. The explanation turns out to be in the shape of the distributions, not their means or variances.

Canadian MIN temperature distributions are **left-skewed**: the cold tail is fatter than a normal distribution would predict. Arctic air masses produce sharp, episodic cold snaps that push MIN temperatures well beyond the −2σ fence, more often than summer heat pushes MAX temperatures above the +2σ fence.

**Vancouver is the exception.** Its oceanic climate moderates both extremes, producing near-symmetric distributions and roughly equal warm and cold extreme counts — 131 warm days versus 121 cold days over the full dataset. Every other city skews heavily cold.

This is not a modelling artefact. It is a structural property of Canadian continental climate: cold extremes are more frequent and more severe than warm extremes, at least for the 2004–2023 period.

**Image:** `outputs/plots/temperature_kde.png`
**Caption:** Actual temperature distributions (KDE) per city. The shaded areas show days beyond the ±2σ extreme fence. The blue (MIN temperature) shaded regions are consistently larger than the orange (MAX temperature) regions — the cold tail is fatter in every continental city.

---

## SECTION 8 — Seasonal Risk

**Block type:** Text block + image (side by side)

**Image:** `outputs/plots/seasonal_risk.png`
**Caption:** Observed extreme-day frequency by month and city, 2004–2023, split into cold and heat extremes.

**Text:**
The seasonal breakdown confirms what the climate zones predict: winter dominates extreme-day counts across the prairies, while Vancouver shows a more balanced seasonal pattern. Spring and fall see the fewest extremes in most cities — the transitional seasons buffer against both Arctic cold and summer heat.

---

## CLOSING

**Block type:** Text block

**Heading:** Reproducible from a Single Command

The entire pipeline — from API ingestion to trained model to interactive map — runs end-to-end with five commands and no manual data downloads. The data source is open-licence, the code is public on GitHub under the MIT licence, and the outputs are deterministic given the same training period. An automated test checks that no input for a forecast day depends on that day's own weather.

**GitHub repository:** github.com/navitd/climateflow
**Data source:** Environment and Climate Change Canada — MSC GeoMet OGC API

---

*Built with Python · pandas · XGBoost · GeoPandas · Folium · SQLite · Matplotlib*

---

## ASSEMBLY NOTES

**Suggested StoryMap template:** Cascade (full-width sections with scrolling narrative)

**Image uploads needed (in order):**
1. `outputs/plots/time_series.png` — cover background
2. `outputs/plots/temperature_normals.png` — Section 2
3. `outputs/plots/distributions.png` — Section 4
4. `outputs/plots/feature_importance.png` — Section 5
5. `outputs/plots/baseline_comparison.png` — Section 6
6. `outputs/plots/city_f1.png` — Section 6 (optional second image)
7. `outputs/plots/temperature_kde.png` — Section 7 (full width)
8. `outputs/plots/seasonal_risk.png` — Section 8

**Embed block (Section 3):**
URL: https://navitd.github.io/climateflow/outputs/maps/climateflow_map.html
Set height to at least 500px in the embed settings.

**Colour palette (optional, for visual consistency):**
- Warm/orange: #e05c2b
- Cool/blue: #3a7abf
- Text: #1a1a1a
