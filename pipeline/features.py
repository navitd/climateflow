"""
Step 3 — Feature Engineering
Reads climate_daily_clean from SQLite, engineers all model-ready features,
and saves climate_features + city_thresholds tables.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("data/processed/climateflow.db")
TRAIN_END_YEAR = 2019
FULL_START = "2004-01-01"
FULL_END   = "2023-12-31"

NULL_RATE_CEILING = 0.15
EXTREME_RATE_MIN  = 0.01
EXTREME_RATE_MAX  = 0.10


# ---------------------------------------------------------------------------
# Load + entry validation
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM climate_daily_clean", conn)
    conn.close()

    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])

    # --- entry validation ---
    if df["city"].isnull().sum() > 0:
        raise ValueError("city column has unexpected NaNs — pipeline integrity failure")

    if df["LOCAL_DATE"].isnull().sum() > 0:
        raise ValueError("LOCAL_DATE column has NaT values — date parsing failed in Step 2")

    for col in ["MEAN_TEMPERATURE", "MAX_TEMPERATURE", "MIN_TEMPERATURE", "TOTAL_PRECIPITATION"]:
        rate = df[col].isnull().mean()
        if rate > NULL_RATE_CEILING:
            raise ValueError(
                f"{col} null rate is {rate:.1%} — exceeds {NULL_RATE_CEILING:.0%} ceiling. "
                "Indicates a pipeline failure, not normal station downtime."
            )

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Threshold computation (training period only)
# ---------------------------------------------------------------------------

def compute_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["LOCAL_DATE"].dt.year <= TRAIN_END_YEAR]
    thresholds = train.groupby("city").agg(
        upper_fence=("MAX_TEMPERATURE", lambda s: s.mean() + 2 * s.std()),
        lower_fence=("MIN_TEMPERATURE", lambda s: s.mean() - 2 * s.std()),
    ).round(2).reset_index()
    print(f"Thresholds computed from training period (2004–{TRAIN_END_YEAR}):")
    print(thresholds.to_string(index=False))
    return thresholds


# ---------------------------------------------------------------------------
# Reindex to full calendar (guard against date-gap bleed in lags/rolling)
# ---------------------------------------------------------------------------

def reindex_to_full_calendar(df: pd.DataFrame) -> tuple:
    # pre-conditions
    if df["city"].isnull().sum() > 0:
        raise ValueError("city has NaNs before reindexing — cannot use is_inserted flag safely")

    dupes = df[df.duplicated(subset=["city", "LOCAL_DATE"])]
    if len(dupes) > 0:
        sample = dupes[["city", "LOCAL_DATE"]].head(5).to_string(index=False)
        raise ValueError(f"Duplicate city+date rows found:\n{sample}")

    n_before = len(df)
    full_index = pd.date_range(FULL_START, FULL_END, freq="D")

    frames = []
    for city_name, group in df.groupby("city", sort=False):
        original_dates = set(group["LOCAL_DATE"])
        expanded = (
            group.set_index("LOCAL_DATE")
                 .reindex(full_index)
                 .rename_axis("LOCAL_DATE")
                 .reset_index()
        )
        expanded["city"] = city_name
        expanded["is_inserted"] = ~expanded["LOCAL_DATE"].isin(original_dates)
        frames.append(expanded)

    result = pd.concat(frames, ignore_index=True)
    n_inserted = int(result["is_inserted"].sum())
    print(f"Reindexed: inserted {n_inserted:,} placeholder rows across all cities")
    return result, n_before


def drop_inserted(df: pd.DataFrame, n_before: int) -> pd.DataFrame:
    df = df[~df["is_inserted"]].drop(columns=["is_inserted"]).reset_index(drop=True)

    if len(df) != n_before:
        raise ValueError(
            f"Row count changed after reindexing: expected {n_before:,}, got {len(df):,}. "
            "Real rows were accidentally dropped."
        )
    return df


# ---------------------------------------------------------------------------
# Relative features
# ---------------------------------------------------------------------------

def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    if df["NORMAL_MEAN_TEMP"].isnull().sum() > 0:
        affected = df[df["NORMAL_MEAN_TEMP"].isnull()]["city"].unique().tolist()
        raise ValueError(f"NORMAL_MEAN_TEMP has NaNs for cities: {affected}")

    df["temp_relative"]     = df["MEAN_TEMPERATURE"] - df["NORMAL_MEAN_TEMP"]
    df["temp_max_relative"] = df["MAX_TEMPERATURE"]  - df["NORMAL_MAX_TEMP"]
    df["temp_min_relative"] = df["MIN_TEMPERATURE"]  - df["NORMAL_MIN_TEMP"]

    days_in_month = df["LOCAL_DATE"].dt.days_in_month
    df["precip_relative"] = (
        df["TOTAL_PRECIPITATION"] - (df["NORMAL_TOTAL_PRECIP"] / days_in_month)
    )
    return df


# ---------------------------------------------------------------------------
# Lag features (computed after reindexing)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "LOCAL_DATE"]).reset_index(drop=True)
    for lag in [1, 2, 3]:
        df[f"temp_lag_{lag}"] = (
            df.groupby("city")["MEAN_TEMPERATURE"].shift(lag)
        )
    return df


# ---------------------------------------------------------------------------
# Rolling features (computed after reindexing)
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "LOCAL_DATE"]).reset_index(drop=True)

    df["temp_roll_5"] = (
        df.groupby("city")["MEAN_TEMPERATURE"]
        .transform(lambda s: s.rolling(5, min_periods=3).mean())
    )
    df["temp_roll_7"] = (
        df.groupby("city")["MEAN_TEMPERATURE"]
        .transform(lambda s: s.rolling(7, min_periods=4).mean())
    )
    df["precip_roll_7"] = (
        df.groupby("city")["TOTAL_PRECIPITATION"]
        .transform(lambda s: s.rolling(7, min_periods=4).sum())
    )
    return df


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["temp_range"]    = df["MAX_TEMPERATURE"] - df["MIN_TEMPERATURE"]
    df["snow_present"]  = (df["SNOW_ON_GROUND"] > 0).astype(float)

    negative = (df["temp_range"] < 0).sum()
    if negative > 0:
        bad = df[df["temp_range"] < 0][["city", "LOCAL_DATE", "temp_range"]].head(5)
        raise ValueError(
            f"temp_range is negative for {negative} rows — MAX/MIN temperatures are corrupted:\n"
            f"{bad.to_string(index=False)}"
        )
    return df


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_year"] = df["LOCAL_DATE"].dt.day_of_year
    df["month"]       = df["LOCAL_DATE"].dt.month

    def to_season(month):
        if month in (12, 1, 2):  return 0  # winter
        if month in (3, 4, 5):   return 1  # spring
        if month in (6, 7, 8):   return 2  # summer
        return 3                            # fall

    df["season"] = df["month"].map(to_season)
    return df


# ---------------------------------------------------------------------------
# Rainy season feature
# ---------------------------------------------------------------------------

def _find_contiguous_runs(wet_months: set) -> list:
    """Return list of contiguous month runs, handling Dec–Jan wraparound."""
    if not wet_months:
        return []
    months = sorted(wet_months)
    runs, current = [], [months[0]]
    for m in months[1:]:
        if m == current[-1] + 1:
            current.append(m)
        else:
            runs.append(current)
            current = [m]
    runs.append(current)
    # Merge Dec–Jan wraparound into one season
    if len(runs) > 1 and runs[0][0] == 1 and runs[-1][-1] == 12:
        runs = [runs[-1] + runs[0]] + runs[1:-1]
    return runs


def add_rainy_season_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each city × month as rainy_season=1 if NORMAL_TOTAL_PRECIP is above
    that city's own monthly mean. Contiguous wet months form seasons; at most 3
    seasons are kept per city (by total precipitation). Based on normals only —
    no leakage from observed daily values.
    """
    normals = (
        df.groupby(["city", "LOCAL_MONTH"])["NORMAL_TOTAL_PRECIP"]
        .first()
        .reset_index()
    )

    records = []
    print("Rainy season labels derived from climate normals:")
    for city, group in normals.groupby("city"):
        monthly   = group.set_index("LOCAL_MONTH")["NORMAL_TOTAL_PRECIP"].reindex(range(1, 13))
        threshold = monthly.mean()
        wet_months = set(monthly[monthly > threshold].index.tolist())

        runs = _find_contiguous_runs(wet_months)

        if len(runs) > 3:
            run_totals = sorted(
                [(sum(monthly.get(m, 0) for m in run), run) for run in runs],
                reverse=True,
            )
            runs = [r for _, r in run_totals[:3]]

        rainy_months = {m for run in runs for m in run}

        rainy_str = str(sorted(rainy_months)) if rainy_months else "none"
        print(f"  {city:<12} months {rainy_str}")

        for month in range(1, 13):
            records.append({
                "city":         city,
                "LOCAL_MONTH":  month,
                "rainy_season": 1 if month in rainy_months else 0,
            })

    rainy_df = pd.DataFrame(records)
    df = df.merge(rainy_df, on=["city", "LOCAL_MONTH"], how="left")
    return df


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def add_target(df: pd.DataFrame, thresholds: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    df = df.merge(thresholds, on="city", how="left")

    both_null = df["MAX_TEMPERATURE"].isnull() & df["MIN_TEMPERATURE"].isnull()
    extreme   = (
        (df["MAX_TEMPERATURE"] > df["upper_fence"]) |
        (df["MIN_TEMPERATURE"] < df["lower_fence"])
    )

    df["is_extreme_day"] = extreme.astype(float)
    df.loc[both_null, "is_extreme_day"] = np.nan

    # validation
    valid = df["is_extreme_day"].dropna()
    assert set(valid.unique()).issubset({0.0, 1.0}), "is_extreme_day contains unexpected values"

    if validate:
        rates = df.groupby("city")["is_extreme_day"].mean()
        bad = rates[(rates < EXTREME_RATE_MIN) | (rates > EXTREME_RATE_MAX)]
        if len(bad) > 0:
            details = "\n".join(f"  {city}: {rate:.1%}" for city, rate in bad.items())
            raise ValueError(
                f"Extreme day rate outside [{EXTREME_RATE_MIN:.0%}, {EXTREME_RATE_MAX:.0%}] "
                f"for:\n{details}\nCheck threshold computation."
            )
    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    feature_cols = [
        "LOCAL_DATE", "city", "LOCAL_YEAR", "LOCAL_MONTH", "LOCAL_DAY",
        "MEAN_TEMPERATURE", "MAX_TEMPERATURE", "MIN_TEMPERATURE",
        "TOTAL_PRECIPITATION", "SNOW_ON_GROUND",
        "temp_relative", "temp_max_relative", "temp_min_relative", "precip_relative",
        "temp_lag_1", "temp_lag_2", "temp_lag_3",
        "temp_roll_5", "temp_roll_7", "precip_roll_7",
        "temp_range", "snow_present",
        "day_of_year", "month", "season", "rainy_season",
        "is_extreme_day",
        "lat", "lon", "elevation", "province",
    ]
    out = df[feature_cols].copy()
    out["LOCAL_DATE"] = out["LOCAL_DATE"].astype(str)

    conn = sqlite3.connect(DB_PATH)
    out.to_sql("climate_features", conn, if_exists="replace", index=False)
    thresholds.to_sql("city_thresholds", conn, if_exists="replace", index=False)
    conn.close()

    csv_path = Path("data/processed/climate_features.csv")
    out.to_csv(csv_path, index=False)

    print(f"  SQLite : climate_features ({len(out):,} rows), city_thresholds")
    print(f"  CSV    : {csv_path}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def report(df: pd.DataFrame) -> None:
    feature_cols = [
        "temp_relative", "temp_max_relative", "temp_min_relative", "precip_relative",
        "temp_lag_1", "temp_lag_2", "temp_lag_3",
        "temp_roll_5", "temp_roll_7", "precip_roll_7",
        "temp_range", "snow_present",
        "day_of_year", "month", "season", "rainy_season",
    ]
    print("\nNull rates on engineered features:")
    for col in feature_cols:
        pct = df[col].isnull().mean() * 100
        print(f"  {col:<25} {pct:.1f}%")

    print("\nTarget class balance per city:")
    summary = df.groupby("city")["is_extreme_day"].agg(
        total="count", extreme="sum"
    )
    summary["pct"] = (summary["extreme"] / summary["total"] * 100).round(1)
    print(summary.to_string())
    total = summary["extreme"].sum()
    total_days = summary["total"].sum()
    print(f"\nOverall: {int(total):,} extreme days / {int(total_days):,} ({total/total_days*100:.1f}%)")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run() -> None:
    print("=== Step 3 — Feature Engineering ===\n")

    df = load()
    thresholds = compute_thresholds(df)

    df, n_before = reindex_to_full_calendar(df)
    df = add_lag_features(df)       # must run while inserted rows are present
    df = add_rolling_features(df)   # must run while inserted rows are present
    df = drop_inserted(df, n_before)

    df = add_relative_features(df)
    df = add_derived_features(df)
    df = add_calendar_features(df)
    df = add_rainy_season_feature(df)
    df = add_target(df, thresholds)

    report(df)
    save(df, thresholds)
    print("\nStep 3 complete.")


if __name__ == "__main__":
    run()
