"""
Step 2 — Clean
Reads raw CSVs, applies cleaning rules, merges daily data with climate
normals, and saves to SQLite + CSV.
"""
import sqlite3
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
DB_PATH = PROCESSED / "climateflow.db"

# Halifax (LAKE MAJOR) has 100% null temperatures in the test period 2020-2023 —
# excluded so it doesn't create an untestable city in the model.
EXCLUDE_CITIES = ["Halifax"]

KEY_COLS = [
    "MEAN_TEMPERATURE", "MIN_TEMPERATURE", "MAX_TEMPERATURE",
    "TOTAL_PRECIPITATION", "SNOW_ON_GROUND",
    "NORMAL_MEAN_TEMP", "NORMAL_TOTAL_PRECIP",
]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(RAW / "climate_daily_all.csv", low_memory=False)
    normals = pd.read_csv(RAW / "climate_normals.csv")
    return daily, normals


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])

    # Drop quality-code flag columns — not needed downstream
    flag_cols = [c for c in df.columns if c.endswith("_FLAG")]
    df = df.drop(columns=flag_cols)

    # longitude/latitude duplicates lat/lon (added by ingest metadata join)
    df = df.drop(columns=["longitude", "latitude"])

    # SNOW_ON_GROUND: null means no snow was reported on that day → 0
    df["SNOW_ON_GROUND"] = df["SNOW_ON_GROUND"].fillna(0.0)

    # Forward-fill short temperature maintenance gaps per city (≤7 days)
    temp_cols = ["MEAN_TEMPERATURE", "MIN_TEMPERATURE", "MAX_TEMPERATURE"]
    df = df.sort_values(["city", "LOCAL_DATE"])
    df[temp_cols] = df.groupby("city")[temp_cols].transform(
        lambda s: s.ffill(limit=7)
    )

    return df


def repair_temperature_inversions(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["MAX_TEMPERATURE"].notna() &
        df["MIN_TEMPERATURE"].notna() &
        (df["MAX_TEMPERATURE"] < df["MIN_TEMPERATURE"])
    )
    n = mask.sum()
    if n > 0:
        print(f"  Nullifying {n} rows where MAX_TEMPERATURE < MIN_TEMPERATURE:")
        bad = df[mask][["city", "LOCAL_DATE", "MIN_TEMPERATURE", "MAX_TEMPERATURE"]]
        print(bad.to_string(index=False))
        df.loc[mask, ["MAX_TEMPERATURE", "MIN_TEMPERATURE", "MEAN_TEMPERATURE"]] = float("nan")
    return df


def clean_normals(df: pd.DataFrame) -> pd.DataFrame:
    # MONTH 13 is the annual summary row — drop before joining to monthly data
    return df[df["MONTH"] != 13].copy()


def merge_normals(daily: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    normals_slim = normals[[
        "city", "MONTH",
        "NORMAL_MEAN_TEMP", "NORMAL_MAX_TEMP",
        "NORMAL_MIN_TEMP", "NORMAL_TOTAL_PRECIP",
    ]]
    merged = daily.merge(
        normals_slim,
        left_on=["city", "LOCAL_MONTH"],
        right_on=["city", "MONTH"],
        how="left",
    )
    return merged.drop(columns=["MONTH"])


def save(df: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    df_db = df.copy()
    df_db["LOCAL_DATE"] = df_db["LOCAL_DATE"].astype(str)
    conn = sqlite3.connect(DB_PATH)
    df_db.to_sql("climate_daily_clean", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  SQLite : {DB_PATH}  ({len(df):,} rows)")

    csv_path = PROCESSED / "climate_daily_clean.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV    : {csv_path}")


def run() -> None:
    print("=== Step 2 — Clean ===")

    daily, normals = load_raw()
    print(f"Loaded  daily={daily.shape}  normals={normals.shape}")

    daily   = daily[~daily["city"].isin(EXCLUDE_CITIES)].copy()
    normals = normals[~normals["city"].isin(EXCLUDE_CITIES)].copy()
    print(f"Excluded cities {EXCLUDE_CITIES}: daily={daily.shape}  normals={normals.shape}")

    n_before = daily.shape[1]
    daily = clean_daily(daily)
    n_dropped = n_before - daily.shape[1]
    print(f"Cleaned daily → {daily.shape}  (dropped {n_dropped} cols)")

    print("Repairing temperature inversions...")
    daily = repair_temperature_inversions(daily)

    normals = clean_normals(normals)
    print(f"Cleaned normals → {normals.shape}  (dropped month-13 rows)")

    merged = merge_normals(daily, normals)
    print(f"Merged  → {merged.shape}")

    print("\nNull rates on key columns after cleaning:")
    for col in KEY_COLS:
        pct = merged[col].isnull().mean() * 100
        print(f"  {col:<30} {pct:.1f}%")

    save(merged)
    print("\nStep 2 complete.")


if __name__ == "__main__":
    run()
