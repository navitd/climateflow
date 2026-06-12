"""
pipeline/ingest.py

Pulls three datasets from public APIs and saves raw CSVs to data/raw/:
  - climate_stations.csv      : metadata for all Canadian climate stations
  - climate_daily_<city>.csv  : daily observations for each target city
  - climate_daily_all.csv     : combined daily file across all cities
  - climate_normals.csv       : 1981-2010 monthly normals per station
                                (used downstream to compute anomalies)
"""

import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
BASE_URL = "https://api.weather.gc.ca/collections"

START_DATE = "2004-01-01"
END_DATE   = "2023-12-31"

# Target cities: name → approximate lat/lon used to find the nearest active station
CITIES = {
    "Ottawa":     (45.42,  -75.69),
    "Toronto":    (43.65,  -79.38),
    "Montreal":   (45.50,  -73.57),
    "Vancouver":  (49.25, -123.12),
    "Calgary":    (51.05, -114.07),
    "Winnipeg":   (49.90,  -97.13),
    "Halifax":    (44.65,  -63.57),
    "Edmonton":   (53.55, -113.50),
    "Quebec":     (46.81,  -71.21),
    "Saskatoon":  (52.13, -106.67),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, params: dict, retries: int = 3) -> dict:
    """GET with retry/back-off. Raises on persistent failure."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"API request failed after {retries} attempts: {e}")


def _features_to_df(features: list) -> pd.DataFrame:
    """Flatten a GeoJSON FeatureCollection into a flat DataFrame."""
    rows = []
    for feat in features:
        row = dict(feat["properties"])
        coords = feat.get("geometry", {}).get("coordinates")
        if coords:
            row["longitude"] = coords[0]
            row["latitude"]  = coords[1]
        rows.append(row)
    return pd.DataFrame(rows)


def _paginate(endpoint: str, params: dict, page_size: int = 10_000) -> list:
    """Collect all features from a paginated OGC API endpoint."""
    params = {**params, "f": "json", "limit": page_size, "offset": 0}
    all_features = []
    while True:
        data = _get_json(endpoint, params)
        batch = data.get("features", [])
        all_features.extend(batch)
        if len(batch) < page_size:
            break
        params["offset"] += page_size
        time.sleep(0.2)
    return all_features


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_stations() -> pd.DataFrame:
    """
    Download metadata for all Canadian climate stations.
    Returns a DataFrame and saves to data/raw/climate_stations.csv.
    """
    print("Fetching station metadata...")
    features = _paginate(
        f"{BASE_URL}/climate-stations/items",
        params={"COUNTRY": "CAN"},
    )
    df = _features_to_df(features)

    # Normalise lat/lon (raw values are stored as integers × 1e7)
    for col in ("LATITUDE", "LONGITUDE"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].abs().max() > 1000:
                df[col] = df[col] / 1e7

    # Use geometry-derived coords where available; fall back to property cols
    df["lat"] = df["latitude"].combine_first(df.get("LATITUDE"))
    df["lon"] = df["longitude"].combine_first(df.get("LONGITUDE"))
    df = df.drop(columns=["latitude", "longitude"], errors="ignore")

    out = RAW_DIR / "climate_stations.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df):,} stations → {out}")
    return df


def select_city_stations(stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each target city, find the nearest station that has daily data
    covering the full requested period.
    Returns a DataFrame with one row per city.
    """
    import numpy as np

    # Keep only stations with daily data that overlaps our range
    df = stations_df.copy()
    df["DLY_FIRST_DATE"] = pd.to_datetime(df["DLY_FIRST_DATE"], errors="coerce")
    df["DLY_LAST_DATE"]  = pd.to_datetime(df["DLY_LAST_DATE"],  errors="coerce")
    df = df[
        df["DLY_FIRST_DATE"].notna() &
        df["DLY_LAST_DATE"].notna()  &
        (df["DLY_FIRST_DATE"] <= pd.Timestamp(START_DATE)) &
        (df["DLY_LAST_DATE"]  >= pd.Timestamp(END_DATE))
    ].copy()

    results = []
    for city, (city_lat, city_lon) in CITIES.items():
        if df.empty:
            print(f"  WARNING: no candidate stations for {city}")
            continue
        dist = np.sqrt(
            (df["lat"] - city_lat) ** 2 +
            (df["lon"] - city_lon) ** 2
        )
        nearest = df.loc[dist.idxmin()].copy()
        nearest["city"] = city
        results.append(nearest)

    selected = pd.DataFrame(results)[
        ["city", "STATION_NAME", "CLIMATE_IDENTIFIER", "lat", "lon",
         "ELEVATION", "PROV_STATE_TERR_CODE", "DLY_FIRST_DATE", "DLY_LAST_DATE"]
    ].reset_index(drop=True)

    print("\nSelected stations:")
    print(selected[["city", "STATION_NAME", "CLIMATE_IDENTIFIER"]].to_string(index=False))
    return selected


def fetch_daily_climate(climate_id: str, city_name: str) -> pd.DataFrame:
    """
    Download daily climate observations for one station over START_DATE–END_DATE.
    Returns a DataFrame and saves to data/raw/climate_daily_<city>.csv.
    """
    out = RAW_DIR / f"climate_daily_{city_name.lower().replace(' ', '_')}.csv"
    if out.exists():
        print(f"  {city_name}: cache hit, skipping download")
        return pd.read_csv(out)

    features = _paginate(
        f"{BASE_URL}/climate-daily/items",
        params={
            "CLIMATE_IDENTIFIER": climate_id,
            "datetime": f"{START_DATE}/{END_DATE}",
        },
    )

    if not features:
        print(f"  WARNING: no data returned for {city_name} ({climate_id})")
        return pd.DataFrame()

    df = _features_to_df(features)
    df["city"] = city_name
    df.to_csv(out, index=False)
    print(f"  {city_name}: {len(df):,} daily records → {out.name}")
    return df


def fetch_normals(selected_stations: pd.DataFrame) -> pd.DataFrame:
    """
    Download 1981-2010 monthly climate normals for each target city.

    The selected daily-data stations are often too new to have 30-year normals.
    This function instead searches for the nearest station per city that DOES
    have normals, then fetches and pivots to wide format:
      one row per city × month, columns:
        city, MONTH, normals_station_id, normals_station_name,
        NORMAL_MEAN_TEMP, NORMAL_MAX_TEMP, NORMAL_MIN_TEMP, NORMAL_TOTAL_PRECIP

    Saves to data/raw/climate_normals.csv and returns the wide DataFrame.
    """
    import numpy as np

    out = RAW_DIR / "climate_normals.csv"
    if out.exists():
        print("climate_normals.csv: cache hit, skipping download")
        return pd.read_csv(out)

    ELEMENTS = {
        "Mean daily temperature deg C":     "NORMAL_MEAN_TEMP",
        "Mean daily max temperature deg C": "NORMAL_MAX_TEMP",
        "Mean daily min temperature deg C": "NORMAL_MIN_TEMP",
        "Total precipitation mm":           "NORMAL_TOTAL_PRECIP",
    }

    # Province for each city — used to narrow the normals search
    CITY_PROVINCE = dict(zip(
        selected_stations["city"],
        selected_stations["PROV_STATE_TERR_CODE"]
    ))
    CITY_COORDS = dict(zip(
        selected_stations["city"],
        zip(selected_stations["lat"], selected_stations["lon"])
    ))

    print("Fetching climate normals...")
    all_frames = []

    for city, province in tqdm(CITY_PROVINCE.items(), desc="Normals"):
        city_lat, city_lon = CITY_COORDS[city]

        # Pull all stations in this province that have CURRENT_FLAG normals
        # Use a large limit; paginate if needed
        features = _paginate(
            f"{BASE_URL}/climate-normals/items",
            params={
                "PROVINCE_CODE": province,
                "CURRENT_FLAG": "Y",
            },
            page_size=5000,
        )

        if not features:
            print(f"  WARNING: no normals found in province {province} for {city}")
            continue

        df = _features_to_df(features)

        # Find the station nearest to the city centre
        if "longitude" in df.columns and "latitude" in df.columns:
            lat_col, lon_col = "latitude", "longitude"
        else:
            lat_col, lon_col = "LATITUDE", "LONGITUDE"

        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        station_coords = df[[lat_col, lon_col, "CLIMATE_IDENTIFIER",
                              "STATION_NAME"]].drop_duplicates("CLIMATE_IDENTIFIER").dropna()
        if station_coords.empty:
            print(f"  WARNING: no coordinate data for normals in {province}")
            continue

        dist = np.sqrt(
            (station_coords[lat_col] - city_lat) ** 2 +
            (station_coords[lon_col] - city_lon) ** 2
        )
        best_id   = station_coords.loc[dist.idxmin(), "CLIMATE_IDENTIFIER"]
        best_name = station_coords.loc[dist.idxmin(), "STATION_NAME"]

        # Filter long table to the nearest station and the four variables
        subset = df[
            (df["CLIMATE_IDENTIFIER"] == best_id) &
            (df["E_NORMAL_ELEMENT_NAME"].isin(ELEMENTS))
        ].copy()

        if subset.empty:
            print(f"  WARNING: nearest normals station {best_name} lacks key variables")
            continue

        subset["variable"] = subset["E_NORMAL_ELEMENT_NAME"].map(ELEMENTS)
        subset["city"] = city

        wide = subset.pivot_table(
            index=["city", "MONTH"],
            columns="variable",
            values="VALUE",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        wide["normals_station_id"]   = best_id
        wide["normals_station_name"] = best_name
        all_frames.append(wide)
        print(f"  {city}: using normals from '{best_name}' ({best_id})")
        time.sleep(0.2)

    if not all_frames:
        raise RuntimeError("No normals data retrieved — check API connectivity.")

    normals = pd.concat(all_frames, ignore_index=True)

    for col in ELEMENTS.values():
        if col not in normals.columns:
            normals[col] = float("nan")

    normals = normals.sort_values(["city", "MONTH"]).reset_index(drop=True)
    normals.to_csv(out, index=False)
    print(f"\n  Saved {len(normals):,} city-month normals → {out}")
    return normals


def run_ingestion() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full ingestion pipeline:
      1. Fetch all station metadata
      2. Select best station per city
      3. Fetch daily climate for each selected station
      4. Concatenate into one combined daily file
      5. Fetch 1981-2010 monthly normals for each station

    Returns (stations_df, daily_df, normals_df).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 — stations
    stations_df = fetch_stations()

    # Step 2 — select one station per city
    selected = select_city_stations(stations_df)
    selected.to_csv(RAW_DIR / "selected_stations.csv", index=False)

    # Step 3 — daily data per city
    print(f"\nFetching daily climate data ({START_DATE} → {END_DATE})...")
    daily_frames = []
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Cities"):
        df = fetch_daily_climate(row["CLIMATE_IDENTIFIER"], row["city"])
        if not df.empty:
            # Attach lat/lon/elevation from station metadata
            df["lat"]       = row["lat"]
            df["lon"]       = row["lon"]
            df["elevation"] = row["ELEVATION"]
            df["province"]  = row["PROV_STATE_TERR_CODE"]
            daily_frames.append(df)
        time.sleep(0.3)

    # Step 4 — combine daily files
    if not daily_frames:
        raise RuntimeError("No daily data was retrieved — check API connectivity.")

    combined = pd.concat(daily_frames, ignore_index=True)
    out = RAW_DIR / "climate_daily_all.csv"
    combined.to_csv(out, index=False)
    print(f"\nCombined daily file: {len(combined):,} rows → {out}")

    # Step 5 — climate normals
    print()
    normals = fetch_normals(selected)

    return stations_df, combined, normals


if __name__ == "__main__":
    run_ingestion()
