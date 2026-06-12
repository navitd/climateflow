"""
Test suite for pipeline/clean.py

Four categories:
  1. Unit tests  — one transformation function at a time
  2. Join tests  — merge_normals() integrity
  3. Schema      — output columns and dtypes
  4. Invariants  — domain-knowledge assertions on clean data
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.clean import clean_daily, clean_normals, merge_normals


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_daily(n=10, cities=None, start="2010-01-01") -> pd.DataFrame:
    """Minimal daily DataFrame with all columns clean.py expects."""
    if cities is None:
        cities = ["Ottawa"] * n
    dates = pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "LOCAL_DATE": dates,
        "LOCAL_MONTH": pd.date_range(start, periods=n, freq="D").month,
        "city": cities,
        "MEAN_TEMPERATURE": [5.0] * n,
        "MIN_TEMPERATURE": [1.0] * n,
        "MAX_TEMPERATURE": [9.0] * n,
        "TOTAL_PRECIPITATION": [2.0] * n,
        "SNOW_ON_GROUND": [np.nan] * n,
        "MEAN_TEMPERATURE_FLAG": ["M"] * n,
        "MAX_TEMPERATURE_FLAG": ["M"] * n,
        "latitude": [45.0] * n,
        "longitude": [-75.0] * n,
        "lat": [45.0] * n,
        "lon": [-75.0] * n,
    })


def make_normals(cities=None) -> pd.DataFrame:
    """Monthly normals for one or more cities, months 1–12."""
    if cities is None:
        cities = ["Ottawa"]
    rows = []
    for city in cities:
        for month in range(1, 13):
            rows.append({
                "city": city,
                "MONTH": month,
                "NORMAL_MEAN_TEMP": float(month),
                "NORMAL_MAX_TEMP": float(month) + 3,
                "NORMAL_MIN_TEMP": float(month) - 3,
                "NORMAL_TOTAL_PRECIP": 80.0,
                "normals_station_id": 999,
                "normals_station_name": "TEST STATION",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Unit tests
# ---------------------------------------------------------------------------

class TestCleanDaily:

    def test_date_parsed_to_datetime(self):
        df = clean_daily(make_daily())
        assert pd.api.types.is_datetime64_any_dtype(df["LOCAL_DATE"]), (
            "LOCAL_DATE should be datetime64 after cleaning"
        )

    def test_flag_columns_dropped(self):
        df = clean_daily(make_daily())
        flag_cols = [c for c in df.columns if c.endswith("_FLAG")]
        assert flag_cols == [], f"FLAG columns still present: {flag_cols}"

    def test_duplicate_coords_dropped(self):
        df = clean_daily(make_daily())
        assert "longitude" not in df.columns
        assert "latitude" not in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns

    def test_snow_nulls_imputed_to_zero(self):
        df = clean_daily(make_daily())
        assert df["SNOW_ON_GROUND"].isnull().sum() == 0
        assert (df["SNOW_ON_GROUND"] == 0.0).all()

    def test_snow_existing_values_unchanged(self):
        raw = make_daily(n=3)
        raw["SNOW_ON_GROUND"] = [np.nan, 5.0, np.nan]
        df = clean_daily(raw)
        assert df["SNOW_ON_GROUND"].tolist() == [0.0, 5.0, 0.0]

    def test_temperature_ffill_fills_short_gap(self):
        """A 3-day null gap should be forward-filled."""
        raw = make_daily(n=6)
        raw["MEAN_TEMPERATURE"] = [10.0, np.nan, np.nan, np.nan, 20.0, 20.0]
        df = clean_daily(raw)
        # Days 1-3 (index 1-3) should be filled with 10.0
        assert df["MEAN_TEMPERATURE"].iloc[1] == 10.0
        assert df["MEAN_TEMPERATURE"].iloc[2] == 10.0
        assert df["MEAN_TEMPERATURE"].iloc[3] == 10.0

    def test_temperature_ffill_respects_7day_limit(self):
        """A gap longer than 7 days should leave trailing values as NaN."""
        raw = make_daily(n=12)
        # Day 0 = 10.0, days 1-9 = NaN (9-day gap), day 10 onward = 20.0
        temps = [10.0] + [np.nan] * 9 + [20.0, 20.0]
        raw["MEAN_TEMPERATURE"] = temps
        df = clean_daily(raw)
        filled = df["MEAN_TEMPERATURE"].tolist()
        # Days 1–7 should be filled (10.0)
        assert all(v == 10.0 for v in filled[1:8]), f"Expected fill up to day 7: {filled}"
        # Days 8–9 should still be NaN
        assert np.isnan(filled[8]), f"Day 8 should remain NaN: {filled}"
        assert np.isnan(filled[9]), f"Day 9 should remain NaN: {filled}"

    def test_ffill_does_not_cross_city_boundary(self):
        """Forward-fill must not bleed from one city into another."""
        cities = ["Ottawa"] * 3 + ["Toronto"] * 3
        raw = make_daily(n=6, cities=cities)
        raw["MEAN_TEMPERATURE"] = [10.0, 10.0, 10.0, np.nan, np.nan, np.nan]
        df = clean_daily(raw)
        toronto_rows = df[df["city"] == "Toronto"]
        assert toronto_rows["MEAN_TEMPERATURE"].isnull().all(), (
            "Ottawa values should not fill Toronto gaps"
        )


class TestCleanNormals:

    def test_month13_dropped(self):
        normals = make_normals()
        # Add a month-13 annual summary row
        annual = normals.iloc[0].copy()
        annual["MONTH"] = 13
        normals = pd.concat([normals, pd.DataFrame([annual])], ignore_index=True)
        result = clean_normals(normals)
        assert 13 not in result["MONTH"].values

    def test_months_1_to_12_preserved(self):
        normals = make_normals()
        annual = normals.iloc[0].copy()
        annual["MONTH"] = 13
        normals = pd.concat([normals, pd.DataFrame([annual])], ignore_index=True)
        result = clean_normals(normals)
        assert set(result["MONTH"].unique()) == set(range(1, 13))


# ---------------------------------------------------------------------------
# 2. Join integrity tests
# ---------------------------------------------------------------------------

class TestMergeNormals:

    def test_merge_adds_normal_columns(self):
        daily = clean_daily(make_daily(n=12))
        daily["LOCAL_MONTH"] = range(1, 13)
        normals = make_normals()
        result = merge_normals(daily, normals)
        for col in ["NORMAL_MEAN_TEMP", "NORMAL_MAX_TEMP", "NORMAL_MIN_TEMP", "NORMAL_TOTAL_PRECIP"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_merge_preserves_row_count(self):
        n = 24
        daily = clean_daily(make_daily(n=n))
        daily["LOCAL_MONTH"] = list(range(1, 13)) * 2
        normals = make_normals()
        result = merge_normals(daily, normals)
        assert len(result) == n, f"Expected {n} rows, got {len(result)}"

    def test_merge_no_nulls_when_all_cities_matched(self):
        cities = ["Ottawa", "Toronto"]
        daily_rows = []
        for city in cities:
            rows = make_daily(n=12, cities=[city] * 12)
            rows["LOCAL_MONTH"] = range(1, 13)
            daily_rows.append(rows)
        daily = clean_daily(pd.concat(daily_rows, ignore_index=True))
        normals = make_normals(cities=cities)
        result = merge_normals(daily, normals)
        assert result["NORMAL_MEAN_TEMP"].isnull().sum() == 0

    def test_merge_month_column_not_duplicated(self):
        """The join key MONTH from normals should be dropped after merge."""
        daily = clean_daily(make_daily(n=12))
        daily["LOCAL_MONTH"] = range(1, 13)
        normals = make_normals()
        result = merge_normals(daily, normals)
        assert "MONTH" not in result.columns


# ---------------------------------------------------------------------------
# 3. Schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:

    @pytest.fixture
    def cleaned(self):
        daily = clean_daily(make_daily(n=12))
        daily["LOCAL_MONTH"] = range(1, 13)
        normals = make_normals()
        return merge_normals(daily, normals)

    def test_required_columns_present(self, cleaned):
        required = [
            "LOCAL_DATE", "LOCAL_MONTH", "city",
            "MEAN_TEMPERATURE", "MIN_TEMPERATURE", "MAX_TEMPERATURE",
            "TOTAL_PRECIPITATION", "SNOW_ON_GROUND",
            "NORMAL_MEAN_TEMP", "NORMAL_TOTAL_PRECIP",
            "lat", "lon",
        ]
        missing = [c for c in required if c not in cleaned.columns]
        assert missing == [], f"Missing required columns: {missing}"

    def test_local_date_is_datetime(self, cleaned):
        assert pd.api.types.is_datetime64_any_dtype(cleaned["LOCAL_DATE"])

    def test_temperature_columns_are_float(self, cleaned):
        for col in ["MEAN_TEMPERATURE", "MIN_TEMPERATURE", "MAX_TEMPERATURE"]:
            assert cleaned[col].dtype == np.float64, f"{col} should be float64"

    def test_normal_columns_are_float(self, cleaned):
        for col in ["NORMAL_MEAN_TEMP", "NORMAL_MAX_TEMP", "NORMAL_MIN_TEMP"]:
            assert cleaned[col].dtype == np.float64, f"{col} should be float64"


# ---------------------------------------------------------------------------
# 4. Data invariant tests
# ---------------------------------------------------------------------------

class TestDataInvariants:

    @pytest.fixture
    def cleaned(self):
        daily = clean_daily(make_daily(n=12))
        daily["LOCAL_MONTH"] = range(1, 13)
        normals = make_normals()
        return merge_normals(daily, normals)

    def test_snow_on_ground_has_no_nulls(self, cleaned):
        nulls = cleaned["SNOW_ON_GROUND"].isnull().sum()
        assert nulls == 0, f"SNOW_ON_GROUND has {nulls} unexpected nulls"

    def test_no_flag_columns_in_output(self, cleaned):
        flag_cols = [c for c in cleaned.columns if c.endswith("_FLAG")]
        assert flag_cols == [], f"FLAG columns should not reach the output: {flag_cols}"

    def test_normals_fully_joined(self, cleaned):
        null_rate = cleaned["NORMAL_MEAN_TEMP"].isnull().mean()
        assert null_rate == 0.0, (
            f"NORMAL_MEAN_TEMP null rate is {null_rate:.1%} — "
            "check that every city has a matching normals station"
        )

    def test_temperatures_in_plausible_range(self, cleaned):
        """Catches unit errors (e.g. Kelvin instead of Celsius)."""
        assert cleaned["MEAN_TEMPERATURE"].dropna().between(-60, 45).all(), (
            "MEAN_TEMPERATURE contains values outside the plausible Canadian range [-60, 45]°C"
        )

    def test_no_duplicate_coords_columns(self, cleaned):
        assert "longitude" not in cleaned.columns
        assert "latitude" not in cleaned.columns
