"""
Test suite for pipeline/features.py

Four categories:
  1. Unit tests      — one transformation function at a time
  2. Leakage tests   — rolling ends yesterday, thresholds from training only,
                       model inputs for day t do not depend on day t's weather
  3. Invariant tests — domain-knowledge assertions
  4. Schema tests    — output columns and dtypes
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    add_calendar_features,
    add_derived_features,
    add_lag_features,
    add_rolling_features,
    add_relative_features,
    add_previous_day_features,
    add_target,
    build_features,
    compute_thresholds,
    drop_inserted,
    reindex_to_full_calendar,
)
from models.forecast import FEATURE_SETS

TRAIN_END_YEAR = 2019


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_daily(city="Ottawa", start="2010-01-01", n=30,
               mean_temp=5.0, max_temp=9.0, min_temp=1.0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "LOCAL_DATE":          dates,
        "LOCAL_YEAR":          dates.year,
        "LOCAL_MONTH":         dates.month,
        "LOCAL_DAY":           dates.day,
        "city":                city,
        "MEAN_TEMPERATURE":    mean_temp,
        "MAX_TEMPERATURE":     max_temp,
        "MIN_TEMPERATURE":     min_temp,
        "TOTAL_PRECIPITATION": 2.0,
        "SNOW_ON_GROUND":      0.0,
        "NORMAL_MEAN_TEMP":    3.0,
        "NORMAL_MAX_TEMP":     6.0,
        "NORMAL_MIN_TEMP":     0.0,
        "NORMAL_TOTAL_PRECIP": 60.0,
    })


def make_two_cities(n=30) -> pd.DataFrame:
    return pd.concat([
        make_daily("Ottawa",  start="2010-01-01", n=n),
        make_daily("Toronto", start="2010-01-01", n=n),
    ], ignore_index=True)


def make_thresholds(cities=("Ottawa",),
                    upper=20.0, lower=-20.0) -> pd.DataFrame:
    return pd.DataFrame({
        "city":        list(cities),
        "upper_fence": upper,
        "lower_fence": lower,
    })


# ---------------------------------------------------------------------------
# 1. Unit tests
# ---------------------------------------------------------------------------

class TestReindexToFullCalendar:

    def test_inserted_rows_flagged(self):
        df = make_daily(n=5, start="2010-01-01")
        # Remove Jan 3 to create a gap
        df = df[df["LOCAL_DATE"] != pd.Timestamp("2010-01-03")].reset_index(drop=True)
        expanded, _ = reindex_to_full_calendar(df)
        jan3 = expanded[expanded["LOCAL_DATE"] == pd.Timestamp("2010-01-03")]
        assert len(jan3) == 1
        assert jan3["is_inserted"].iloc[0] is True or jan3["is_inserted"].iloc[0] == True

    def test_real_rows_not_flagged(self):
        df = make_daily(n=5)
        expanded, _ = reindex_to_full_calendar(df)
        real = expanded[~expanded["is_inserted"]]
        assert real["is_inserted"].any() == False

    def test_city_filled_on_inserted_rows(self):
        df = make_daily(city="Ottawa", n=5, start="2010-01-01")
        df = df[df["LOCAL_DATE"] != pd.Timestamp("2010-01-03")].reset_index(drop=True)
        expanded, _ = reindex_to_full_calendar(df)
        jan3 = expanded[expanded["LOCAL_DATE"] == pd.Timestamp("2010-01-03")]
        assert jan3["city"].iloc[0] == "Ottawa"

    def test_row_count_preserved_after_drop(self):
        df = make_daily(n=10)
        df = df[df["LOCAL_DATE"] != pd.Timestamp("2010-01-05")].reset_index(drop=True)
        n_before = len(df)
        expanded, n_stored = reindex_to_full_calendar(df)
        restored = drop_inserted(expanded, n_stored)
        assert len(restored) == n_before

    def test_duplicate_city_date_raises(self):
        df = make_daily(n=5)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate"):
            reindex_to_full_calendar(df)

    def test_nan_city_raises(self):
        df = make_daily(n=5)
        df.loc[0, "city"] = None
        with pytest.raises(ValueError, match="city has NaNs"):
            reindex_to_full_calendar(df)


class TestLagFeatures:

    def test_lag1_correct_value(self):
        df = make_daily(n=5)
        df["MEAN_TEMPERATURE"] = [10.0, 11.0, 12.0, 13.0, 14.0]
        df = df.sort_values(["city", "LOCAL_DATE"]).reset_index(drop=True)
        result = add_lag_features(df)
        assert result["temp_lag_1"].iloc[1] == 10.0
        assert result["temp_lag_1"].iloc[2] == 11.0

    def test_lag1_first_row_is_nan(self):
        df = make_daily(n=5)
        result = add_lag_features(df)
        assert pd.isna(result["temp_lag_1"].iloc[0])

    def test_lag_does_not_cross_city_boundary(self):
        df = make_two_cities(n=5)
        df = df.sort_values(["city", "LOCAL_DATE"]).reset_index(drop=True)
        result = add_lag_features(df)
        # First row of Toronto should not contain Ottawa's last value
        toronto_first = result[result["city"] == "Toronto"].iloc[0]
        assert pd.isna(toronto_first["temp_lag_1"]), (
            "temp_lag_1 for Toronto's first day should be NaN, not Ottawa's last value"
        )

    def test_all_three_lags_present(self):
        df = make_daily(n=10)
        result = add_lag_features(df)
        for col in ["temp_lag_1", "temp_lag_2", "temp_lag_3"]:
            assert col in result.columns


class TestRollingFeatures:

    def test_roll5_correct_value(self):
        df = make_daily(n=10)
        df["MEAN_TEMPERATURE"] = [float(i) for i in range(10)]
        df = df.sort_values(["city", "LOCAL_DATE"]).reset_index(drop=True)
        result = add_rolling_features(df)
        # Row 5 (day 6): rolling mean of days 0-4 = (0+1+2+3+4)/5 = 2.0
        assert result["temp_roll_5"].iloc[5] == pytest.approx(2.0)

    def test_roll_does_not_cross_city_boundary(self):
        ottawa = make_daily("Ottawa",  n=5, mean_temp=100.0)
        toronto = make_daily("Toronto", n=5, mean_temp=0.0)
        df = pd.concat([ottawa, toronto], ignore_index=True)
        df = df.sort_values(["city", "LOCAL_DATE"]).reset_index(drop=True)
        result = add_rolling_features(df)
        toronto_rows = result[result["city"] == "Toronto"]
        assert (toronto_rows["temp_roll_5"].dropna() < 50).all(), (
            "Toronto rolling mean should not be influenced by Ottawa values"
        )

    def test_precip_roll7_never_negative(self):
        df = make_daily(n=20)
        result = add_rolling_features(df)
        assert (result["precip_roll_7"].dropna() >= 0).all()


class TestRelativeFeatures:

    def test_temp_relative_correct(self):
        df = make_daily(n=5, mean_temp=8.0)
        # NORMAL_MEAN_TEMP = 3.0 → temp_relative should be 5.0
        result = add_relative_features(df)
        assert np.allclose(result["temp_relative"].dropna(), 5.0)

    def test_missing_normal_mean_raises(self):
        df = make_daily(n=5)
        df["NORMAL_MEAN_TEMP"] = np.nan
        with pytest.raises(ValueError, match="NORMAL_MEAN_TEMP"):
            add_relative_features(df)

    def test_precip_relative_uses_days_in_month(self):
        # February has 28 days in 2010 — daily normal = 60/28
        df = make_daily(n=1, start="2010-02-01")
        df["TOTAL_PRECIPITATION"] = 0.0
        result = add_relative_features(df)
        expected = 0.0 - (60.0 / 28)
        assert result["precip_relative"].iloc[0] == pytest.approx(expected)


class TestDerivedFeatures:

    def test_temp_range_correct(self):
        df = make_daily(n=5, max_temp=10.0, min_temp=2.0)
        result = add_derived_features(df)
        assert np.allclose(result["temp_range"], 8.0)

    def test_negative_temp_range_raises(self):
        df = make_daily(n=5, max_temp=2.0, min_temp=10.0)
        with pytest.raises(ValueError, match="temp_range is negative"):
            add_derived_features(df)

    def test_snow_present_binary(self):
        df = make_daily(n=4)
        df["SNOW_ON_GROUND"] = [0.0, 5.0, 0.0, 10.0]
        result = add_derived_features(df)
        assert result["snow_present"].tolist() == [0.0, 1.0, 0.0, 1.0]


class TestCalendarFeatures:

    def test_day_of_year_range(self):
        df = make_daily(n=366, start="2012-01-01")  # 2012 is a leap year
        result = add_calendar_features(df)
        assert result["day_of_year"].between(1, 366).all()

    def test_month_range(self):
        df = make_daily(n=365, start="2010-01-01")
        result = add_calendar_features(df)
        assert result["month"].between(1, 12).all()

    def test_season_values(self):
        df = make_daily(n=365, start="2010-01-01")
        result = add_calendar_features(df)
        assert set(result["season"].unique()).issubset({0, 1, 2, 3})

    def test_season_winter_months(self):
        for month_start in ["2010-01-01", "2010-02-01", "2010-12-01"]:
            df = make_daily(n=1, start=month_start)
            result = add_calendar_features(df)
            assert result["season"].iloc[0] == 0, f"{month_start} should be winter (0)"

    def test_season_summer_months(self):
        for month_start in ["2010-06-01", "2010-07-01", "2010-08-01"]:
            df = make_daily(n=1, start=month_start)
            result = add_calendar_features(df)
            assert result["season"].iloc[0] == 2, f"{month_start} should be summer (2)"


class TestTargetVariable:

    def test_extreme_hot_labeled(self):
        df = make_daily(n=3, max_temp=35.0, min_temp=10.0)
        thresholds = make_thresholds(upper=30.0, lower=-20.0)
        result = add_target(df, thresholds, validate=False)
        assert (result["is_extreme_day"] == 1.0).all()

    def test_extreme_cold_labeled(self):
        df = make_daily(n=3, max_temp=5.0, min_temp=-25.0)
        thresholds = make_thresholds(upper=30.0, lower=-20.0)
        result = add_target(df, thresholds, validate=False)
        assert (result["is_extreme_day"] == 1.0).all()

    def test_normal_day_labeled_zero(self):
        df = make_daily(n=3, max_temp=15.0, min_temp=5.0)
        thresholds = make_thresholds(upper=30.0, lower=-20.0)
        result = add_target(df, thresholds, validate=False)
        assert (result["is_extreme_day"] == 0.0).all()

    def test_both_null_gives_nan_target(self):
        df = make_daily(n=3)
        df["MAX_TEMPERATURE"] = np.nan
        df["MIN_TEMPERATURE"] = np.nan
        thresholds = make_thresholds(upper=30.0, lower=-20.0)
        result = add_target(df, thresholds, validate=False)
        assert result["is_extreme_day"].isna().all()

    def test_one_null_other_extreme_labeled(self):
        df = make_daily(n=1, max_temp=np.nan, min_temp=-25.0)
        thresholds = make_thresholds(upper=30.0, lower=-20.0)
        result = add_target(df, thresholds, validate=False)
        assert result["is_extreme_day"].iloc[0] == 1.0

    def test_extreme_rate_out_of_bounds_raises(self):
        # All days extreme → rate = 100%, way above 10% ceiling
        df = make_daily(n=100, max_temp=50.0, min_temp=5.0)
        df["LOCAL_YEAR"] = 2010
        thresholds = make_thresholds(upper=30.0, lower=-20.0)
        with pytest.raises(ValueError, match="Extreme day rate"):
            add_target(df, thresholds, validate=True)


# ---------------------------------------------------------------------------
# 2. Data leakage tests
# ---------------------------------------------------------------------------

class TestNoDataLeakage:

    def test_rolling_ends_yesterday(self):
        df = make_daily(n=20)
        temps = [float(i) for i in range(20)]
        df["MEAN_TEMPERATURE"] = temps
        df = df.sort_values(["city", "LOCAL_DATE"]).reset_index(drop=True)
        result = add_rolling_features(df)

        # At row index 9 (day 10): mean of rows 4-8 = (4+5+6+7+8)/5 = 6.0
        # Including today would give rows 5-9 = 7.0; a centred mean rows 7-11 = 9.0
        assert result["temp_roll_5"].iloc[9] == pytest.approx(6.0), (
            "Rolling mean includes day t — this is data leakage"
        )

    def test_thresholds_unchanged_by_test_data(self):
        train_only = make_daily(n=365, start="2010-01-01", mean_temp=5.0,
                                max_temp=9.0, min_temp=1.0)
        train_only["LOCAL_YEAR"] = 2010

        with_test = pd.concat([
            train_only,
            make_daily(n=365, start="2021-01-01", mean_temp=50.0,
                       max_temp=60.0, min_temp=40.0),
        ], ignore_index=True)
        with_test.loc[with_test["LOCAL_DATE"].dt.year == 2021, "LOCAL_YEAR"] = 2021

        t1 = compute_thresholds(train_only)
        t2 = compute_thresholds(with_test)

        merged = t1.merge(t2, on="city", suffixes=("_train", "_full"))
        assert (merged["upper_fence_train"] == merged["upper_fence_full"]).all(), (
            "Thresholds changed when test-period data was added — training period filter leaked"
        )


class TestForecastInputsExcludeDayT:
    """Changing day t's observed weather must not change any model input for day t."""

    MODEL_FEATURES = sorted({f for feats in FEATURE_SETS.values() for f in feats})
    DAY_T = pd.Timestamp("2010-01-20")

    @staticmethod
    def make_series():
        df = make_daily(n=40, start="2010-01-01")
        rng = np.random.default_rng(0)
        df["MEAN_TEMPERATURE"] = rng.normal(-10, 5, len(df)).round(1)
        df["MAX_TEMPERATURE"]  = df["MEAN_TEMPERATURE"] + 4
        df["MIN_TEMPERATURE"]  = df["MEAN_TEMPERATURE"] - 4
        df["TOTAL_PRECIPITATION"] = rng.uniform(0, 5, len(df)).round(1)
        return df

    def perturb_day_t(self, df):
        df = df.copy()
        day = df["LOCAL_DATE"] == self.DAY_T
        df.loc[day, ["MAX_TEMPERATURE", "MEAN_TEMPERATURE", "MIN_TEMPERATURE"]] = [10.0, -15.0, -40.0]
        df.loc[day, "TOTAL_PRECIPITATION"] = 80.0
        df.loc[day, "SNOW_ON_GROUND"] = 50.0
        return df

    def build_pair(self):
        thresholds = make_thresholds(upper=5.0, lower=-30.0)
        base = build_features(self.make_series(), thresholds, validate=False)
        perturbed = build_features(self.perturb_day_t(self.make_series()), thresholds, validate=False)
        return base.set_index("LOCAL_DATE"), perturbed.set_index("LOCAL_DATE")

    def test_model_inputs_for_day_t_unchanged(self):
        base, perturbed = self.build_pair()
        # Sanity: the perturbation does change day t's label
        assert base.loc[self.DAY_T, "is_extreme_day"] != perturbed.loc[self.DAY_T, "is_extreme_day"]
        pd.testing.assert_series_equal(
            base.loc[self.DAY_T, self.MODEL_FEATURES],
            perturbed.loc[self.DAY_T, self.MODEL_FEATURES],
            check_names=False,
            obj="Model inputs for day t changed when day t's weather changed — target leakage",
        )

    def test_model_inputs_for_day_after_do_change(self):
        # Guards against the test above passing because the features ignore weather entirely
        base, perturbed = self.build_pair()
        day_after = self.DAY_T + pd.Timedelta(days=1)
        changed = [
            f for f in self.MODEL_FEATURES
            if not np.isclose(base.loc[day_after, f], perturbed.loc[day_after, f], equal_nan=True)
        ]
        for f in ["prev_temp_max_relative", "prev_temp_min_relative", "temp_lag_1",
                  "temp_roll_5"]:
            assert f in changed, f"{f} for day t+1 should reflect day t's weather"

    def test_previous_day_missing_gives_nan_not_older_value(self):
        df = make_daily(n=10)
        df["MAX_TEMPERATURE"] = [float(i) for i in range(10)]
        df["MIN_TEMPERATURE"] = df["MAX_TEMPERATURE"] - 5
        df = df[df["LOCAL_DATE"] != pd.Timestamp("2010-01-05")].reset_index(drop=True)
        result = build_features(df, make_thresholds(upper=50.0, lower=-50.0), validate=False)
        jan6 = result[result["LOCAL_DATE"] == pd.Timestamp("2010-01-06")].iloc[0]
        assert pd.isna(jan6["prev_temp_max_relative"]), (
            "Day after a gap took an older day's value as 'yesterday'"
        )
        assert pd.isna(jan6["temp_lag_1"])


# ---------------------------------------------------------------------------
# 3. Invariant tests
# ---------------------------------------------------------------------------

class TestInvariants:

    @pytest.fixture
    def full_features(self):
        df = make_daily(n=365, start="2010-01-01",
                        mean_temp=5.0, max_temp=9.0, min_temp=1.0)
        df["LOCAL_YEAR"] = 2010
        thresholds = make_thresholds(upper=30.0, lower=-20.0)
        return build_features(df, thresholds, validate=False)

    def test_temp_range_never_negative(self, full_features):
        assert (full_features["temp_range"].dropna() >= 0).all()

    def test_snow_present_only_binary(self, full_features):
        assert set(full_features["snow_present"].unique()).issubset({0.0, 1.0})

    def test_season_only_valid_values(self, full_features):
        assert set(full_features["season"].unique()).issubset({0, 1, 2, 3})

    def test_is_extreme_day_only_valid_values(self, full_features):
        valid = full_features["is_extreme_day"].dropna().unique()
        assert set(valid).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 4. Schema tests
# ---------------------------------------------------------------------------

class TestSchema:

    @pytest.fixture
    def full_features(self):
        df = make_daily(n=365, start="2010-01-01",
                        mean_temp=5.0, max_temp=9.0, min_temp=1.0)
        df["LOCAL_YEAR"] = 2010
        thresholds = make_thresholds(upper=30.0, lower=-20.0)
        return build_features(df, thresholds, validate=False)

    def test_all_feature_columns_present(self, full_features):
        required = [
            "temp_relative", "temp_max_relative", "temp_min_relative", "precip_relative",
            "temp_lag_1", "temp_lag_2", "temp_lag_3",
            "temp_roll_5", "temp_roll_7", "precip_roll_7",
            "temp_range", "snow_present",
            "prev_temp_relative", "prev_temp_max_relative", "prev_temp_min_relative",
            "prev_precip_relative", "prev_temp_range", "prev_snow_present",
            "prev_is_extreme_day",
            "day_of_year", "month", "season", "rainy_season",
            "is_extreme_day",
        ]
        missing = [c for c in required if c not in full_features.columns]
        assert missing == [], f"Missing feature columns: {missing}"

    def test_local_date_is_datetime(self, full_features):
        assert pd.api.types.is_datetime64_any_dtype(full_features["LOCAL_DATE"])

    def test_feature_columns_are_numeric(self, full_features):
        for col in ["prev_temp_relative", "temp_lag_1", "temp_roll_5", "prev_temp_range"]:
            assert pd.api.types.is_numeric_dtype(full_features[col]), (
                f"{col} should be numeric"
            )
