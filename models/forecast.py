"""
Step 4 — Model
Trains an XGBoost classifier to predict is_extreme_day.
Runs ablation test: with_season vs without_season feature set.
Saves metrics, plots, and the winning model.
"""
import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

DB_PATH    = Path("data/processed/climateflow.db")
OUT_DIR    = Path("outputs")
PLOTS_DIR  = OUT_DIR / "plots"

TRAIN_END_YEAR  = 2019
TEST_START_YEAR = 2020

CITY_ORDER = [
    "Vancouver", "Calgary", "Edmonton", "Saskatoon", "Winnipeg",
    "Toronto", "Ottawa", "Montreal", "Quebec",
]

BASE_FEATURES = [
    "temp_relative", "temp_max_relative", "temp_min_relative", "precip_relative",
    "temp_lag_1", "temp_lag_2", "temp_lag_3",
    "temp_roll_5", "temp_roll_7", "precip_roll_7",
    "temp_range", "snow_present",
    "day_of_year", "month",
]

FEATURE_SETS = {
    "with_season":             BASE_FEATURES + ["season"],
    "without_season":          BASE_FEATURES,
    "with_season_and_rainy":   BASE_FEATURES + ["season", "rainy_season"],
}

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM climate_features", conn)
    conn.close()
    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"])
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> tuple:
    # Drop rows with no target label
    n_before = len(df)
    df = df[df["is_extreme_day"].notna()].copy()
    print(f"Dropped {n_before - len(df)} rows with NaN target — {len(df):,} remaining")

    train = df[df["LOCAL_YEAR"] <= TRAIN_END_YEAR].copy()
    test  = df[df["LOCAL_YEAR"] >= TEST_START_YEAR].copy()

    # Temporal integrity check
    assert train["LOCAL_DATE"].max() < test["LOCAL_DATE"].min(), \
        "Train/test dates overlap — temporal split is corrupted"

    # scale_pos_weight from training data only
    n_neg = (train["is_extreme_day"] == 0).sum()
    n_pos = (train["is_extreme_day"] == 1).sum()
    scale_pos_weight = round(n_neg / n_pos, 2)

    print(f"\nTrain: {len(train):,} rows  ({int(n_pos)} extreme, {int(n_neg)} normal)")
    print(f"Test : {len(test):,} rows   ({int((test['is_extreme_day']==1).sum())} extreme, "
          f"{int((test['is_extreme_day']==0).sum())} normal)")
    print(f"scale_pos_weight = {scale_pos_weight}")

    return train, test, scale_pos_weight


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(X_train: pd.DataFrame, y_train: pd.Series,
          scale_pos_weight: float) -> XGBClassifier:
    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model: XGBClassifier, X: pd.DataFrame,
             y: pd.Series, cities: pd.Series) -> tuple:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    overall = {
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y, y_prob), 4),
    }

    per_city = []
    for city in CITY_ORDER:
        mask = cities == city
        if mask.sum() == 0:
            continue
        yc, pc, prc = y[mask], y_pred[mask], y_prob[mask]
        per_city.append({
            "city":      city,
            "n_total":   int(mask.sum()),
            "n_extreme": int(yc.sum()),
            "f1":        round(f1_score(yc, pc, zero_division=0), 4),
            "precision": round(precision_score(yc, pc, zero_division=0), 4),
            "recall":    round(recall_score(yc, pc, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(yc, prc), 4) if yc.nunique() > 1 else float("nan"),
        })

    return overall, pd.DataFrame(per_city)


# ---------------------------------------------------------------------------
# Ablation test
# ---------------------------------------------------------------------------

def run_ablation(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 scale_pos_weight: float) -> tuple:
    results   = {}
    models    = {}
    per_city_results = {}

    for name, features in FEATURE_SETS.items():
        print(f"\n  Training '{name}' ({len(features)} features)...")
        X_tr = train_df[features]
        y_tr = train_df["is_extreme_day"]
        X_te = test_df[features]
        y_te = test_df["is_extreme_day"]

        model = train(X_tr, y_tr, scale_pos_weight)
        overall, per_city = evaluate(model, X_te, y_te, test_df["city"])

        results[name]          = overall
        per_city_results[name] = per_city
        models[name]           = model
        print(f"    F1={overall['f1']}  Precision={overall['precision']}  "
              f"Recall={overall['recall']}  ROC-AUC={overall['roc_auc']}")

    # --- print comparison table ---
    set_names = list(FEATURE_SETS.keys())
    col_w = 22

    print("\n" + "=" * 75)
    print("  ABLATION TEST RESULTS")
    print("=" * 75)
    header = f"  {'Metric':<12}" + "".join(f"{n:>{col_w}}" for n in set_names)
    print(header)
    print(f"  {'-'*72}")

    for metric in ["f1", "precision", "recall", "roc_auc"]:
        row = f"  {metric:<12}"
        for n in set_names:
            row += f"{results[n][metric]:>{col_w}.4f}"
        print(row)

    print(f"  {'='*72}")

    # Winner = best F1
    winner = max(results, key=lambda n: results[n]["f1"])
    winner_f1 = results[winner]["f1"]
    baseline_f1 = results["with_season"]["f1"]
    rainy_f1 = results["with_season_and_rainy"]["f1"]

    print(f"\n  Best F1: {winner}  ({winner_f1:.4f})")

    # Season ablation conclusion
    wos_f1 = results["without_season"]["f1"]
    if wos_f1 > baseline_f1:
        print(f"  → Removing 'season' IMPROVES the model  "
              f"(+{wos_f1 - baseline_f1:.4f} F1)")
    else:
        print(f"  → Removing 'season' does NOT improve the model  "
              f"({wos_f1 - baseline_f1:+.4f} F1)")

    # Rainy season conclusion
    if rainy_f1 > baseline_f1:
        print(f"  → Adding 'rainy_season' IMPROVES the model  "
              f"(+{rainy_f1 - baseline_f1:.4f} F1 vs with_season)")
    else:
        print(f"  → Adding 'rainy_season' does NOT improve the model  "
              f"({rainy_f1 - baseline_f1:+.4f} F1 vs with_season)")

    print("=" * 75)

    season_removed_helps = wos_f1 > baseline_f1
    return winner, models[winner], results, per_city_results[winner], \
           FEATURE_SETS[winner], season_removed_helps


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_feature_importance(model: XGBClassifier,
                            features: list, winner: str) -> None:
    scores = model.get_booster().get_score(importance_type="gain")
    importance = pd.Series(scores).reindex(features).fillna(0).sort_values()
    importance = importance[importance > 0].tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Feature importance (gain)  —  {winner}", fontsize=12)
    ax.set_xlabel("Gain")
    ax.set_ylabel("")
    fig.tight_layout()

    path = PLOTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_city_predictions(model: XGBClassifier, test_df: pd.DataFrame,
                          features: list, winner: str) -> None:
    X_te   = test_df[features]
    y_pred = model.predict(X_te)
    y_true = test_df["is_extreme_day"].values

    cities = [c for c in CITY_ORDER if c in test_df["city"].values]
    actual, predicted = [], []
    for city in cities:
        mask = test_df["city"].values == city
        actual.append(y_true[mask].mean() * 100)
        predicted.append(y_pred[mask].mean() * 100)

    x = np.arange(len(cities))
    w = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, actual,    w, label="Actual",    color="steelblue",  alpha=0.85)
    ax.bar(x + w/2, predicted, w, label="Predicted", color="darkorange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Extreme days (%)")
    ax.set_title(f"Actual vs predicted extreme-day frequency  —  test period 2020–2023"
                 f"\n(model: {winner})", fontsize=11)
    ax.legend()
    fig.tight_layout()

    path = PLOTS_DIR / "city_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_ablation_comparison(results: dict) -> None:
    metrics = ["f1", "precision", "recall", "roc_auc"]
    labels  = ["F1", "Precision", "Recall", "ROC-AUC"]
    ws  = [results["with_season"][m]    for m in metrics]
    wos = [results["without_season"][m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, ws,  w, label="with season",    color="steelblue",  alpha=0.85)
    ax.bar(x + w/2, wos, w, label="without season", color="darkorange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: with season vs without season", fontsize=12)
    ax.legend()
    fig.tight_layout()

    path = PLOTS_DIR / "ablation_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_metrics(results: dict, per_city: pd.DataFrame,
                 winner: str, model: XGBClassifier) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ablation results — always saved
    ablation = pd.DataFrame(results).T.reset_index().rename(columns={"index": "feature_set"})
    ablation["winner"] = ablation["feature_set"] == winner
    ablation.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    print(f"  Saved {OUT_DIR / 'ablation_results.csv'}")

    # per-city metrics for winning model
    per_city.to_csv(OUT_DIR / "metrics_by_city.csv", index=False)
    print(f"  Saved {OUT_DIR / 'metrics_by_city.csv'}")

    # trained model
    model_path = OUT_DIR / "model.joblib"
    joblib.dump({"model": model, "features": FEATURE_SETS[winner], "winner": winner}, model_path)
    print(f"  Saved {model_path}  (feature set: {winner})")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run() -> None:
    print("=== Step 4 — Model ===\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load()
    train_df, test_df, scale_pos_weight = prepare(df)

    print("\nRunning ablation test...")
    winner, best_model, all_results, per_city, winning_features, season_removed_helps = \
        run_ablation(train_df, test_df, scale_pos_weight)

    print("\nGenerating plots...")
    plot_feature_importance(best_model, winning_features, winner)
    plot_city_predictions(best_model, test_df, winning_features, winner)
    if season_removed_helps:
        plot_ablation_comparison(all_results)

    print("\nSaving metrics and model...")
    save_metrics(all_results, per_city, winner, best_model)

    print("\n=== Step 4 complete ===")


if __name__ == "__main__":
    run()
