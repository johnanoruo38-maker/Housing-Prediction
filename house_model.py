import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join("dataset", "house.csv")
MODEL_OUTPUT  = os.path.join("backend", "model.joblib")
CHART_OUTPUT  = os.path.join("backend", "price_vs_size.png")
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
TARGET_COL    = "price"
CAT_FEATURES  = ["location"]
NUM_FEATURES  = ["size", "rooms"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    log.info("Loading dataset from '%s'", path)
    df = pd.read_csv(path)
    required = set(CAT_FEATURES + NUM_FEATURES + [TARGET_COL])
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    log.info("Loaded %d rows × %d columns", *df.shape)
    return df    


def split_features_target(df: pd.DataFrame):
    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df[TARGET_COL]
    return X, y


# ── Preprocessing Pipeline ────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline,       NUM_FEATURES),
        ("cat", categorical_pipeline,   CAT_FEATURES),
    ])


# ── Model Definitions ─────────────────────────────────────────────────────────
def get_models(preprocessor: ColumnTransformer) -> dict:
    return {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model",         LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model",         RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2":   r2_score(y_test, y_pred),
    }


def print_results(name: str, metrics: dict) -> None:
    log.info(
        "%-22s  MAE=$%s  RMSE=$%s  R²=%.4f",
        name,
        f"{metrics['MAE']:,.0f}",
        f"{metrics['RMSE']:,.0f}",
        metrics["R2"],
    )


# ── Visualization ─────────────────────────────────────────────────────────────
def save_price_vs_size_chart(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"urban": "#2563eb", "suburban": "#16a34a",
              "rural": "#d97706", "coastal": "#db2777"}

    for loc, group in df.groupby("location"):
        ax.scatter(
            group["size"], group["price"] / 1_000,
            label=loc.capitalize(),
            color=colors.get(loc, "#6b7280"),
            alpha=0.75, s=45, edgecolors="white", linewidths=0.4,
        )

    ax.set_xlabel("Size (sq ft)", fontsize=11)
    ax.set_ylabel("Price ($K)", fontsize=11)
    ax.set_title("House Price vs. Size by Location", fontsize=13, fontweight="bold")
    ax.legend(title="Location", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Chart saved → '%s'", path)


# ── Model Persistence ─────────────────────────────────────────────────────────
def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log.info("Model saved → '%s'", path)


# ── Main Orchestration ────────────────────────────────────────────────────────
def train() -> None:
    log.info("═" * 55)
    log.info("  House Price Prediction — Training Pipeline")
    log.info("═" * 55)

    df = load_data(DATASET_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info("Train/test split: %d / %d samples", len(X_train), len(X_test))

    preprocessor = build_preprocessor()
    models       = get_models(preprocessor)
    results      = {}

    log.info("─" * 55)
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = {"pipeline": pipeline, "metrics": metrics}
        print_results(name, metrics)
    log.info("─" * 55)

    # Select best model by R² score
    best_name = max(results, key=lambda n: results[n]["metrics"]["R2"])
    best_pipe  = results[best_name]["pipeline"]
    log.info("✓ Best model: %s (R²=%.4f)",
             best_name, results[best_name]["metrics"]["R2"])

    # Persist model and chart
    save_model(best_pipe, MODEL_OUTPUT)
    save_price_vs_size_chart(df, CHART_OUTPUT)

    # Store location categories for API reference
    encoder = best_pipe.named_steps["preprocessor"] \
                       .named_transformers_["cat"] \
                       .named_steps["encoder"]
    known_locations = encoder.categories_[0].tolist()

    metadata = {
        "model_name":        best_name,
        "known_locations":   known_locations,
        "cat_features":      CAT_FEATURES,
        "num_features":      NUM_FEATURES,
        "metrics":           results[best_name]["metrics"],
    }
    meta_path = os.path.join("backend", "model_meta.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved → '%s'", meta_path)
    log.info("═" * 55)
    log.info("Training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        log.error("Training failed: %s", exc)
        sys.exit(1)