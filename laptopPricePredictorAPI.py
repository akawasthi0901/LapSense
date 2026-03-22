"""
============================================================
  Laptop Price Prediction System — Complete ML Pipeline
  Author  : Akash Awasthi  (refactored & extended)
  Version : 2.0
============================================================

Pipeline stages
───────────────
  1. Data Loading
  2. Exploratory Data Analysis (EDA)
  3. Feature Engineering
  4. Outlier Detection & Removal  (IQR + Z-score)
  5. Train / Test Split
  6. Preprocessing (OneHotEncoding + Scaling)
  7. Multi-Model Training & Evaluation
  8. GridSearchCV Hyperparameter Tuning
  9. Final Model Serialisation
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ── Models ───────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ─── Logging Configuration ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = BASE_DIR / "data" / "laptop_price.csv"
MODEL_DIR  = BASE_DIR / "model"
PLOT_DIR   = BASE_DIR / "eda_plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. DATA LOADING                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_data(path: Path) -> pd.DataFrame:
    """
    Read the raw CSV.
    Uses latin1 encoding to handle special currency / brand characters.
    """
    log.info(f"Loading dataset  →  {path}")
    df = pd.read_csv(path, encoding="latin1")
    log.info(f"Raw shape: {df.shape}  |  Columns: {df.columns.tolist()}")
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. EXPLORATORY DATA ANALYSIS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_eda(df: pd.DataFrame) -> None:
    """
    Full EDA:
      • Shape, dtypes, missing values
      • Numeric distributions & box-plots
      • Categorical frequency bars
      • Correlation heat-map
      • Target (Price) distribution + log-transformed view
    All plots saved to  eda_plots/
    """
    log.info("=" * 60)
    log.info("EXPLORATORY DATA ANALYSIS")
    log.info("=" * 60)

    # ── 2.1  Basic Info ──────────────────────────────────────────────────────
    log.info(f"\nShape          : {df.shape}")
    log.info(f"\nColumn dtypes  :\n{df.dtypes.to_string()}")
    log.info(f"\nMissing values :\n{df.isnull().sum().to_string()}")
    log.info(f"\nDescriptive stats (numeric):\n{df.describe().to_string()}")

    # ── 2.2  Target Distribution ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["Price_euros"], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Price Distribution (€)")
    axes[0].set_xlabel("Price (€)")

    # Log-transform view — helps reveal right-skewed pricing
    sns.histplot(np.log1p(df["Price_euros"]), kde=True, ax=axes[1], color="coral")
    axes[1].set_title("Log(Price + 1) Distribution")
    axes[1].set_xlabel("log(Price + 1)")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_price_distribution.png", dpi=150)
    plt.close()
    log.info("Saved  →  01_price_distribution.png")

    # ── 2.3  Price vs Company (box-plot) ─────────────────────────────────────
    plt.figure(figsize=(14, 5))
    order = df.groupby("Company")["Price_euros"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Company", y="Price_euros", order=order, palette="Set2")
    plt.xticks(rotation=45, ha="right")
    plt.title("Price Distribution by Company")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_price_by_company.png", dpi=150)
    plt.close()
    log.info("Saved  →  02_price_by_company.png")

    # ── 2.4  Price vs TypeName ───────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="TypeName", y="Price_euros", palette="pastel")
    plt.title("Price Distribution by Laptop Type")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_price_by_type.png", dpi=150)
    plt.close()

    # ── 2.5  Categorical frequency bars ─────────────────────────────────────
    cat_cols = ["Company", "TypeName", "OpSys"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, cat_cols):
        vc = df[col].value_counts()
        sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="viridis")
        ax.set_title(f"Frequency — {col}")
        ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_categorical_frequencies.png", dpi=150)
    plt.close()
    log.info("Saved  →  04_categorical_frequencies.png")

    # ── 2.6  RAM, Inches vs Price (scatter) ─────────────────────────────────
    # Parse Ram temporarily for plotting
    ram_tmp = df["Ram"].str.replace("GB", "").astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(ram_tmp, df["Price_euros"], alpha=0.3, color="teal")
    axes[0].set_xlabel("RAM (GB)")
    axes[0].set_ylabel("Price (€)")
    axes[0].set_title("RAM vs Price")

    axes[1].scatter(df["Inches"], df["Price_euros"], alpha=0.3, color="purple")
    axes[1].set_xlabel("Screen Size (inches)")
    axes[1].set_ylabel("Price (€)")
    axes[1].set_title("Screen Size vs Price")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_numeric_vs_price.png", dpi=150)
    plt.close()
    log.info("Saved  →  05_numeric_vs_price.png")

    log.info("EDA complete.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. FEATURE ENGINEERING                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw columns into model-ready features:

    Column          Transformation
    ──────────────  ─────────────────────────────────────────────────
    Ram             Strip 'GB'  →  int
    Weight          Strip 'kg'  →  float
    ScreenResolution Extract Touchscreen (0/1), IPS (0/1), PPI (float)
    Cpu             Extract brand (Intel / AMD / Samsung)
    Memory          Binary flags: HDD(0/1), SSD(0/1)
    Gpu             Extract GPU brand (Intel / Nvidia / AMD)
    OpSys           Simplify to Windows / macOS / Linux / Other
    laptop_ID,      Dropped — no predictive value
    Product, Inches,
    ScreenResolution,
    Cpu, Memory,
    Gpu, OpSys
    """
    log.info("Applying feature engineering...")
    df = df.copy()

    # ── RAM: numeric ─────────────────────────────────────────────────────────
    df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).astype(int)

    # ── Weight: numeric ──────────────────────────────────────────────────────
    df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).astype(float)

    # ── Screen: Touchscreen flag ──────────────────────────────────────────────
    df["Touchscreen"] = df["ScreenResolution"].apply(
        lambda x: 1 if "Touchscreen" in str(x) else 0
    )

    # ── Screen: IPS panel flag ────────────────────────────────────────────────
    df["Ips"] = df["ScreenResolution"].apply(
        lambda x: 1 if "IPS" in str(x) else 0
    )

    # ── Screen: Pixels-per-Inch (PPI) ────────────────────────────────────────
    #   PPI = sqrt(res_x² + res_y²) / diagonal_inches
    resolution = df["ScreenResolution"].str.extract(r"(\d{3,4})x(\d{3,4})")
    df["ppi"] = (
        (resolution[0].astype(float) ** 2 + resolution[1].astype(float) ** 2) ** 0.5
        / df["Inches"]
    )

    # ── CPU: brand extraction ─────────────────────────────────────────────────
    #   Keep only first token (Intel / AMD / Samsung)
    df["Cpu brand"] = df["Cpu"].apply(lambda x: str(x).split()[0])

    # ── Memory: HDD flag ─────────────────────────────────────────────────────
    df["HDD"] = df["Memory"].apply(lambda x: 1 if "HDD" in str(x) else 0)

    # ── Memory: SSD flag ─────────────────────────────────────────────────────
    df["SSD"] = df["Memory"].apply(lambda x: 1 if "SSD" in str(x) else 0)

    # ── GPU: brand extraction ─────────────────────────────────────────────────
    df["Gpu brand"] = df["Gpu"].apply(lambda x: str(x).split()[0])

    # ── OS: simplify to 4 categories ─────────────────────────────────────────
    def simplify_os(x):
        x = str(x)
        if "Windows" in x:
            return "Windows"
        elif "macOS" in x or "Mac OS X" in x:
            return "macOS"
        elif "Linux" in x or "Chrome OS" in x:
            return "Linux"
        else:
            return "Other"

    df["os"] = df["OpSys"].apply(simplify_os)

    # ── Drop raw / high-cardinality columns ──────────────────────────────────
    drop_cols = [
        "laptop_ID", "Product", "Inches",
        "ScreenResolution", "Cpu",
        "Memory", "Gpu", "OpSys",
    ]
    df.drop(columns=drop_cols, inplace=True)

    log.info(f"After feature engineering shape: {df.shape}")
    log.info(f"Features used: {df.columns.tolist()}")
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. OUTLIER DETECTION & REMOVAL                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage outlier removal applied on the TARGET (Price_euros) and
    key numeric features (Ram, Weight, ppi).

    Stage 1 — IQR method
    ────────────────────
    Lower fence = Q1 − 1.5 × IQR
    Upper fence = Q3 + 1.5 × IQR
    Rows outside the fence are flagged.

    Stage 2 — Z-score method
    ─────────────────────────
    |Z| > 3  →  more than 3 std-devs from mean  →  flagged.

    A row is removed only if it is flagged by BOTH methods
    (conservative union intersection approach), preventing
    over-aggressive trimming on genuinely premium laptops.
    """
    log.info("=" * 60)
    log.info("OUTLIER DETECTION & REMOVAL")
    log.info("=" * 60)

    numeric_cols = ["Price_euros", "Ram", "Weight", "ppi"]
    df_clean = df.copy()
    initial_count = len(df_clean)

    # ── IQR Flagging ─────────────────────────────────────────────────────────
    iqr_outlier_mask = pd.Series(False, index=df_clean.index)
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        col_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        iqr_outlier_mask |= col_mask
        log.info(
            f"  IQR [{col}]  →  lower={lower:.2f}  upper={upper:.2f}  "
            f"flagged={col_mask.sum()}"
        )

    # ── Z-score Flagging ─────────────────────────────────────────────────────
    z_outlier_mask = pd.Series(False, index=df_clean.index)
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        col_mask = z_scores > 3
        z_outlier_mask |= col_mask
        log.info(
            f"  Z-score [{col}]  →  |z|>3  flagged={col_mask.sum()}"
        )

    # ── Conservative removal: flagged by BOTH methods ─────────────────────────
    combined_mask = iqr_outlier_mask & z_outlier_mask
    df_clean = df_clean[~combined_mask]

    removed = initial_count - len(df_clean)
    log.info(
        f"\nOutliers removed (IQR ∩ Z-score): {removed}  "
        f"({removed / initial_count * 100:.1f}%)"
    )
    log.info(f"Remaining rows: {len(df_clean)}")

    # ── Visual: before vs after price box-plot ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(df["Price_euros"])
    axes[0].set_title("Price — Before Outlier Removal")
    axes[0].set_ylabel("Price (€)")

    axes[1].boxplot(df_clean["Price_euros"])
    axes[1].set_title("Price — After Outlier Removal")
    axes[1].set_ylabel("Price (€)")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_outlier_boxplots.png", dpi=150)
    plt.close()
    log.info("Saved  →  06_outlier_boxplots.png")

    return df_clean


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5 + 6. TRAIN/TEST SPLIT & PRE-PROCESSING                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Sklearn ColumnTransformer:
      • Categorical  →  OneHotEncoder  (handle_unknown='ignore')
      • Numerical    →  StandardScaler (zero mean, unit variance)

    StandardScaler benefits tree models less but is critical for
    distance/linear models (SVR, KNN, Ridge, Lasso) in the comparison.
    """
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    log.info(f"Categorical features: {cat_cols}")
    log.info(f"Numerical  features : {num_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. MULTI-MODEL COMPARISON                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_models(X_train, X_test, y_train, y_test, preprocessor) -> dict:
    """
    Train and evaluate 10 regression models.

    Metrics reported per model:
      • MAE   — Mean Absolute Error (€)
      • RMSE  — Root Mean Squared Error (€)
      • R²    — Coefficient of Determination  [0, 1]
      • CV-R² — 5-fold cross-validated R² (mean ± std)

    Returns a dict of {model_name: fitted_pipeline}
    """
    log.info("=" * 60)
    log.info("MULTI-MODEL COMPARISON")
    log.info("=" * 60)

    candidate_models = {
        "Ridge Regression"         : Ridge(alpha=1.0),
        "Lasso Regression"         : Lasso(alpha=1.0, max_iter=5000),
        "ElasticNet"               : ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
        "Decision Tree"            : DecisionTreeRegressor(random_state=42),
        "Random Forest"            : RandomForestRegressor(n_estimators=100, random_state=42),
        "Extra Trees"              : ExtraTreesRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting"        : GradientBoostingRegressor(n_estimators=100, random_state=42),
        "AdaBoost"                 : AdaBoostRegressor(n_estimators=100, random_state=42),
        "K-Nearest Neighbours"     : KNeighborsRegressor(n_neighbors=5),
        "Support Vector Regressor" : SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
    }

    results = {}
    fitted_pipelines = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, regressor in candidate_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", regressor),
        ])

        # ── Fit ──────────────────────────────────────────────────────────────
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        # ── 5-fold CV on FULL training set ───────────────────────────────────
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1
        )

        results[name] = {
            "MAE"    : round(mae, 2),
            "RMSE"   : round(rmse, 2),
            "R2"     : round(r2, 4),
            "CV_R2"  : round(cv_scores.mean(), 4),
            "CV_std" : round(cv_scores.std(), 4),
        }
        fitted_pipelines[name] = pipe

        log.info(
            f"  {name:<30}  MAE={mae:>7.2f}  RMSE={rmse:>7.2f}  "
            f"R²={r2:.4f}  CV-R²={cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    log.info(f"\n{'='*60}\nModel Comparison Summary (sorted by R²):\n{results_df.to_string()}\n{'='*60}")

    # ── Bar chart comparison ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["R2", "RMSE", "MAE"]
    colors  = ["steelblue", "coral", "mediumseagreen"]

    for ax, metric, color in zip(axes, metrics, colors):
        sorted_df = results_df.sort_values(metric, ascending=(metric != "R2"))
        ax.barh(sorted_df.index, sorted_df[metric], color=color, edgecolor="white")
        ax.set_title(f"Model Comparison — {metric}")
        ax.set_xlabel(metric)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_model_comparison.png", dpi=150)
    plt.close()
    log.info("Saved  →  07_model_comparison.png")

    best_model_name = results_df["R2"].idxmax()
    log.info(f"\n★ Best model by R²: {best_model_name}")

    return fitted_pipelines, results_df, best_model_name


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. GRIDSEARCHCV HYPERPARAMETER TUNING                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tune_best_model(X_train, y_train, preprocessor) -> Pipeline:
    """
    GridSearchCV over RandomForest + GradientBoosting.

    RandomForest and GradientBoosting consistently win on tabular data;
    we search both and return whichever has the higher mean CV score.

    Param grids are kept modest to stay within reasonable run times.
    """
    log.info("=" * 60)
    log.info("HYPERPARAMETER TUNING  (GridSearchCV)")
    log.info("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── Random Forest grid ────────────────────────────────────────────────────
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42)),
    ])
    rf_grid = {
        "model__n_estimators" : [100, 200, 300],
        "model__max_depth"    : [None, 10, 15, 20],
        "model__min_samples_split" : [2, 5],
    }
    log.info("Searching RandomForest param space...")
    rf_search = GridSearchCV(
        rf_pipe, rf_grid, cv=kf, scoring="r2",
        n_jobs=-1, verbose=0, refit=True
    )
    rf_search.fit(X_train, y_train)
    log.info(f"  Best RF params : {rf_search.best_params_}")
    log.info(f"  Best RF CV-R²  : {rf_search.best_score_:.4f}")

    # ── Gradient Boosting grid ────────────────────────────────────────────────
    gb_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42)),
    ])
    gb_grid = {
        "model__n_estimators"  : [100, 200],
        "model__learning_rate" : [0.05, 0.1, 0.15],
        "model__max_depth"     : [3, 5, 7],
    }
    log.info("Searching GradientBoosting param space...")
    gb_search = GridSearchCV(
        gb_pipe, gb_grid, cv=kf, scoring="r2",
        n_jobs=-1, verbose=0, refit=True
    )
    gb_search.fit(X_train, y_train)
    log.info(f"  Best GB params : {gb_search.best_params_}")
    log.info(f"  Best GB CV-R²  : {gb_search.best_score_:.4f}")

    # ── Pick the winner ───────────────────────────────────────────────────────
    if rf_search.best_score_ >= gb_search.best_score_:
        log.info("→ RandomForest wins the tuning round.")
        best_tuned = rf_search.best_estimator_
        tuned_name = "RandomForest (tuned)"
    else:
        log.info("→ GradientBoosting wins the tuning round.")
        best_tuned = gb_search.best_estimator_
        tuned_name = "GradientBoosting (tuned)"

    log.info(f"Final tuned model: {tuned_name}")
    return best_tuned, tuned_name


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9. FINAL MODEL EVALUATION & SERIALISATION                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def final_evaluation(model: Pipeline, X_test, y_test, tuned_name: str) -> None:
    """
    Comprehensive report on the final tuned model:
      • MAE, RMSE, R²
      • Actual vs Predicted scatter plot
      • Residual distribution plot
      • Feature importances (top-20)
    """
    log.info("=" * 60)
    log.info("FINAL MODEL EVALUATION")
    log.info("=" * 60)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    # Adjusted R² penalises for extra features
    n, p = X_test.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    log.info(f"\n  Model   : {tuned_name}")
    log.info(f"  MAE     : {mae:.2f} €")
    log.info(f"  RMSE    : {rmse:.2f} €")
    log.info(f"  R²      : {r2:.4f}")
    log.info(f"  Adj-R²  : {adj_r2:.4f}")

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.4, color="steelblue", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Price (€)")
    axes[0].set_ylabel("Predicted Price (€)")
    axes[0].set_title(f"Actual vs Predicted — {tuned_name}")
    axes[0].legend()

    # ── Residuals ─────────────────────────────────────────────────────────────
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=40, color="coral", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual (€)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "08_final_model_evaluation.png", dpi=150)
    plt.close()
    log.info("Saved  →  08_final_model_evaluation.png")

    # ── Feature importances (tree-based models only) ──────────────────────────
    try:
        reg = model.named_steps["model"]
        importances = reg.feature_importances_
        ohe_cols = (
            model.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out()
        )
        num_cols_order = (
            model.named_steps["preprocessor"]
            .named_transformers_["num"]
            .named_steps["scaler"]
            .feature_names_in_
        )
        all_feature_names = list(ohe_cols) + list(num_cols_order)
        feat_imp = (
            pd.Series(importances, index=all_feature_names)
            .sort_values(ascending=False)
            .head(20)
        )

        plt.figure(figsize=(10, 6))
        feat_imp.sort_values().plot(kind="barh", color="steelblue")
        plt.title(f"Top-20 Feature Importances — {tuned_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "09_feature_importances.png", dpi=150)
        plt.close()
        log.info("Saved  →  09_feature_importances.png")
        log.info(f"\nTop-10 features:\n{feat_imp.head(10).to_string()}")
    except AttributeError:
        log.info("Feature importances not available for this model type.")

    # ── Correlation heat-map on engineered features ───────────────────────────
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Adj_R2": adj_r2}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN ORCHESTRATOR                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  LAPTOP PRICE PREDICTOR — FULL ML PIPELINE v2.0  ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df_raw = load_data(DATA_PATH)

    # ── 2. EDA (on raw data, before engineering) ──────────────────────────────
    run_eda(df_raw)

    # ── 3. Feature Engineering ────────────────────────────────────────────────
    df_eng = feature_engineering(df_raw)

    # ── 4. Outlier Removal ────────────────────────────────────────────────────
    df_clean = remove_outliers(df_eng)

    # ── 5. Split ──────────────────────────────────────────────────────────────
    X = df_clean.drop("Price_euros", axis=1)
    y = df_clean["Price_euros"]
    feature_order = X.columns.tolist()

    log.info(f"\nTrain/Test split: 80/20  |  Total rows: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info(f"  Training samples : {len(X_train)}")
    log.info(f"  Test     samples : {len(X_test)}")

    # ── 6. Preprocessor ───────────────────────────────────────────────────────
    preprocessor = build_preprocessor(X_train)

    # ── 7. Multi-model comparison ─────────────────────────────────────────────
    fitted_pipelines, results_df, best_baseline_name = evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # ── 8. Hyperparameter tuning ──────────────────────────────────────────────
    best_tuned_model, tuned_name = tune_best_model(X_train, y_train, preprocessor)

    # ── 9. Final evaluation ───────────────────────────────────────────────────
    metrics = final_evaluation(best_tuned_model, X_test, y_test, tuned_name)

    # ── Save model bundle ─────────────────────────────────────────────────────
    bundle = {
        "model"    : best_tuned_model,
        "features" : feature_order,
        "metrics"  : metrics,
        "model_name": tuned_name,
    }
    model_path = MODEL_DIR / "laptop_price_model.pkl"
    joblib.dump(bundle, model_path)
    log.info(f"\n✔  Model bundle saved  →  {model_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("FINAL RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"  Best Model  : {tuned_name}")
    log.info(f"  MAE         : {metrics['MAE']:.2f} €")
    log.info(f"  RMSE        : {metrics['RMSE']:.2f} €")
    log.info(f"  R²          : {metrics['R2']:.4f}")
    log.info(f"  Adjusted R² : {metrics['Adj_R2']:.4f}")
    log.info(f"\n  EDA plots saved to : {PLOT_DIR}")
    log.info(f"  Model saved to     : {model_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
