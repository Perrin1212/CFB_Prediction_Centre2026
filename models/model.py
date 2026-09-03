
"""
CFB Prediction Centre
Model Training Pipeline

Purpose:
    Train and evaluate CFB winner prediction models using the
    leakage-safe training dataset produced by features.dataset.

Dataset:
    data/training_dataset.csv

Evaluation design:
    2023-2024 -> training
    2025      -> completely unseen test season

The model predicts:
    target = 1 -> home team wins
    target = 0 -> away team wins
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "training_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "target"

# These columns are metadata/result information and must NOT
# be used as predictive model features.
#
# IMPORTANT:
# season and week are deliberately excluded as model inputs.
#
# We still retain them in the dataframe so we can perform
# chronological/time-based train/test splitting.
NON_FEATURE_COLUMNS = {
    # Identifiers
    "game_id",
    "cfbd_id",

    # Game metadata
    "season",
    "week",
    "start_date",

    # Teams
    "home_team",
    "away_team",

    # Final results
    "home_points",
    "away_points",
    "point_margin",

    # Target
    "target",
}


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the leakage-safe training dataset.
    """

    print_header(
        "[1] Loading training dataset"
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{DATA_PATH}\n\n"
            f"Run:\n"
            f"python -m features.dataset"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"✓ Dataset loaded: "
        f"{len(df):,} rows"
    )

    print(
        f"✓ Columns: "
        f"{len(df.columns)}"
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    """
    Prepare the numeric model features.

    Metadata such as season/week is retained in the source
    dataframe but excluded from predictive inputs.
    """

    print_header(
        "[2] Preparing features"
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"is missing from dataset."
        )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # Remove metadata / target / result columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in numeric_columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No usable numeric feature columns found."
        )

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET_COLUMN
    ].astype(int).copy()

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    forbidden_model_inputs = {
        "home_points",
        "away_points",
        "point_margin",
        "target",
        "season",
        "week",
    }

    leaked = sorted(
        forbidden_model_inputs
        & set(feature_columns)
    )

    if leaked:
        raise ValueError(
            "Forbidden columns detected in model "
            f"features: {leaked}"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        f"✓ Numeric model features: "
        f"{len(feature_columns)}"
    )

    print(
        f"✓ Target rows: "
        f"{len(y):,}"
    )

    print()
    print(
        "Features being used:"
    )

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    return (
        X,
        y,
        feature_columns,
    )


# ============================================================
# TIME-BASED SPLIT
# ============================================================

def split_by_season(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Perform a strict chronological split.

    Training:
        2023
        2024

    Test:
        2025

    2025 is never used during model fitting.
    """

    print_header(
        "[3] Creating time-based train/test split"
    )

    if "season" not in df.columns:
        raise ValueError(
            "Dataset does not contain "
            "a 'season' column."
        )

    train_mask = (
        df["season"] <= 2024
    )

    test_mask = (
        df["season"] == 2025
    )

    X_train = X.loc[
        train_mask
    ].copy()

    X_test = X.loc[
        test_mask
    ].copy()

    y_train = y.loc[
        train_mask
    ].copy()

    y_test = y.loc[
        test_mask
    ].copy()

    print(
        "Training seasons : 2023-2024"
    )

    print(
        "Test season      : 2025"
    )

    print()

    print(
        f"Training rows    : "
        f"{len(X_train):,}"
    )

    print(
        f"Test rows        : "
        f"{len(X_test):,}"
    )

    if len(X_train) == 0:
        raise ValueError(
            "Training set is empty."
        )

    if len(X_test) == 0:
        raise ValueError(
            "2025 test set is empty."
        )

    # --------------------------------------------------------
    # Target distributions
    # --------------------------------------------------------

    print()
    print(
        "Training target distribution:"
    )

    print(
        y_train
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print()
    print(
        "Test target distribution:"
    )

    print(
        y_test
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models() -> dict:
    """
    Create baseline models.

    Logistic Regression:
        Strong baseline for probability prediction.

    Random Forest:
        Non-linear tree ensemble.

    HistGradientBoosting:
        Gradient boosted tree model.
    """

    models = {}

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    models[
        "Logistic Regression"
    ] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models[
        "Random Forest"
    ] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # HistGradientBoosting
    # --------------------------------------------------------

    models[
        "HistGradientBoosting"
    ] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )

    return models


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Fit all candidate models.
    """

    print_header(
        "[4] Training models"
    )

    fitted_models = {}

    for name, model in models.items():

        print()
        print(
            f"Training: {name}"
        )

        model.fit(
            X_train,
            y_train,
        )

        fitted_models[name] = model

        print(
            "✓ Complete"
        )

    return fitted_models


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a fitted binary classification model.
    """

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    logloss = log_loss(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    print()
    print(name)
    print("-" * 60)

    print(
        f"Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{auc:.4f}"
    )

    print(
        f"Log Loss : "
        f"{logloss:.4f}"
    )

    print(
        f"Brier    : "
        f"{brier:.4f}"
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "roc_auc": auc,
        "log_loss": logloss,
        "brier": brier,
    }


# ============================================================
# LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

def show_logistic_coefficients(
    model,
    feature_columns: list[str],
    top_n: int = 20,
) -> None:
    """
    Display the strongest positive and negative Logistic
    Regression coefficients.

    Positive coefficient:
        Pushes probability toward home win.

    Negative coefficient:
        Pushes probability toward away win.
    """

    if (
        not hasattr(model, "named_steps")
        or "model" not in model.named_steps
    ):
        return

    estimator = (
        model.named_steps["model"]
    )

    if not isinstance(
        estimator,
        LogisticRegression,
    ):
        return

    coefficients = (
        estimator.coef_[0]
    )

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": coefficients,
            "absolute": np.abs(
                coefficients
            ),
        }
    )

    coefficient_df = (
        coefficient_df
        .sort_values(
            "absolute",
            ascending=False,
        )
    )

    print()
    print(
        f"Top {top_n} Logistic Regression "
        "features"
    )

    print("-" * 60)

    for _, row in (
        coefficient_df
        .head(top_n)
        .iterrows()
    ):

        direction = (
            "HOME"
            if row["coefficient"] > 0
            else "AWAY"
        )

        print(
            f"{row['feature']:<35} "
            f"{row['coefficient']:>10.5f} "
            f"({direction})"
        )


# ============================================================
# TREE FEATURE IMPORTANCE
# ============================================================

def show_tree_feature_importance(
    model,
    feature_columns: list[str],
    top_n: int = 20,
) -> None:
    """
    Display feature importance for tree models.
    """

    if (
        not hasattr(model, "named_steps")
        or "model" not in model.named_steps
    ):
        return

    estimator = (
        model.named_steps["model"]
    )

    if not hasattr(
        estimator,
        "feature_importances_",
    ):
        return

    importances = (
        estimator.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print()
    print(
        f"Top {top_n} tree features"
    )

    print("-" * 60)

    for _, row in (
        importance_df
        .head(top_n)
        .iterrows()
    ):

        print(
            f"{row['feature']:<35} "
            f"{row['importance']:.5f}"
        )


# ============================================================
# PROBABILITY SUMMARY
# ============================================================

def show_probability_summary(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Show how the model distributes its predictions.

    This helps us determine whether the model is too cautious,
    too aggressive, or actually separating strong and weak
    predictions.
    """

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    print()
    print(
        "2025 PROBABILITY SUMMARY"
    )

    print("-" * 60)

    print(
        f"Minimum probability : "
        f"{probabilities.min():.3f}"
    )

    print(
        f"25th percentile     : "
        f"{np.percentile(probabilities, 25):.3f}"
    )

    print(
        f"Median              : "
        f"{np.median(probabilities):.3f}"
    )

    print(
        f"75th percentile     : "
        f"{np.percentile(probabilities, 75):.3f}"
    )

    print(
        f"Maximum probability : "
        f"{probabilities.max():.3f}"
    )

    # --------------------------------------------------------
    # Confidence bands
    # --------------------------------------------------------

    bands = {
        ">= 0.90": probabilities >= 0.90,
        ">= 0.80": probabilities >= 0.80,
        ">= 0.70": probabilities >= 0.70,
        ">= 0.60": probabilities >= 0.60,
        "< 0.40": probabilities < 0.40,
        "< 0.30": probabilities < 0.30,
        "< 0.20": probabilities < 0.20,
        "< 0.10": probabilities < 0.10,
    }

    print()
    print(
        "CONFIDENCE BANDS"
    )

    for label, mask in bands.items():

        count = int(
            mask.sum()
        )

        if count == 0:
            print(
                f"  {label:<8} "
                f"0 games"
            )
            continue

        band_accuracy = float(
            (
                (
                    probabilities[mask]
                    >= 0.5
                ).astype(int)
                == y_test.to_numpy()[mask]
            ).mean()
        )

        print(
            f"  {label:<8} "
            f"{count:>5,} games"
        )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    model_name: str,
    feature_columns: list[str],
) -> None:
    """
    Save the fitted model and exact feature list.
    """

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_path = (
        MODEL_DIR
        / f"{safe_name}.joblib"
    )

    features_path = (
        MODEL_DIR
        / f"{safe_name}_features.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        feature_columns,
        features_path,
    )

    print()
    print(
        f"✓ Saved model: "
        f"{model_path}"
    )

    print(
        f"✓ Saved features: "
        f"{features_path}"
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

def save_test_predictions(
    df: pd.DataFrame,
    test_mask: pd.Series,
    model,
    model_name: str,
    feature_columns: list[str],
) -> None:
    """
    Save 2025 out-of-sample predictions.

    Uses the exact same feature list used during training.
    """

    predictions = (
        df.loc[
            test_mask
        ].copy()
    )

    X_predictions = (
        predictions[
            feature_columns
        ].copy()
    )

    probabilities = (
        model.predict_proba(
            X_predictions
        )[:, 1]
    )

    predictions[
        "predicted_home_win_probability"
    ] = probabilities

    predictions[
        "predicted_home_win"
    ] = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    predictions[
        "prediction_confidence"
    ] = np.maximum(
        probabilities,
        1.0 - probabilities,
    )

    # --------------------------------------------------------
    # Predicted side
    # --------------------------------------------------------

    predictions[
        "predicted_team"
    ] = np.where(
        probabilities >= 0.5,
        predictions["home_team"],
        predictions["away_team"],
    )

    # --------------------------------------------------------
    # Sort strongest predictions first
    # --------------------------------------------------------

    predictions = (
        predictions
        .sort_values(
            "prediction_confidence",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    output_path = (
        MODEL_DIR
        / f"{safe_name}_2025_predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print(
        f"✓ Saved test predictions: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "CFB PREDICTION CENTRE"
    )

    print(
        "MODEL TRAINING PIPELINE"
    )

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. Prepare features
    # --------------------------------------------------------

    (
        X,
        y,
        feature_columns,
    ) = prepare_features(
        df
    )

    # --------------------------------------------------------
    # 3. Split by season
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_by_season(
        df,
        X,
        y,
    )

    # --------------------------------------------------------
    # 4. Build models
    # --------------------------------------------------------

    models = build_models()

    # --------------------------------------------------------
    # 5. Train
    # --------------------------------------------------------

    fitted_models = train_models(
        models,
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    print_header(
        "[5] 2025 OUT-OF-SAMPLE RESULTS"
    )

    results = []

    for name, model in (
        fitted_models.items()
    ):

        result = evaluate_model(
            name,
            model,
            X_test,
            y_test,
        )

        results.append(
            result
        )

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "roc_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # 7. Comparison
    # --------------------------------------------------------

    print_header(
        "[6] MODEL COMPARISON"
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # 8. Best model
    # --------------------------------------------------------

    best_model_name = (
        results_df.iloc[0]["model"]
    )

    best_model = (
        fitted_models[
            best_model_name
        ]
    )

    print()
    print(
        f"BEST MODEL: "
        f"{best_model_name}"
    )

    # --------------------------------------------------------
    # 9. Feature analysis
    # --------------------------------------------------------

    print_header(
        "[7] FEATURE ANALYSIS"
    )

    if (
        best_model_name
        == "Logistic Regression"
    ):

        show_logistic_coefficients(
            best_model,
            feature_columns,
            top_n=25,
        )

    else:

        show_tree_feature_importance(
            best_model,
            feature_columns,
            top_n=25,
        )

    # --------------------------------------------------------
    # 10. Probability analysis
    # --------------------------------------------------------

    show_probability_summary(
        best_model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # 11. Save best model
    # --------------------------------------------------------

    print_header(
        "[8] SAVING BEST MODEL"
    )

    save_model(
        best_model,
        best_model_name,
        feature_columns,
    )

    # --------------------------------------------------------
    # 12. Save 2025 predictions
    # --------------------------------------------------------

    test_mask = (
        df["season"] == 2025
    )

    save_test_predictions(
        df=df,
        test_mask=test_mask,
        model=best_model,
        model_name=best_model_name,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print_header(
        "MODEL TRAINING COMPLETE"
    )

    print()
    print(
        f"Best model : "
        f"{best_model_name}"
    )

    print(
        "Test season: 2025"
    )

    print(
        f"Test games : "
        f"{len(X_test):,}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Use the 2025 out-of-sample "
        "feature analysis and probability"
    )

    print(
        "distribution to identify where "
        "the model is strong and weak."
    )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

