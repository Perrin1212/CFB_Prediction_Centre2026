from pathlib import Path

import json
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_PATH = Path(
    "data/features/matchup_features.csv"
)

MODEL_PATH = Path(
    "data/models/cfb_home_win_model.joblib"
)

FEATURE_COLUMNS_PATH = Path(
    "data/models/feature_columns.json"
)

TARGET = "home_win"

# Historical time split.
#
# Games in Weeks 1-12 are used for training.
# Games in Weeks 13+ are used for validation.
TRAIN_MAX_WEEK = 12


# ============================================================
# COLUMNS THAT MUST NOT BE MODEL FEATURES
# ============================================================

EXCLUDED_COLUMNS = {
    # Target
    "home_win",

    # Post-game information
    "home_points",
    "away_points",

    # Text identifiers
    "home_team",
    "away_team",

    # Database / API identifiers
    "game_id",
    "cfbd_id",

    # Time identifiers
    #
    # These remain in the dataset so that we can perform
    # chronological splits, but they are NOT given to
    # the model.
    "season",
    "week",
}


# ============================================================
# MODEL
# ============================================================

def build_model() -> HistGradientBoostingClassifier:

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    print()
    print("=" * 60)
    print("CFB MODEL TRAINING")
    print("=" * 60)

    print()
    print("Loading feature dataset...")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}"
        )

    df = pd.read_csv(
        FEATURE_PATH
    )

    print(
        f"✓ Rows loaded: {len(df):,}"
    )

    print(
        f"✓ Columns loaded: {len(df.columns):,}"
    )

    required_columns = {
        "season",
        "week",
        TARGET,
    }

    missing = required_columns - set(
        df.columns
    )

    if missing:
        raise ValueError(
            "Feature dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "season",
            "week",
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    feature_columns = []

    for column in df.columns:

        if column in EXCLUDED_COLUMNS:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        feature_columns.append(
            column
        )

    if not feature_columns:
        raise ValueError(
            "No numeric model features found."
        )

    return feature_columns


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:

    print()
    print("=" * 60)
    print("PREPARING MODEL DATA")
    print("=" * 60)

    print()
    print(
        f"✓ Model features: {len(feature_columns)}"
    )

    print()
    print(
        "Excluded columns:"
    )

    for column in sorted(
        EXCLUDED_COLUMNS
    ):
        if column in df.columns:
            print(
                f"  - {column}"
            )

    # --------------------------------------------------------
    # Check feature types
    # --------------------------------------------------------

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if non_numeric:
        raise ValueError(
            "Non-numeric model features found: "
            + ", ".join(non_numeric)
        )

    # --------------------------------------------------------
    # Time split
    # --------------------------------------------------------

    train_df = df[
        df["week"] <= TRAIN_MAX_WEEK
    ].copy()

    validation_df = df[
        df["week"] > TRAIN_MAX_WEEK
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if validation_df.empty:
        raise ValueError(
            "Validation dataset is empty."
        )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET
    ].astype(int)

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        TARGET
    ].astype(int)

    print()
    print(
        "Target distribution:"
    )

    print(
        df[TARGET]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print()
    print("=" * 60)
    print("TIME-BASED VALIDATION")
    print("=" * 60)

    print()
    print(
        f"Training weeks:      1-{TRAIN_MAX_WEEK}"
    )

    print(
        f"Validation weeks:    {TRAIN_MAX_WEEK + 1}+"
    )

    print()
    print(
        f"Training rows:       {len(train_df):,}"
    )

    print(
        f"Validation rows:     {len(validation_df):,}"
    )

    print()
    print(
        "Training target:"
    )

    print(
        y_train.value_counts(
            normalize=True
        ).sort_index()
    )

    print()
    print(
        "Validation target:"
    )

    print(
        y_validation.value_counts(
            normalize=True
        ).sort_index()
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


# ============================================================
# SAVE FEATURE METADATA
# ============================================================

def save_feature_metadata(
    feature_columns: list[str],
) -> None:

    FEATURE_COLUMNS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        FEATURE_COLUMNS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            feature_columns,
            file,
            indent=2,
        )

    print()
    print(
        "✓ Model feature metadata saved to:"
    )

    print(
        f"  {FEATURE_COLUMNS_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    df = load_data()

    feature_columns = get_feature_columns(
        df
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = prepare_data(
        df=df,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    model = build_model()

    print()
    print(
        "✓ HistGradientBoostingClassifier configured"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    print()
    print("Training...")

    model.fit(
        X_train,
        y_train,
    )

    print(
        "✓ Training complete"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    brier = brier_score_loss(
        y_validation,
        probabilities,
    )

    loss = log_loss(
        y_validation,
        probabilities,
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    print()
    print(
        f"ROC AUC:       {auc:.4f}"
    )

    print(
        f"Brier Score:   {brier:.4f}"
    )

    print(
        f"Log Loss:      {loss:.4f}"
    )

    print(
        f"Accuracy:      {accuracy:.4f}"
    )

    print()
    print(
        "Interpretation:"
    )

    if auc >= 0.70:
        print(
            "• Strong validation performance."
        )
    elif auc >= 0.65:
        print(
            "• Useful baseline, but substantial room for improvement."
        )
    elif auc >= 0.60:
        print(
            "• Weak predictive signal; feature improvement is required."
        )
    else:
        print(
            "• Model is currently close to non-predictive."
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print(
        "✓ Model saved to:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    save_feature_metadata(
        feature_columns
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        "The model now excludes:"
    )

    print(
        "  game_id, cfbd_id, season and week"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Run the walk-forward backtest again "
        "using the same clean feature set."
    )

    print()


if __name__ == "__main__":
    main()