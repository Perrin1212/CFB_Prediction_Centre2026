from pathlib import Path

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

FEATURE_PATH = Path(
    "data/features/matchup_features.csv"
)

OUTPUT_PATH = Path(
    "data/analysis/feature_importance.csv"
)

TARGET = "home_win"

# Same split used by train_model.py
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
    # These are still used to split the data chronologically,
    # but they should NOT be given to the model.
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
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    print()
    print("=" * 60)
    print("CFB FEATURE IMPORTANCE ANALYSIS")
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
        f"✓ Columns loaded: {len(df.columns)}"
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
            f"Missing required columns: {sorted(missing)}"
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
# MAIN
# ============================================================

def main() -> None:

    df = load_data()

    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    feature_columns = get_feature_columns(
        df
    )

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
    # Check for remaining non-numeric columns
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
    # Remove rows with missing model data
    # --------------------------------------------------------

    required_columns = (
        feature_columns
        + [TARGET]
    )

    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    removed = before - len(df)

    if removed:
        print()
        print(
            f"✓ Removed {removed:,} rows "
            f"with missing values"
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

    # --------------------------------------------------------
    # Time-based split
    # --------------------------------------------------------

    train_df = df[
        df["week"] <= TRAIN_MAX_WEEK
    ].copy()

    validation_df = df[
        df["week"] > TRAIN_MAX_WEEK
    ].copy()

    print()
    print("=" * 60)
    print("TIME-BASED SPLIT")
    print("=" * 60)

    print()
    print(
        f"Training weeks:      1-{TRAIN_MAX_WEEK}"
    )

    print(
        f"Validation weeks:    {TRAIN_MAX_WEEK + 1}+"
    )

    print(
        f"Training rows:       {len(train_df):,}"
    )

    print(
        f"Validation rows:     {len(validation_df):,}"
    )

    # --------------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    if y_train.nunique() < 2:
        raise ValueError(
            "Training target contains only one class."
        )

    if y_validation.nunique() < 2:
        raise ValueError(
            "Validation target contains only one class."
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
    # Baseline performance
    # --------------------------------------------------------

    validation_prob = model.predict_proba(
        X_validation
    )[:, 1]

    baseline_auc = roc_auc_score(
        y_validation,
        validation_prob,
    )

    print()
    print("=" * 60)
    print("BASELINE PERFORMANCE")
    print("=" * 60)

    print()
    print(
        f"Validation ROC AUC: "
        f"{baseline_auc:.4f}"
    )

    # --------------------------------------------------------
    # Permutation importance
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PERMUTATION IMPORTANCE")
    print("=" * 60)

    print()
    print(
        "Calculating importance on validation data..."
    )

    result = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": (
                result.importances_mean
            ),
            "importance_std": (
                result.importances_std
            ),
        }
    )

    # --------------------------------------------------------
    # Sort by actual importance
    # --------------------------------------------------------

    importance_df = (
        importance_df
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df["rank"] = (
        importance_df.index + 1
    )

    # --------------------------------------------------------
    # TOP POSITIVE FEATURES
    # --------------------------------------------------------

    print()
    print(
        "Top 30 features:"
    )

    print(
        "-" * 60
    )

    positive_features = (
        importance_df[
            importance_df[
                "importance_mean"
            ] > 0
        ]
        .head(30)
    )

    if positive_features.empty:

        print(
            "No positive-importance features."
        )

    else:

        print(
            positive_features[
                [
                    "rank",
                    "feature",
                    "importance_mean",
                    "importance_std",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # NEGATIVE FEATURES
    # --------------------------------------------------------

    print()
    print(
        "Bottom 15 features:"
    )

    print(
        "-" * 60
    )

    negative_features = (
        importance_df[
            importance_df[
                "importance_mean"
            ] < 0
        ]
        .sort_values(
            "importance_mean"
        )
        .head(15)
    )

    if negative_features.empty:

        print(
            "No negative-importance features."
        )

    else:

        print(
            negative_features[
                [
                    "rank",
                    "feature",
                    "importance_mean",
                    "importance_std",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("FEATURE IMPORTANCE COMPLETE")
    print("=" * 60)

    print()
    print(
        "✓ Results saved to:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "• Positive importance = "
        "feature improves validation AUC."
    )

    print(
        "• Near-zero importance = "
        "feature contributes little."
    )

    print(
        "• Negative importance = "
        "feature may be hurting performance."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "season, week, game_id and cfbd_id "
        "are used for data organisation/chronology "
        "but are NOT model features."
    )

    print()


if __name__ == "__main__":
    main()