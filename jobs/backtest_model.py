from pathlib import Path

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_PATH = Path(
    "data/features/matchup_features.csv"
)

OUTPUT_PATH = Path(
    "data/backtests/walk_forward_backtest.csv"
)

# Start testing once we have enough historical games
# for the model to have something meaningful to learn from.
FIRST_TEST_WEEK = 7

# Do not test beyond the final week available.
LAST_TEST_WEEK = 16


# ============================================================
# COLUMNS THAT MUST NOT BE MODEL FEATURES
# ============================================================

EXCLUDED_COLUMNS = {
    # Target
    "home_win",

    # Results only known after the game
    "home_points",
    "away_points",

    # Team identifiers / text
    "home_team",
    "away_team",

    # Database / chronology identifiers
    "game_id",
    "cfbd_id",
    "season",
    "week",
}


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title: str) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    print_header("CFB WALK-FORWARD BACKTEST")

    print()
    print("Loading feature dataset...")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}"
        )

    df = pd.read_csv(
        FEATURE_PATH
    )

    required_columns = {
        "season",
        "week",
        "home_win",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Feature dataset is missing required columns: "
            f"{sorted(missing)}"
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

    print(
        f"✓ Rows loaded: {len(df):,}"
    )

    print(
        f"✓ Columns loaded: {len(df.columns):,}"
    )

    print()
    print(
        "Seasons available:"
    )

    for season in sorted(
        df["season"].unique()
    ):

        season_rows = df[
            df["season"] == season
        ]

        print(
            f"  {season}: "
            f"{len(season_rows):,} games"
        )

    return df


# ============================================================
# FEATURE PREPARATION
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    feature_columns = []

    for column in df.columns:

        if column in EXCLUDED_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            feature_columns.append(
                column
            )

    if not feature_columns:
        raise ValueError(
            "No numeric model features found."
        )

    return feature_columns


# ============================================================
# MODEL
# ============================================================

def build_model() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# SINGLE WEEK
# ============================================================

def backtest_week(
    df: pd.DataFrame,
    feature_columns: list[str],
    season: int,
    test_week: int,
) -> dict | None:

    # --------------------------------------------------------
    # TRAINING DATA
    #
    # Only games BEFORE the test week.
    # --------------------------------------------------------

    train_df = df[
        (
            df["season"] < season
        )
        |
        (
            (df["season"] == season)
            &
            (df["week"] < test_week)
        )
    ].copy()

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    test_df = df[
        (df["season"] == season)
        &
        (df["week"] == test_week)
    ].copy()

    if train_df.empty:

        print(
            f"  ⚠ Week {test_week}: "
            f"no training data"
        )

        return None

    if test_df.empty:
        return None

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "home_win"
    ].astype(int)

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "home_win"
    ].astype(int)

    # --------------------------------------------------------
    # Need both target classes
    # --------------------------------------------------------

    if y_train.nunique() < 2:

        print(
            f"  ⚠ Week {test_week}: "
            f"training data contains only one class"
        )

        return None

    if y_test.nunique() < 2:

        print(
            f"  ⚠ Week {test_week}: "
            f"test data contains only one class"
        )

        return None

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    loss = log_loss(
        y_test,
        probabilities,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    return {
        "season": season,
        "week": test_week,
        "train_games": len(train_df),
        "test_games": len(test_df),
        "auc": auc,
        "brier": brier,
        "log_loss": loss,
        "accuracy": accuracy,
    }


# ============================================================
# WALK-FORWARD TEST
# ============================================================

def run_backtest(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:

    print_header("WALK-FORWARD TEST")

    seasons = sorted(
        df["season"].unique()
    )

    results = []

    for season in seasons:

        season_weeks = sorted(
            df.loc[
                df["season"] == season,
                "week",
            ]
            .dropna()
            .unique()
        )

        test_weeks = [
            int(week)
            for week in season_weeks
            if (
                FIRST_TEST_WEEK
                <= week
                <= LAST_TEST_WEEK
            )
        ]

        if not test_weeks:
            continue

        print()
        print(
            f"Season {season}"
        )

        print(
            "-" * 50
        )

        for week in test_weeks:

            result = backtest_week(
                df=df,
                feature_columns=feature_columns,
                season=season,
                test_week=week,
            )

            if result is None:
                continue

            results.append(
                result
            )

            print(
                f"  Week {week:2d} | "
                f"Games: {result['test_games']:3d} | "
                f"Train: {result['train_games']:4d} | "
                f"AUC: {result['auc']:.4f} | "
                f"Brier: {result['brier']:.4f} | "
                f"LogLoss: {result['log_loss']:.4f} | "
                f"Acc: {result['accuracy']:.3f}"
            )

    if not results:

        raise RuntimeError(
            "No backtest results were generated."
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results: pd.DataFrame,
) -> None:

    print_header("BACKTEST SUMMARY")

    print()

    print(
        f"Test weeks: "
        f"{len(results):,}"
    )

    print(
        f"Test games: "
        f"{results['test_games'].sum():,}"
    )

    print()

    print(
        "Overall metrics"
    )

    print(
        "-" * 50
    )

    print(
        f"AUC:       "
        f"{results['auc'].mean():.4f}"
    )

    print(
        f"Brier:     "
        f"{results['brier'].mean():.4f}"
    )

    print(
        f"Log Loss:  "
        f"{results['log_loss'].mean():.4f}"
    )

    print(
        f"Accuracy:  "
        f"{results['accuracy'].mean():.4f}"
    )

    # --------------------------------------------------------
    # Weighted overall accuracy
    # --------------------------------------------------------

    total_games = results[
        "test_games"
    ].sum()

    weighted_accuracy = (
        (
            results["accuracy"]
            * results["test_games"]
        ).sum()
        / total_games
    )

    print(
        f"Weighted Acc: "
        f"{weighted_accuracy:.4f}"
    )

    print()

    print(
        "Week-by-week results:"
    )

    print()

    display_df = results[
        [
            "season",
            "week",
            "test_games",
            "auc",
            "brier",
            "log_loss",
            "accuracy",
        ]
    ].copy()

    print(
        display_df.to_string(
            index=False,
            formatters={
                "auc": "{:.4f}".format,
                "brier": "{:.4f}".format,
                "log_loss": "{:.4f}".format,
                "accuracy": "{:.3f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # Best / worst
    # --------------------------------------------------------

    best_auc = results.loc[
        results["auc"].idxmax()
    ]

    worst_auc = results.loc[
        results["auc"].idxmin()
    ]

    print()

    print(
        f"Best AUC:   "
        f"Week {int(best_auc['week'])} "
        f"({best_auc['auc']:.4f})"
    )

    print(
        f"Worst AUC:  "
        f"Week {int(worst_auc['week'])} "
        f"({worst_auc['auc']:.4f})"
    )

    # --------------------------------------------------------
    # Early vs late
    # --------------------------------------------------------

    early = results[
        results["week"] <= 9
    ]

    late = results[
        results["week"] >= 10
    ]

    print()

    if not early.empty:

        print(
            f"Early season AUC "
            f"(Weeks {FIRST_TEST_WEEK}-9): "
            f"{early['auc'].mean():.4f}"
        )

    if not late.empty:

        print(
            f"Late season AUC "
            f"(Weeks 10+): "
            f"{late['auc'].mean():.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: pd.DataFrame,
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        "✓ Backtest results saved to:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    df = load_data()

    feature_columns = get_feature_columns(
        df
    )

    print()

    print(
        f"✓ Model features: "
        f"{len(feature_columns)}"
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

    results = run_backtest(
        df=df,
        feature_columns=feature_columns,
    )

    print_summary(
        results
    )

    save_results(
        results
    )

    print()

    print("=" * 70)

    print(
        "WALK-FORWARD BACKTEST COMPLETE"
    )

    print()

    print(
        "The backtest now uses the same clean "
        "feature set as the training model."
    )

    print()

    print(
        "No game_id, cfbd_id, season or week "
        "information is used as a model feature."
    )

    print()


if __name__ == "__main__":
    main()