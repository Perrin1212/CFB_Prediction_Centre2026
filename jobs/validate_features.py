from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_PATH = Path(
    "data/features/matchup_features.csv"
)


def main() -> None:

    print("🏈 CFB Prediction Centre")
    print("=" * 60)

    print()
    print("FEATURE DATA VALIDATION")
    print("=" * 60)

    if not FEATURE_PATH.exists():

        print(
            f"❌ Feature file not found:"
        )

        print(
            FEATURE_PATH
        )

        return

    df = pd.read_csv(
        FEATURE_PATH
    )

    print()
    print("1. BASIC DATASET")
    print("-" * 60)

    print(
        f"Rows:       {len(df):,}"
    )

    print(
        f"Columns:    {len(df.columns):,}"
    )

    print(
        f"Memory:     "
        f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
    )

    # --------------------------------------------------------------
    # TARGET
    # --------------------------------------------------------------

    print()
    print("2. TARGET")
    print("-" * 60)

    if "home_win" not in df.columns:

        print(
            "❌ home_win target is missing."
        )

        return

    print(
        df["home_win"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Home win rate: "
        f"{df['home_win'].mean():.3%}"
    )

    # --------------------------------------------------------------
    # SEASONS
    # --------------------------------------------------------------

    print()
    print("3. SEASONS")
    print("-" * 60)

    print(
        df.groupby(
            "season"
        ).size().to_string()
    )

    # --------------------------------------------------------------
    # WEEKS
    # --------------------------------------------------------------

    print()
    print("4. GAMES BY WEEK")
    print("-" * 60)

    weekly = (
        df.groupby(
            [
                "season",
                "week",
            ]
        )
        .size()
        .reset_index(
            name="games"
        )
    )

    print(
        weekly.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # CURRENT-SEASON INFORMATION
    # --------------------------------------------------------------

    print()
    print("5. CURRENT-SEASON INFORMATION")
    print("-" * 60)

    required_columns = [
        "home_current_games",
        "away_current_games",
        "home_current_weight",
        "away_current_weight",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print(
            "⚠ Missing expected columns:"
        )

        for column in missing:
            print(
                f"  - {column}"
            )

    else:

        df["min_current_games"] = df[
            [
                "home_current_games",
                "away_current_games",
            ]
        ].min(axis=1)

        print(
            "Games with NO current-season "
            "information:"
        )

        print(
            (
                df["min_current_games"] == 0
            ).sum()
        )

        print()

        print(
            "Games with BOTH teams having "
            "current-season information:"
        )

        print(
            (
                df["min_current_games"] > 0
            ).sum()
        )

        print()

        print(
            "Current-season weighting:"
        )

        print(
            df[
                [
                    "season",
                    "week",
                    "home_current_weight",
                    "away_current_weight",
                ]
            ]
            .groupby(
                [
                    "season",
                    "week",
                ]
            )
            .mean()
            .to_string()
        )

    # --------------------------------------------------------------
    # HISTORICAL INFORMATION
    # --------------------------------------------------------------

    print()
    print("6. HISTORICAL INFORMATION")
    print("-" * 60)

    historical_columns = [
        "home_historical_weight",
        "away_historical_weight",
    ]

    if all(
        column in df.columns
        for column in historical_columns
    ):

        print(
            "Average historical weight:"
        )

        print(
            df[
                historical_columns
            ].mean()
        )

    else:

        print(
            "⚠ Historical weight columns "
            "not found."
        )

    # --------------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------------

    print()
    print("7. MISSING VALUES")
    print("-" * 60)

    missing_values = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_values = missing_values[
        missing_values > 0
    ]

    if missing_values.empty:

        print(
            "✓ No missing values."
        )

    else:

        print(
            missing_values.to_string()
        )

    # --------------------------------------------------------------
    # ZERO-INFORMATION ROWS
    # --------------------------------------------------------------

    print()
    print("8. ZERO-INFORMATION ROWS")
    print("-" * 60)

    information_columns = [
        "home_games_played",
        "away_games_played",
    ]

    if all(
        column in df.columns
        for column in information_columns
    ):

        zero_information = df[
            (
                df[
                    "home_games_played"
                ] == 0
            )
            &
            (
                df[
                    "away_games_played"
                ] == 0
            )
        ]

        print(
            f"Both teams have zero prior "
            f"games: {len(zero_information):,}"
        )

        if len(zero_information) > 0:

            print()
            print(
                zero_information[
                    [
                        "season",
                        "week",
                        "home_team",
                        "away_team",
                        "home_win",
                    ]
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )

    # --------------------------------------------------------------
    # DATA LEAKAGE CHECK
    # --------------------------------------------------------------

    print()
    print("9. POTENTIAL LEAKAGE CHECK")
    print("-" * 60)

    leakage_columns = [
        "home_points",
        "away_points",
    ]

    print(
        "Target/result columns present:"
    )

    for column in leakage_columns:

        print(
            f"  {column}: "
            f"{column in df.columns}"
        )

    print()

    print(
        "These columns are expected to exist "
        "for evaluation, but MUST NOT be "
        "used as model features."
    )

    # --------------------------------------------------------------
    # FEATURE TYPES
    # --------------------------------------------------------------

    print()
    print("10. FEATURE TYPES")
    print("-" * 60)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    object_columns = df.select_dtypes(
        include="object"
    ).columns.tolist()

    print(
        f"Numeric columns: "
        f"{len(numeric_columns)}"
    )

    print(
        f"Text columns:    "
        f"{len(object_columns)}"
    )

    print()

    print(
        "Text columns:"
    )

    for column in object_columns:

        print(
            f"  - {column}"
        )

    # --------------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------------

    print()
    print("11. SAMPLE FEATURE ROWS")
    print("-" * 60)

    sample_columns = [
        "season",
        "week",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
        "home_win",
        "home_point_diff",
        "away_point_diff",
        "home_current_weight",
        "away_current_weight",
        "home_current_games",
        "away_current_games",
    ]

    sample_columns = [
        column
        for column in sample_columns
        if column in df.columns
    ]

    print(
        df[
            sample_columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 60)
    print("FEATURE VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()