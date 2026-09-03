from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from database.database import SessionLocal
from features.team_form import TeamFormEngine


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_PATH = Path("data/training_dataset.csv")

# These columns describe the game/result but MUST NOT be used
# as model input features.
NON_FEATURE_COLUMNS = {
    "game_id",
    "cfbd_id",
    "season",
    "week",
    "start_date",

    "home_team",
    "away_team",

    "home_points",
    "away_points",
    "point_margin",

    "target",
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any) -> float:
    """
    Convert a value safely to float.

    Missing/non-numeric values become 0.0.
    """

    if value is None:
        return 0.0

    try:
        value = float(value)

        if pd.isna(value):
            return 0.0

        return value

    except (TypeError, ValueError):
        return 0.0


def prefix_features(
    features: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    """
    Add home_/away_ prefix to team features.
    """

    return {
        f"{prefix}{key}": safe_float(value)
        for key, value in features.items()
    }


# ============================================================
# BUILD MATCHUP FEATURES
# ============================================================

def build_matchup_features(
    engine: TeamFormEngine,
    games_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build leakage-safe pre-game matchup features.

    IMPORTANT:

    Every snapshot is calculated using only information
    available BEFORE the target game.

    No final score is used to construct the features.
    """

    rows: list[dict[str, Any]] = []

    total_games = len(games_df)

    print()
    print("[4] Building pre-game matchup features...")
    print()

    for index, game in enumerate(
        games_df.itertuples(index=False),
        start=1,
    ):

        game_id = game.game_id
        season = int(game.season)

        week = game.week

        # --------------------------------------------------------
        # Week safety
        # --------------------------------------------------------

        if week is None:
            continue

        try:
            week = int(week)
        except (TypeError, ValueError):
            continue

        # --------------------------------------------------------
        # Team names
        # --------------------------------------------------------

        home_team = str(game.home_team)
        away_team = str(game.away_team)

        # --------------------------------------------------------
        # PRE-GAME TEAM SNAPSHOTS
        # --------------------------------------------------------

        home_snapshot = engine.get_team_snapshot(
            team=home_team,
            season=season,
            week=week,
            history=history_df,
        )

        away_snapshot = engine.get_team_snapshot(
            team=away_team,
            season=season,
            week=week,
            history=history_df,
        )

        # --------------------------------------------------------
        # HOME FEATURES
        # --------------------------------------------------------

        home_features = prefix_features(
            home_snapshot,
            "home_",
        )

        # --------------------------------------------------------
        # AWAY FEATURES
        # --------------------------------------------------------

        away_features = prefix_features(
            away_snapshot,
            "away_",
        )

        # --------------------------------------------------------
        # DIFFERENCE FEATURES
        # --------------------------------------------------------

        difference_features: dict[str, float] = {}

        shared_keys = (
            set(home_snapshot.keys())
            & set(away_snapshot.keys())
        )

        for key in shared_keys:

            # These are handled separately below.
            #
            # recent_form_diff is deliberately excluded because
            # it is explicitly calculated from recent_win_rate.
            #
            # This prevents the feature from being created twice.
            if key in {
                "historical_weight",
                "current_weight",
                "current_games",
                "games_played",
                "recent_form_diff",
            }:
                continue

            home_value = safe_float(
                home_snapshot.get(key)
            )

            away_value = safe_float(
                away_snapshot.get(key)
            )

            difference_features[
                f"{key}_diff"
            ] = (
                home_value
                -
                away_value
            )

        # --------------------------------------------------------
        # MATCHUP-LEVEL FEATURES
        # --------------------------------------------------------

        current_weight_diff = (
            safe_float(
                home_snapshot.get(
                    "current_weight",
                    0.0,
                )
            )
            -
            safe_float(
                away_snapshot.get(
                    "current_weight",
                    0.0,
                )
            )
        )

        # Recent form is explicitly defined as the difference
        # between the teams' recent win rates.
        #
        # This is intentionally created here rather than allowing
        # the generic difference loop to create another copy.

        recent_form_diff = (
            safe_float(
                home_snapshot.get(
                    "recent_win_rate",
                    0.0,
                )
            )
            -
            safe_float(
                away_snapshot.get(
                    "recent_win_rate",
                    0.0,
                )
            )
        )

        # --------------------------------------------------------
        # HOME-FIELD ADVANTAGE
        # --------------------------------------------------------

        neutral_site = bool(
            game.neutral_site
        )

        home_field_advantage = (
            0
            if neutral_site
            else 1
        )

        # --------------------------------------------------------
        # BUILD ROW
        # --------------------------------------------------------

        row: dict[str, Any] = {}

        row.update(home_features)
        row.update(away_features)
        row.update(difference_features)

        # Explicit matchup features.
        row["current_weight_diff"] = (
            current_weight_diff
        )

        row["recent_form_diff"] = (
            recent_form_diff
        )

        row["home_field_advantage"] = (
            home_field_advantage
        )

        # --------------------------------------------------------
        # GAME METADATA
        # --------------------------------------------------------

        row["game_id"] = game_id

        row["cfbd_id"] = game.cfbd_id

        row["season"] = season

        row["week"] = week

        row["start_date"] = game.start_date

        row["home_team"] = home_team

        row["away_team"] = away_team

        row["neutral_site"] = int(
            bool(game.neutral_site)
        )

        row["conference_game"] = int(
            bool(game.conference_game)
        )

        # --------------------------------------------------------
        # TARGET
        # --------------------------------------------------------

        home_points = game.home_points
        away_points = game.away_points

        if (
            home_points is None
            or away_points is None
        ):
            continue

        home_points = int(home_points)

        away_points = int(away_points)

        row["target"] = int(
            home_points > away_points
        )

        # --------------------------------------------------------
        # RESULT COLUMNS
        #
        # These are retained temporarily so we can verify the
        # dataset, but they are removed before model training.
        # --------------------------------------------------------

        row["home_points"] = home_points

        row["away_points"] = away_points

        row["point_margin"] = (
            home_points
            -
            away_points
        )

        rows.append(row)

        # --------------------------------------------------------
        # PROGRESS
        # --------------------------------------------------------

        if (
            index == 1
            or index % 500 == 1
            or index == total_games
        ):
            print(
                f"  Processed: "
                f"{index:,}/{total_games:,}"
            )

    return pd.DataFrame(rows)


# ============================================================
# REMOVE LEAKAGE / NON-FEATURE COLUMNS
# ============================================================

def clean_model_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure the final dataset contains only legitimate
    pre-game model features plus metadata/target.

    Final-score/result columns are deliberately excluded
    from the model feature set.
    """

    forbidden = {
        "home_points",
        "away_points",
        "point_margin",
    }

    df = df.drop(
        columns=list(
            forbidden & set(df.columns)
        ),
        errors="ignore",
    )

    return df


# ============================================================
# DUPLICATE FEATURE CHECK
# ============================================================

def check_duplicate_columns(
    df: pd.DataFrame,
) -> None:
    """
    Ensure duplicate column names have not been created.

    Pandas normally handles duplicate column names poorly for
    downstream model training, so fail immediately if any exist.
    """

    duplicate_columns = (
        df.columns[
            df.columns.duplicated()
        ]
        .tolist()
    )

    print()
    print("FEATURE COLUMN CHECK")

    if duplicate_columns:

        print(
            "  ✗ Duplicate columns found:"
        )

        for column in duplicate_columns:
            print(
                f"    - {column}"
            )

        raise ValueError(
            "Duplicate feature columns detected."
        )

    # Also check for pandas-style duplicate suffixes such as .1,
    # .2, .3 which can appear after CSV creation.

    suffixed_duplicates = [
        column
        for column in df.columns
        if column.endswith(".1")
        or column.endswith(".2")
        or column.endswith(".3")
    ]

    if suffixed_duplicates:

        print(
            "  ✗ Possible duplicated columns found:"
        )

        for column in suffixed_duplicates:
            print(
                f"    - {column}"
            )

        raise ValueError(
            "Possible duplicate feature columns "
            "detected."
        )

    print(
        "  ✓ No duplicate feature columns."
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_for_leakage(
    df: pd.DataFrame,
) -> None:
    """
    Verify that no final-result columns are present
    in the final dataset.
    """

    forbidden = {
        "home_points",
        "away_points",
        "point_margin",
    }

    leaked = sorted(
        forbidden & set(df.columns)
    )

    print()
    print("LEAKAGE CHECK")

    if leaked:

        print(
            "  ✗ WARNING: Result columns found:"
        )

        for column in leaked:
            print(
                f"    - {column}"
            )

        raise ValueError(
            "Final-score/result columns "
            "are present in the dataset."
        )

    print(
        "  ✓ No final-score/result columns "
        "present as features."
    )


# ============================================================
# DATA QUALITY
# ============================================================

def check_data_quality(
    df: pd.DataFrame,
) -> None:
    """
    Run basic dataset quality checks.
    """

    print()
    print("DATA QUALITY")

    # --------------------------------------------------------
    # Missing targets
    # --------------------------------------------------------

    missing_targets = int(
        df["target"].isna().sum()
    )

    print(
        f"  Missing targets: "
        f"{missing_targets:,}"
    )

    if missing_targets:
        raise ValueError(
            "Dataset contains missing targets."
        )

    # --------------------------------------------------------
    # Numeric missing values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    missing_numeric = int(
        df[numeric_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"  Missing numeric values: "
        f"{missing_numeric:,}"
    )

    if missing_numeric:
        raise ValueError(
            "Dataset contains missing numeric values."
        )

    # --------------------------------------------------------
    # Duplicate games
    # --------------------------------------------------------

    duplicates = int(
        df["game_id"]
        .duplicated()
        .sum()
    )

    print(
        f"  Duplicate games remaining: "
        f"{duplicates:,}"
    )

    if duplicates:
        raise ValueError(
            "Duplicate game IDs remain in dataset."
        )


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset() -> pd.DataFrame:
    """
    Main dataset-building pipeline.
    """

    print(
        "=" * 60
    )

    print(
        "CFB TRAINING DATASET BUILDER"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    session = SessionLocal()

    try:

        engine = TeamFormEngine(
            session=session
        )

        # ----------------------------------------------------
        # LOAD GAMES
        # ----------------------------------------------------

        print()
        print(
            "[1] Loading games..."
        )

        games_df = engine.load_games()

        print(
            f"✓ Games loaded: "
            f"{len(games_df):,}"
        )

        if games_df.empty:
            raise ValueError(
                "No games were loaded."
            )

        # ----------------------------------------------------
        # LOAD TEAM STATS
        # ----------------------------------------------------

        print()
        print(
            "[2] Loading team statistics..."
        )

        stats_df = engine.load_team_stats()

        print(
            f"✓ Team-stat records loaded: "
            f"{len(stats_df):,}"
        )

        if stats_df.empty:
            raise ValueError(
                "No team statistics were loaded."
            )

        # ----------------------------------------------------
        # BUILD TEAM HISTORY
        # ----------------------------------------------------

        print()
        print(
            "[3] Building team history..."
        )

        history_df = engine.build_team_history(
            stats_df=stats_df,
            games_df=games_df,
        )

        print(
            f"✓ Team history records: "
            f"{len(history_df):,}"
        )

        if history_df.empty:
            raise ValueError(
                "Team history is empty."
            )

        # ----------------------------------------------------
        # BUILD MATCHUP FEATURES
        # ----------------------------------------------------

        matchup_df = build_matchup_features(
            engine=engine,
            games_df=games_df,
            history_df=history_df,
        )

        # ----------------------------------------------------
        # CREATE TRAINING DATAFRAME
        # ----------------------------------------------------

        print()
        print(
            "[5] Creating training dataframe..."
        )

        if matchup_df.empty:
            raise ValueError(
                "No training rows were created."
            )

        df = matchup_df.copy()

        # ----------------------------------------------------
        # SORT CHRONOLOGICALLY
        # ----------------------------------------------------

        sort_columns = [
            column
            for column in [
                "season",
                "week",
                "start_date",
                "game_id",
            ]
            if column in df.columns
        ]

        if sort_columns:

            df = df.sort_values(
                sort_columns
            ).reset_index(
                drop=True
            )

        # ----------------------------------------------------
        # REMOVE RESULT LEAKAGE
        # ----------------------------------------------------

        df = clean_model_features(
            df
        )

        # ----------------------------------------------------
        # REMOVE DUPLICATE GAMES
        # ----------------------------------------------------

        before_duplicates = len(df)

        df = df.drop_duplicates(
            subset=["game_id"],
            keep="first",
        ).reset_index(
            drop=True
        )

        duplicates_removed = (
            before_duplicates
            -
            len(df)
        )

        # ----------------------------------------------------
        # DATA TYPES
        # ----------------------------------------------------

        for column in [
            "neutral_site",
            "conference_game",
            "home_field_advantage",
            "target",
        ]:

            if column in df.columns:

                df[column] = (
                    pd.to_numeric(
                        df[column],
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(int)
                )

        # ----------------------------------------------------
        # FILL NUMERIC MISSING VALUES
        # ----------------------------------------------------

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        df[numeric_columns] = (
            df[numeric_columns]
            .replace(
                [
                    float("inf"),
                    float("-inf"),
                ],
                0,
            )
            .fillna(0)
        )

        # ----------------------------------------------------
        # QUALITY CHECK
        # ----------------------------------------------------

        check_data_quality(
            df
        )

        print(
            f"  Duplicate games removed: "
            f"{duplicates_removed:,}"
        )

        # ----------------------------------------------------
        # DUPLICATE FEATURE CHECK
        # ----------------------------------------------------

        check_duplicate_columns(
            df
        )

        # ----------------------------------------------------
        # FINAL LEAKAGE CHECK
        # ----------------------------------------------------

        check_for_leakage(
            df
        )

        # ----------------------------------------------------
        # TARGET DISTRIBUTION
        # ----------------------------------------------------

        home_wins = int(
            (df["target"] == 1).sum()
        )

        away_wins = int(
            (df["target"] == 0).sum()
        )

        total = len(df)

        home_pct = (
            home_wins / total * 100
            if total
            else 0
        )

        away_pct = (
            away_wins / total * 100
            if total
            else 0
        )

        # ----------------------------------------------------
        # SEASON COUNTS
        # ----------------------------------------------------

        season_counts = (
            df["season"]
            .value_counts()
            .sort_index()
        )

        # ----------------------------------------------------
        # HISTORY COVERAGE
        # ----------------------------------------------------

        home_games_column = (
            "home_games_played"
        )

        away_games_column = (
            "away_games_played"
        )

        home_current_column = (
            "home_current_games"
        )

        away_current_column = (
            "away_current_games"
        )

        games_with_no_home_history = (
            int(
                (
                    df[home_games_column]
                    <= 0
                ).sum()
            )
            if home_games_column in df.columns
            else 0
        )

        games_with_no_away_history = (
            int(
                (
                    df[away_games_column]
                    <= 0
                ).sum()
            )
            if away_games_column in df.columns
            else 0
        )

        games_with_current_home_history = (
            int(
                (
                    df[home_current_column]
                    > 0
                ).sum()
            )
            if home_current_column in df.columns
            else 0
        )

        games_with_current_away_history = (
            int(
                (
                    df[away_current_column]
                    > 0
                ).sum()
            )
            if away_current_column in df.columns
            else 0
        )

        # ----------------------------------------------------
        # FEATURE COUNT
        # ----------------------------------------------------

        model_feature_columns = [
            column
            for column in df.columns
            if column not in NON_FEATURE_COLUMNS
        ]

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print()
        print(
            "=" * 60
        )

        print(
            "DATASET BUILD COMPLETE"
        )

        print(
            "=" * 60
        )

        print()
        print(
            f"✓ Training rows: "
            f"{len(df):,}"
        )

        print(
            f"✓ Features: "
            f"{len(model_feature_columns):,}"
        )

        print()
        print(
            "SEASONS"
        )

        for season, count in season_counts.items():

            print(
                f"  {int(season)}: "
                f"{int(count):,}"
            )

        print()
        print(
            "TARGET DISTRIBUTION"
        )

        print(
            f"  Home wins: "
            f"{home_wins:,} "
            f"({home_pct:.1f}%)"
        )

        print(
            f"  Away wins: "
            f"{away_wins:,} "
            f"({away_pct:.1f}%)"
        )

        print()
        print(
            "HISTORY COVERAGE"
        )

        print(
            f"  Games with no home history: "
            f"{games_with_no_home_history:,}"
        )

        print(
            f"  Games with no away history: "
            f"{games_with_no_away_history:,}"
        )

        print(
            f"  Games with current-season "
            f"home history: "
            f"{games_with_current_home_history:,}"
        )

        print(
            f"  Games with current-season "
            f"away history: "
            f"{games_with_current_away_history:,}"
        )

        print()
        print(
            "DATA QUALITY"
        )

        print(
            f"  Missing targets: "
            f"{int(df['target'].isna().sum()):,}"
        )

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        print(
            f"  Missing numeric values: "
            f"{int(df[numeric_columns].isna().sum().sum()):,}"
        )

        print(
            f"  Duplicate games removed: "
            f"{duplicates_removed:,}"
        )

        print()
        print(
            "LEAKAGE CHECK"
        )

        print(
            "  ✓ No final-score/result "
            "columns present as features."
        )

        print()
        print(
            f"✓ Saved to: "
            f"{OUTPUT_PATH}"
        )

        print()
        print(
            "=" * 60
        )

        return df

    finally:

        session.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_dataset()