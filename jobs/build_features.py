from __future__ import annotations

from pathlib import Path

import pandas as pd

from database.database import SessionLocal
from features.matchups import MatchupFeatureBuilder
from features.team_form import TeamFormEngine


OUTPUT_PATH = Path(
    "data/features/matchup_features.csv"
)


def main() -> None:

    print("🏈 CFB Prediction Centre")
    print("=" * 60)

    session = SessionLocal()

    try:

        print()
        print("Loading historical data...")

        engine = TeamFormEngine(
            session
        )

        games = engine.load_games()
        stats = engine.load_team_stats()

        print(
            f"✓ Games loaded: {len(games)}"
        )

        print(
            f"✓ Team-stat records loaded: "
            f"{len(stats)}"
        )

        if games.empty:
            print(
                "❌ No games available."
            )
            return

        if stats.empty:
            print(
                "❌ No team statistics available."
            )
            return

        print()
        print("Building team history...")

        history = engine.build_team_history(
            stats_df=stats,
            games_df=games,
        )

        print(
            f"✓ Team history rows: "
            f"{len(history)}"
        )

        matchup_builder = (
            MatchupFeatureBuilder()
        )

        feature_rows = []

        # --------------------------------------------------------------
        # PROCESS GAMES IN CHRONOLOGICAL ORDER
        # --------------------------------------------------------------

        games = games.sort_values(
            [
                "season",
                "week",
                "start_date",
            ]
        )

        total_games = len(games)

        for index, (_, game) in enumerate(
            games.iterrows(),
            start=1,
        ):

            season = int(
                game["season"]
            )

            week = int(
                game["week"]
                if pd.notna(game["week"])
                else 0
            )

            home_team = game[
                "home_team"
            ]

            away_team = game[
                "away_team"
            ]

            # ----------------------------------------------------------
            # PRE-GAME TEAM SNAPSHOTS
            # ----------------------------------------------------------

            home_features = (
                engine.get_team_snapshot(
                    team=home_team,
                    season=season,
                    week=week,
                    history=history,
                )
            )

            away_features = (
                engine.get_team_snapshot(
                    team=away_team,
                    season=season,
                    week=week,
                    history=history,
                )
            )

            # ----------------------------------------------------------
            # MATCHUP
            # ----------------------------------------------------------

            matchup = matchup_builder.build(
                home_team=home_team,
                away_team=away_team,
                home_features=home_features,
                away_features=away_features,
                neutral_site=bool(
                    game["neutral_site"]
                ),
            )

            # ----------------------------------------------------------
            # TARGET
            # ----------------------------------------------------------

            if (
                pd.isna(game["home_points"])
                or pd.isna(game["away_points"])
            ):
                continue

            matchup[
                "game_id"
            ] = int(
                game["game_id"]
            )

            matchup[
                "cfbd_id"
            ] = int(
                game["cfbd_id"]
            )

            matchup[
                "season"
            ] = season

            matchup[
                "week"
            ] = week

            matchup[
                "home_points"
            ] = int(
                game["home_points"]
            )

            matchup[
                "away_points"
            ] = int(
                game["away_points"]
            )

            matchup[
                "home_win"
            ] = int(
                game["home_points"]
                >
                game["away_points"]
            )

            feature_rows.append(
                matchup
            )

            if index % 250 == 0:

                print(
                    f"  Processed "
                    f"{index:,}/{total_games:,}"
                )

        # --------------------------------------------------------------
        # DATAFRAME
        # --------------------------------------------------------------

        if not feature_rows:

            print(
                "❌ No feature rows generated."
            )

            return

        feature_df = pd.DataFrame(
            feature_rows
        )

        # --------------------------------------------------------------
        # OUTPUT
        # --------------------------------------------------------------

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        feature_df.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print()
        print("=" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 60)

        print(
            f"✓ Feature rows: "
            f"{len(feature_df):,}"
        )

        print(
            f"✓ Features: "
            f"{len(feature_df.columns):,}"
        )

        print(
            f"✓ Saved to: "
            f"{OUTPUT_PATH}"
        )

        print()
        print("TARGET DISTRIBUTION")
        print("-" * 60)

        print(
            feature_df[
                "home_win"
            ].value_counts(
                normalize=True
            )
        )

        print()
        print("CURRENT-SEASON WEIGHT")
        print("-" * 60)

        if "home_current_weight" in feature_df:

            print(
                feature_df[
                    [
                        "season",
                        "week",
                        "home_current_weight",
                        "away_current_weight",
                    ]
                ].tail(10).to_string(
                    index=False
                )
            )

    finally:

        session.close()


if __name__ == "__main__":
    main()