from __future__ import annotations

from database.database import SessionLocal
from features.team_form import TeamFormEngine


def main() -> None:
    print("=" * 60)
    print("TEAM FORM FEATURE TEST")
    print("=" * 60)

    session = SessionLocal()

    try:
        engine = TeamFormEngine(session)

        print("\n[1] Loading games...")
        games_df = engine.load_games()

        print(f"✓ Games loaded: {len(games_df):,}")

        if games_df.empty:
            print("⚠ No completed games found.")
            return

        print("\n[2] Loading team statistics...")
        stats_df = engine.load_team_stats()

        print(
            f"✓ Team-stat records loaded: "
            f"{len(stats_df):,}"
        )

        if stats_df.empty:
            print("⚠ No team statistics found.")
            return

        print("\n[3] Building team history...")

        history_df = engine.build_team_history(
            stats_df=stats_df,
            games_df=games_df,
        )

        print(
            f"✓ Team history records: "
            f"{len(history_df):,}"
        )

        if history_df.empty:
            print("⚠ Team history is empty.")
            return

        print("\n[4] Testing 2026 Week 1 snapshots...")

        test_teams = [
            "Ohio State",
            "Michigan",
            "Alabama",
            "Georgia",
            "Oregon",
        ]

        for team in test_teams:

            snapshot = engine.get_team_snapshot(
                team=team,
                season=2026,
                week=1,
                history=history_df,
            )

            print("\n" + "-" * 60)
            print(f"TEAM: {team}")
            print("-" * 60)

            print(
                f"Games played:        "
                f"{snapshot.get('games_played', 0):.1f}"
            )

            print(
                f"Current games:       "
                f"{snapshot.get('current_games', 0):.1f}"
            )

            print(
                f"Historical weight:   "
                f"{snapshot.get('historical_weight', 0):.3f}"
            )

            print(
                f"Current weight:      "
                f"{snapshot.get('current_weight', 0):.3f}"
            )

            print(
                f"Points for:          "
                f"{snapshot.get('points_for', 0):.2f}"
            )

            print(
                f"Points against:      "
                f"{snapshot.get('points_against', 0):.2f}"
            )

            print(
                f"Point differential:  "
                f"{snapshot.get('point_diff', 0):.2f}"
            )

            print(
                f"Rush yards:          "
                f"{snapshot.get('rushing_yards', 0):.2f}"
            )

            print(
                f"Pass yards:          "
                f"{snapshot.get('passing_yards', 0):.2f}"
            )

            print(
                f"Total yards:         "
                f"{snapshot.get('total_yards', 0):.2f}"
            )

            print(
                f"Win rate:            "
                f"{snapshot.get('wins', 0):.3f}"
            )

            print(
                f"Recent win rate:     "
                f"{snapshot.get('recent_win_rate', 0):.3f}"
            )

            print(
                f"Recent point diff:   "
                f"{snapshot.get('recent_point_diff', 0):.2f}"
            )

        print("\n" + "=" * 60)
        print("TEAM FORM TEST COMPLETE")
        print("=" * 60)

    finally:
        session.close()


if __name__ == "__main__":
    main()