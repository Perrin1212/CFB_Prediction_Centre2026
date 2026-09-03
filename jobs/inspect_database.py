from sqlalchemy import func, select

from database.database import SessionLocal, init_db
from database.models import Game, GameTeamStats, Team


def main() -> None:
    print("🏈 CFB Prediction Centre")
    print("=" * 40)

    init_db()

    session = SessionLocal()

    try:
        # ----------------------------------------------------------
        # Teams
        # ----------------------------------------------------------

        team_count = session.scalar(
            select(
                func.count(Team.id)
            )
        )

        # ----------------------------------------------------------
        # Games
        # ----------------------------------------------------------

        game_count = session.scalar(
            select(
                func.count(Game.id)
            )
        )

        # ----------------------------------------------------------
        # Team-game statistics
        # ----------------------------------------------------------

        stats_count = session.scalar(
            select(
                func.count(GameTeamStats.id)
            )
        )

        print()
        print("DATABASE SUMMARY")
        print("-" * 40)

        print(
            f"Teams:              {team_count}"
        )

        print(
            f"Games:              {game_count}"
        )

        print(
            f"Team-game stats:    {stats_count}"
        )

        # ----------------------------------------------------------
        # Games by season
        # ----------------------------------------------------------

        print()
        print("GAMES BY SEASON")
        print("-" * 40)

        rows = session.execute(
            select(
                Game.season,
                func.count(Game.id),
            )
            .group_by(
                Game.season
            )
            .order_by(
                Game.season
            )
        ).all()

        for season, count in rows:

            print(
                f"{season}: {count}"
            )

        # ----------------------------------------------------------
        # Stats by season
        # ----------------------------------------------------------

        print()
        print("TEAM STATS BY SEASON")
        print("-" * 40)

        rows = session.execute(
            select(
                GameTeamStats.season,
                func.count(GameTeamStats.id),
            )
            .group_by(
                GameTeamStats.season
            )
            .order_by(
                GameTeamStats.season
            )
        ).all()

        for season, count in rows:

            print(
                f"{season}: {count}"
            )

        # ----------------------------------------------------------
        # Recent games
        # ----------------------------------------------------------

        print()
        print("SAMPLE GAMES")
        print("-" * 40)

        games = session.scalars(
            select(Game)
            .order_by(
                Game.season.desc(),
                Game.week.desc(),
            )
            .limit(10)
        ).all()

        for game in games:

            print(
                f"{game.season} "
                f"W{game.week} "
                f"{game.home_team} "
                f"vs "
                f"{game.away_team}"
            )

    finally:
        session.close()


if __name__ == "__main__":
    main()