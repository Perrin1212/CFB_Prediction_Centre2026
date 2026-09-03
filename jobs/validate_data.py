from collections import Counter

from sqlalchemy import func, select

from database.database import SessionLocal, init_db
from database.models import Game, GameTeamStats, Team


def print_header(title: str) -> None:
    """Print a section header."""

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    print("🏈 CFB Prediction Centre")
    print("=" * 60)
    print("DATABASE VALIDATION")
    print("=" * 60)

    init_db()

    session = SessionLocal()

    try:
        # ==============================================================
        # 1. BASIC COUNTS
        # ==============================================================

        print_header("1. BASIC DATABASE COUNTS")

        team_count = session.scalar(
            select(func.count()).select_from(Team)
        ) or 0

        game_count = session.scalar(
            select(func.count()).select_from(Game)
        ) or 0

        stats_count = session.scalar(
            select(func.count()).select_from(GameTeamStats)
        ) or 0

        print(f"Teams:              {team_count}")
        print(f"Games:              {game_count}")
        print(f"Team-game stats:    {stats_count}")

        # ==============================================================
        # 2. GAMES BY SEASON
        # ==============================================================

        print_header("2. GAMES BY SEASON")

        game_seasons = session.execute(
            select(
                Game.season,
                func.count(Game.id),
            )
            .group_by(Game.season)
            .order_by(Game.season)
        ).all()

        for season, count in game_seasons:
            print(f"{season}: {count}")

        # ==============================================================
        # 3. STATS BY SEASON
        # ==============================================================

        print_header("3. TEAM STATS BY SEASON")

        stat_seasons = session.execute(
            select(
                GameTeamStats.season,
                func.count(GameTeamStats.id),
            )
            .group_by(GameTeamStats.season)
            .order_by(GameTeamStats.season)
        ).all()

        for season, count in stat_seasons:
            print(f"{season}: {count}")

        # ==============================================================
        # 4. STATS WITHOUT A VALID GAME
        # ==============================================================

        print_header("4. ORPHANED TEAM STATS")

        orphaned_stats = session.execute(
            select(
                GameTeamStats.id,
                GameTeamStats.game_id,
            )
            .outerjoin(
                Game,
                Game.id == GameTeamStats.game_id,
            )
            .where(
                Game.id.is_(None)
            )
        ).all()

        if orphaned_stats:
            print(
                f"⚠ Orphaned team-stat records: "
                f"{len(orphaned_stats)}"
            )

            for stat_id, game_id in orphaned_stats[:10]:
                print(
                    f"  Stat ID {stat_id} -> "
                    f"missing game {game_id}"
                )

        else:
            print("✓ No orphaned team-stat records")

        # ==============================================================
        # 5. STATS WITHOUT A VALID TEAM
        # ==============================================================

        print_header("5. TEAM-STAT RECORDS WITHOUT VALID TEAMS")

        orphaned_team_stats = session.execute(
            select(
                GameTeamStats.id,
                GameTeamStats.team_id,
            )
            .outerjoin(
                Team,
                Team.id == GameTeamStats.team_id,
            )
            .where(
                Team.id.is_(None)
            )
        ).all()

        if orphaned_team_stats:
            print(
                f"⚠ Stats referencing missing teams: "
                f"{len(orphaned_team_stats)}"
            )

        else:
            print("✓ All team-stat records reference valid teams")

        # ==============================================================
        # 6. DUPLICATE TEAM-GAME STATS
        # ==============================================================

        print_header("6. DUPLICATE TEAM-GAME STATS")

        duplicate_groups = session.execute(
            select(
                GameTeamStats.game_id,
                GameTeamStats.team_id,
                GameTeamStats.season,
                GameTeamStats.week,
                func.count(GameTeamStats.id),
            )
            .group_by(
                GameTeamStats.game_id,
                GameTeamStats.team_id,
                GameTeamStats.season,
                GameTeamStats.week,
            )
            .having(
                func.count(GameTeamStats.id) > 1
            )
        ).all()

        if duplicate_groups:
            print(
                f"⚠ Duplicate groups found: "
                f"{len(duplicate_groups)}"
            )

            for row in duplicate_groups[:10]:
                print(
                    f"  Game={row[0]} "
                    f"Team={row[1]} "
                    f"Season={row[2]} "
                    f"Week={row[3]} "
                    f"Records={row[4]}"
                )

        else:
            print("✓ No duplicate team-game stat groups")

        # ==============================================================
        # 7. TEAM STATS WITH MISSING CORE DATA
        # ==============================================================

        print_header("7. STAT DATA QUALITY")

        total_stats = stats_count

        missing_points = session.scalar(
            select(func.count())
            .select_from(GameTeamStats)
            .where(
                GameTeamStats.points.is_(None)
            )
        ) or 0

        missing_rushing = session.scalar(
            select(func.count())
            .select_from(GameTeamStats)
            .where(
                GameTeamStats.rushing_yards.is_(None)
            )
        ) or 0

        missing_passing = session.scalar(
            select(func.count())
            .select_from(GameTeamStats)
            .where(
                GameTeamStats.net_passing_yards.is_(None)
            )
        ) or 0

        missing_total_yards = session.scalar(
            select(func.count())
            .select_from(GameTeamStats)
            .where(
                GameTeamStats.total_yards.is_(None)
            )
        ) or 0

        if total_stats:
            print(
                f"Missing points:           "
                f"{missing_points} "
                f"({missing_points / total_stats:.1%})"
            )

            print(
                f"Missing rushing yards:     "
                f"{missing_rushing} "
                f"({missing_rushing / total_stats:.1%})"
            )

            print(
                f"Missing passing yards:     "
                f"{missing_passing} "
                f"({missing_passing / total_stats:.1%})"
            )

            print(
                f"Missing total yards:       "
                f"{missing_total_yards} "
                f"({missing_total_yards / total_stats:.1%})"
            )

        # ==============================================================
        # 8. HOME/AWAY BALANCE
        # ==============================================================

        print_header("8. HOME/AWAY STAT COVERAGE")

        home_count = session.scalar(
            select(func.count())
            .select_from(GameTeamStats)
            .where(
                GameTeamStats.home_away == "home"
            )
        ) or 0

        away_count = session.scalar(
            select(func.count())
            .select_from(GameTeamStats)
            .where(
                GameTeamStats.home_away == "away"
            )
        ) or 0

        unknown_count = total_stats - home_count - away_count

        print(f"Home records:       {home_count}")
        print(f"Away records:       {away_count}")
        print(f"Unknown records:    {unknown_count}")

        # ==============================================================
        # 9. GAMES WITH TWO TEAM STAT RECORDS
        # ==============================================================

        print_header("9. GAME STAT COVERAGE")

        stat_game_counts = session.execute(
            select(
                GameTeamStats.game_id,
                func.count(GameTeamStats.id),
            )
            .group_by(GameTeamStats.game_id)
        ).all()

        coverage_counter = Counter(
            count for _, count in stat_game_counts
        )

        games_with_two = coverage_counter.get(2, 0)
        games_with_one = coverage_counter.get(1, 0)
        games_with_more = sum(
            count
            for records, count in coverage_counter.items()
            if records > 2
        )

        print(
            f"Games with 2 team-stat records: "
            f"{games_with_two}"
        )

        print(
            f"Games with 1 team-stat record:  "
            f"{games_with_one}"
        )

        print(
            f"Games with >2 records:           "
            f"{games_with_more}"
        )

        # ==============================================================
        # 10. 2025 COVERAGE
        # ==============================================================

        print_header("10. 2025 HISTORICAL COVERAGE")

        games_2025 = session.scalar(
            select(func.count())
            .select_from(Game)
            .where(
                Game.season == 2025
            )
        ) or 0

        stats_2025 = session.scalar(
            select(func.count())
            .select_from(GameTeamStats)
            .where(
                GameTeamStats.season == 2025
            )
        ) or 0

        games_with_stats_2025 = session.scalar(
            select(
                func.count(
                    func.distinct(
                        GameTeamStats.game_id
                    )
                )
            )
            .where(
                GameTeamStats.season == 2025
            )
        ) or 0

        print(
            f"2025 games:                 {games_2025}"
        )

        print(
            f"2025 team-stat records:     {stats_2025}"
        )

        print(
            f"2025 games with stats:      "
            f"{games_with_stats_2025}"
        )

        if games_2025:
            print(
                f"2025 game coverage:         "
                f"{games_with_stats_2025 / games_2025:.1%}"
            )

        # ==============================================================
        # 11. 2026 TEAM COVERAGE
        # ==============================================================

        print_header("11. 2026 TEAM COVERAGE")

        games_2026 = session.scalar(
            select(func.count())
            .select_from(Game)
            .where(
                Game.season == 2026
            )
        ) or 0

        print(
            f"2026 scheduled games:       {games_2026}"
        )

        # ==============================================================
        # 12. SAMPLE STAT RECORDS
        # ==============================================================

        print_header("12. SAMPLE HISTORICAL STAT RECORDS")

        samples = session.execute(
            select(
                GameTeamStats,
                Game,
                Team,
            )
            .join(
                Game,
                Game.id == GameTeamStats.game_id,
            )
            .join(
                Team,
                Team.id == GameTeamStats.team_id,
            )
            .where(
                GameTeamStats.season == 2025
            )
            .order_by(
                GameTeamStats.id
            )
            .limit(10)
        ).all()

        for stats, game, team in samples:
            print()
            print(
                f"{game.season} W{game.week} | "
                f"{team.school} | "
                f"{stats.home_away}"
            )

            print(
                f"  Points:       {stats.points}"
            )

            print(
                f"  Rush yards:   {stats.rushing_yards}"
            )

            print(
                f"  Pass yards:   {stats.net_passing_yards}"
            )

            print(
                f"  Total yards:  {stats.total_yards}"
            )

            print(
                f"  Turnovers:    {stats.turnovers}"
            )

        # ==============================================================
        # FINAL RESULT
        # ==============================================================

        print_header("VALIDATION COMPLETE")

        problems = []

        if orphaned_stats:
            problems.append(
                f"{len(orphaned_stats)} orphaned game-stat records"
            )

        if orphaned_team_stats:
            problems.append(
                f"{len(orphaned_team_stats)} stats with missing teams"
            )

        if duplicate_groups:
            problems.append(
                f"{len(duplicate_groups)} duplicate stat groups"
            )

        if problems:
            print("⚠ Issues detected:")

            for problem in problems:
                print(f"  - {problem}")

        else:
            print("✓ No structural database problems detected")

        print()
        print(
            "The database is ready for the feature-engineering stage."
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()