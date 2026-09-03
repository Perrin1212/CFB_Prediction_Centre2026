from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, GameTeamStats, Team
from ingestion.cfbd_api import CFBDClient


class StatsIngestor:
    """Fetches and stores team game statistics from CFBD."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.client = CFBDClient()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def fetch_week_stats(
        self,
        season: int,
        week: int,
    ) -> list[dict[str, Any]]:
        """Fetch team statistics for a specific week."""

        return self.client.get(
            "games/teams",
            params={
                "year": season,
                "week": week,
            },
        )

    # ------------------------------------------------------------------
    # Team lookup / creation
    # ------------------------------------------------------------------

    def get_or_create_team(
        self,
        team_id: int,
        team_name: str | None = None,
        abbreviation: str | None = None,
        conference: str | None = None,
    ) -> Team:
        """
        Find a team by CFBD ID.

        If the team doesn't exist, create a basic record.
        """

        team = self.session.scalar(
            select(Team).where(
                Team.cfbd_id == team_id
            )
        )

        if team is not None:
            return team

        team = Team(
            cfbd_id=team_id,
            school=team_name or f"Unknown Team {team_id}",
            abbreviation=abbreviation,
            conference=conference,
        )

        self.session.add(team)
        self.session.flush()

        print(
            f"  + Added previously unknown team: "
            f"{team.school} ({team_id})"
        )

        return team

    # ------------------------------------------------------------------
    # Stat helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_number(
        value: Any,
    ) -> float | None:
        """Convert a simple numeric CFBD value to float."""

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def parse_pair(
        value: Any,
    ) -> tuple[float | None, float | None]:
        """
        Parse CFBD values such as:

            16-21
            10-16
            2-2

        Returns:

            first, second
        """

        if not isinstance(value, str):
            return None, None

        parts = value.split("-")

        if len(parts) != 2:
            return None, None

        try:
            return (
                float(parts[0]),
                float(parts[1]),
            )
        except ValueError:
            return None, None

    # ------------------------------------------------------------------
    # Individual game
    # ------------------------------------------------------------------

    def process_game(
        self,
        record: dict[str, Any],
        season: int,
        week: int,
    ) -> int:
        """Process statistics for one game."""

        game_id = record.get("id")

        if game_id is None:
            return 0

        game = self.session.scalar(
            select(Game).where(
                Game.cfbd_id == game_id
            )
        )

        if game is None:
            print(
                f"  ⚠ Game {game_id} "
                f"not found in database."
            )

            return 0

        teams = record.get(
            "teams",
            [],
        )

        processed = 0

        for team_data in teams:

            team_id = team_data.get(
                "teamId"
            )

            if team_id is None:
                continue

            team = self.get_or_create_team(
                team_id=team_id,
                team_name=team_data.get(
                    "team"
                ),
                conference=team_data.get(
                    "conference"
                ),
            )

            # ----------------------------------------------------------
            # Convert the API statistics into a dictionary.
            # ----------------------------------------------------------

            stat_dict: dict[str, Any] = {}

            for stat in team_data.get(
                "stats",
                [],
            ):

                category = stat.get(
                    "category"
                )

                value = stat.get(
                    "stat"
                )

                if category:
                    stat_dict[category] = value

            # ----------------------------------------------------------
            # Existing record?
            # ----------------------------------------------------------

            existing = self.session.scalar(
                select(GameTeamStats).where(
                    GameTeamStats.game_id == game.id,
                    GameTeamStats.team_id == team.id,
                    GameTeamStats.season == season,
                    GameTeamStats.week == week,
                )
            )

            if existing is None:

                existing = GameTeamStats(
                    game_id=game.id,
                    season=season,
                    week=week,
                    team_id=team.id,
                )

                self.session.add(
                    existing
                )

            else:

                # Make absolutely sure historical metadata is present.
                existing.season = season
                existing.week = week

            # ----------------------------------------------------------
            # Core
            # ----------------------------------------------------------

            existing.home_away = team_data.get(
                "homeAway"
            )

            existing.points = self.parse_number(
                team_data.get(
                    "points"
                )
            )

            # ----------------------------------------------------------
            # Rushing
            # ----------------------------------------------------------

            existing.rushing_tds = self.parse_number(
                stat_dict.get(
                    "rushingTDs"
                )
            )

            existing.rushing_attempts = self.parse_number(
                stat_dict.get(
                    "rushingAttempts"
                )
            )

            existing.rushing_yards = self.parse_number(
                stat_dict.get(
                    "rushingYards"
                )
            )

            existing.yards_per_rush = self.parse_number(
                stat_dict.get(
                    "yardsPerRushAttempt"
                )
            )

            # ----------------------------------------------------------
            # Passing
            # ----------------------------------------------------------

            existing.passing_tds = self.parse_number(
                stat_dict.get(
                    "passingTDs"
                )
            )

            completions, attempts = self.parse_pair(
                stat_dict.get(
                    "completionAttempts"
                )
            )

            existing.passing_completions = completions
            existing.passing_attempts = attempts

            existing.net_passing_yards = self.parse_number(
                stat_dict.get(
                    "netPassingYards"
                )
            )

            existing.yards_per_pass = self.parse_number(
                stat_dict.get(
                    "yardsPerPass"
                )
            )

            # ----------------------------------------------------------
            # Total offense
            # ----------------------------------------------------------

            existing.total_yards = self.parse_number(
                stat_dict.get(
                    "totalYards"
                )
            )

            existing.first_downs = self.parse_number(
                stat_dict.get(
                    "firstDowns"
                )
            )

            # ----------------------------------------------------------
            # Third / fourth down
            # ----------------------------------------------------------

            third_made, third_attempts = self.parse_pair(
                stat_dict.get(
                    "thirdDownEff"
                )
            )

            existing.third_down_conversions = third_made
            existing.third_down_attempts = third_attempts

            fourth_made, fourth_attempts = self.parse_pair(
                stat_dict.get(
                    "fourthDownEff"
                )
            )

            existing.fourth_down_conversions = fourth_made
            existing.fourth_down_attempts = fourth_attempts

            # ----------------------------------------------------------
            # Turnovers
            # ----------------------------------------------------------

            existing.turnovers = self.parse_number(
                stat_dict.get(
                    "turnovers"
                )
            )

            existing.interceptions = self.parse_number(
                stat_dict.get(
                    "interceptions"
                )
            )

            existing.passes_intercepted = self.parse_number(
                stat_dict.get(
                    "passesIntercepted"
                )
            )

            existing.fumbles_lost = self.parse_number(
                stat_dict.get(
                    "fumblesLost"
                )
            )

            existing.fumbles_recovered = self.parse_number(
                stat_dict.get(
                    "fumblesRecovered"
                )
            )

            # ----------------------------------------------------------
            # Special teams
            # ----------------------------------------------------------

            existing.kicking_points = self.parse_number(
                stat_dict.get(
                    "kickingPoints"
                )
            )

            existing.punt_returns = self.parse_number(
                stat_dict.get(
                    "puntReturns"
                )
            )

            existing.punt_return_yards = self.parse_number(
                stat_dict.get(
                    "puntReturnYards"
                )
            )

            existing.punt_return_tds = self.parse_number(
                stat_dict.get(
                    "puntReturnTDs"
                )
            )

            existing.kick_returns = self.parse_number(
                stat_dict.get(
                    "kickReturns"
                )
            )

            existing.kick_return_yards = self.parse_number(
                stat_dict.get(
                    "kickReturnYards"
                )
            )

            existing.kick_return_tds = self.parse_number(
                stat_dict.get(
                    "kickReturnTDs"
                )
            )

            # ----------------------------------------------------------
            # Discipline / possession
            # ----------------------------------------------------------

            existing.total_penalties_yards = stat_dict.get(
                "totalPenaltiesYards"
            )

            existing.possession_time = stat_dict.get(
                "possessionTime"
            )

            processed += 1

        return processed

    # ------------------------------------------------------------------
    # Week
    # ------------------------------------------------------------------

    def run_week(
        self,
        season: int,
        week: int,
    ) -> int:
        """Fetch and process one week's statistics."""

        print()
        print(
            f"📊 Processing statistics: "
            f"{season} Week {week}"
        )

        records = self.fetch_week_stats(
            season=season,
            week=week,
        )

        successful = 0
        failed = 0

        for record in records:

            game_id = record.get(
                "id"
            )

            try:

                count = self.process_game(
                    record=record,
                    season=season,
                    week=week,
                )

                if count > 0:
                    successful += count

            except Exception as exc:

                # Important:
                # A failed flush can put the SQLAlchemy session into
                # a rollback state. Roll back before processing the
                # next game.
                self.session.rollback()

                failed += 1

                print(
                    f"  ⚠ Failed game "
                    f"{game_id}: {exc}"
                )

        self.session.commit()

        print(
            f"  ✓ Team-stat records processed: "
            f"{successful}"
        )

        if failed:
            print(
                f"  ⚠ Failed game records: "
                f"{failed}"
            )

        return successful