from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game
from ingestion.cfbd_api import CFBDClient


class GameIngestor:
    """Fetches and stores game information from CFBD."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.client = CFBDClient()

    def fetch_games(self, season: int) -> list[dict[str, Any]]:
        """Fetch games for a season."""

        return self.client.get(
            "games",
            params={
                "year": season,
            },
        )

    @staticmethod
    def parse_date(value: str | None) -> datetime | None:
        """Convert an API date string into a Python datetime."""

        if not value:
            return None

        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).replace(tzinfo=None)

    def upsert_game(
        self,
        game_data: dict[str, Any],
    ) -> Game:
        """Insert a game or update an existing game."""

        cfbd_id = game_data.get("id")

        if cfbd_id is None:
            raise ValueError(
                "Game record is missing CFBD id."
            )

        existing_game = self.session.scalar(
            select(Game).where(
                Game.cfbd_id == cfbd_id
            )
        )

        parsed_date = self.parse_date(
            game_data.get("startDate")
        )

        if existing_game is None:

            existing_game = Game(
                cfbd_id=cfbd_id,
                season=game_data.get(
                    "season"
                ),
                week=game_data.get(
                    "week"
                ),
                season_type=game_data.get(
                    "seasonType"
                ),
                start_date=parsed_date,
                home_team=game_data.get(
                    "homeTeam",
                    "Unknown",
                ),
                away_team=game_data.get(
                    "awayTeam",
                    "Unknown",
                ),
                home_points=game_data.get(
                    "homePoints"
                ),
                away_points=game_data.get(
                    "awayPoints"
                ),
                completed=game_data.get(
                    "completed",
                    False,
                ),
                neutral_site=game_data.get(
                    "neutralSite",
                    False,
                ),
                conference_game=game_data.get(
                    "conferenceGame",
                    False,
                ),
                venue=game_data.get(
                    "venue"
                ),
                attendance=game_data.get(
                    "attendance"
                ),
            )

            self.session.add(
                existing_game
            )

        else:

            existing_game.season = game_data.get(
                "season",
                existing_game.season,
            )

            existing_game.week = game_data.get(
                "week",
                existing_game.week,
            )

            existing_game.season_type = game_data.get(
                "seasonType",
                existing_game.season_type,
            )

            existing_game.start_date = parsed_date

            existing_game.home_team = game_data.get(
                "homeTeam",
                existing_game.home_team,
            )

            existing_game.away_team = game_data.get(
                "awayTeam",
                existing_game.away_team,
            )

            existing_game.home_points = game_data.get(
                "homePoints",
                existing_game.home_points,
            )

            existing_game.away_points = game_data.get(
                "awayPoints",
                existing_game.away_points,
            )

            existing_game.completed = game_data.get(
                "completed",
                existing_game.completed,
            )

            existing_game.neutral_site = game_data.get(
                "neutralSite",
                existing_game.neutral_site,
            )

            existing_game.conference_game = game_data.get(
                "conferenceGame",
                existing_game.conference_game,
            )

            existing_game.venue = game_data.get(
                "venue",
                existing_game.venue,
            )

            existing_game.attendance = game_data.get(
                "attendance",
                existing_game.attendance,
            )

        return existing_game

    def run(self, season: int) -> int:
        """Fetch and upsert all games for a season."""

        games = self.fetch_games(season)

        successful = 0
        failed = 0

        for game_data in games:

            try:
                self.upsert_game(
                    game_data
                )

                successful += 1

            except Exception as exc:

                failed += 1

                print(
                    "⚠ Failed to process game:"
                )

                print(
                    f"  ID: {game_data.get('id')}"
                )

                print(
                    f"  Reason: {exc}"
                )

        self.session.commit()

        print(
            f"✓ Successful: {successful}"
        )

        if failed:
            print(
                f"⚠ Failed: {failed}"
            )

        return successful