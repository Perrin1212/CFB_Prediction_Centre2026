from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Team
from ingestion.cfbd_api import CFBDClient


class TeamIngestor:
    """Fetches and stores team information from CFBD."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.client = CFBDClient()

    def fetch_teams(self, season: int) -> list[dict[str, Any]]:
        """Fetch teams from CFBD for a season."""

        return self.client.get(
            "teams",
            params={"year": season},
        )

    def upsert_team(self, team_data: dict[str, Any]) -> Team:
        """Insert a team or update an existing team."""

        cfbd_id = team_data.get("id")

        if cfbd_id is None:
            raise ValueError("Team record is missing CFBD id.")

        existing_team = self.session.scalar(
            select(Team).where(
                Team.cfbd_id == cfbd_id
            )
        )

        if existing_team is None:
            logos = team_data.get("logos")

            logo_url = None

            if isinstance(logos, list) and logos:
                logo_url = logos[0]

            existing_team = Team(
                cfbd_id=cfbd_id,
                school=team_data.get("school", "Unknown"),
                abbreviation=team_data.get("abbreviation"),
                mascot=team_data.get("mascot"),
                conference=team_data.get("conference"),
                classification=team_data.get("classification"),
                color=team_data.get("color"),
                logo_url=logo_url,
            )

            self.session.add(existing_team)

        else:
            existing_team.school = team_data.get(
                "school",
                existing_team.school,
            )

            existing_team.abbreviation = team_data.get(
                "abbreviation",
                existing_team.abbreviation,
            )

            existing_team.mascot = team_data.get(
                "mascot",
                existing_team.mascot,
            )

            existing_team.conference = team_data.get(
                "conference",
                existing_team.conference,
            )

            existing_team.classification = team_data.get(
                "classification",
                existing_team.classification,
            )

            existing_team.color = team_data.get(
                "color",
                existing_team.color,
            )

            logos = team_data.get("logos")

            if isinstance(logos, list) and logos:
                existing_team.logo_url = logos[0]

        return existing_team

    def run(self, season: int) -> int:
        """Fetch and upsert all teams for a season."""

        teams = self.fetch_teams(season)

        successful = 0
        failed = 0

        for team_data in teams:
            try:
                self.upsert_team(team_data)
                successful += 1

            except Exception as exc:
                failed += 1

                school = team_data.get(
                    "school",
                    "Unknown",
                )

                print(
                    f"⚠ Failed to process team: {school}"
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