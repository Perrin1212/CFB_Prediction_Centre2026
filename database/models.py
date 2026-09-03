from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database.database import Base


class Team(Base):
    """A college football team."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    cfbd_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
    )

    school: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    abbreviation: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    mascot: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    conference: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    classification: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    color: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    logo_url: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Game(Base):
    """A college football game."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    cfbd_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
    )

    season: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    week: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )

    season_type: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    start_date: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    home_team: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        index=True,
    )

    away_team: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        index=True,
    )

    home_points: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    away_points: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    completed: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
    )

    neutral_site: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
    )

    conference_game: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
    )

    venue: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    attendance: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class GameTeamStats(Base):
    """Team-level statistics for a single game."""

    __tablename__ = "game_team_stats"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    game_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    season: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    week: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )

    team_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    home_away: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    points: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    rushing_tds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    rushing_attempts: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    rushing_yards: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    yards_per_rush: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    passing_tds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    passing_completions: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    passing_attempts: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    net_passing_yards: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    yards_per_pass: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    total_yards: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    first_downs: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    third_down_made: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    third_down_attempts: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    fourth_down_made: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    fourth_down_attempts: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    turnovers: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    interceptions: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    passes_intercepted: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    fumbles: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    fumbles_lost: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    fumbles_recovered: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    possession_time: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    penalties_yards: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    punt_returns: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    punt_return_yards: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    punt_return_tds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    kick_returns: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    kick_return_yards: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    kick_return_tds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    kicking_points: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    interception_yards: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    interception_tds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )