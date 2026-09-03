
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, GameTeamStats, Team


@dataclass
class TeamFeatureConfig:
    """
    Configuration for team performance features.

    The engine produces leakage-safe pre-game team snapshots.

    Current-season performance is opponent-adjusted using ONLY
    information that was available before each individual game.

    Historical performance is blended with current-season
    performance as the season progresses.
    """

    # Number of games used for recent-form features.
    recent_games: int = 5

    # Number of historical games used from previous seasons.
    historical_games: int = 12

    # --------------------------------------------------------------
    # HISTORICAL / CURRENT BLENDING
    # --------------------------------------------------------------

    # Historical weighting according to completed current-season
    # games:
    #
    # 0 games -> 100% historical
    # 1 game  -> 90% historical
    # 2 games -> 75% historical
    # 3 games -> 60% historical
    # 4 games -> 50% historical
    # 5 games -> 35% historical
    # 6 games -> 20% historical
    # 7+ games -> 10% historical

    historical_weights: tuple[float, ...] = (
        1.00,
        0.90,
        0.75,
        0.60,
        0.50,
        0.35,
        0.20,
        0.10,
    )

    # --------------------------------------------------------------
    # OPPONENT ADJUSTMENT
    # --------------------------------------------------------------

    # Strength of opponent adjustment.
    #
    # 0.00 = no adjustment
    # 1.00 = full adjustment
    #
    # 0.75 = strong but controlled adjustment.

    opponent_adjustment_strength: float = 0.75

    # Number of previous games used when estimating opponent
    # defensive strength.

    opponent_strength_games: int = 5


class TeamFormEngine:
    """
    Builds leakage-safe team performance features.

    IMPORTANT:

    A target game's features may ONLY use information that would
    have been available before that game.

    Current-season games are opponent-adjusted.

    The opponent adjustment is based on the opponent's ACTUAL
    defensive performance:

        - points allowed
        - passing yards allowed
        - rushing yards allowed
        - total yards allowed
        - yards per play allowed

    These are calculated from the opponent's previous games by
    looking at what THEIR opponents actually produced.

    Example:

        Team A throws for 250 yards against Team B.

        If Team B had previously allowed 300 passing yards/game,
        this performance is less impressive.

        If Team B had previously allowed 170 passing yards/game,
        this performance is more impressive.

    No target-game information is used when calculating the
    adjustment.
    """

    def __init__(
        self,
        session: Session,
        config: TeamFeatureConfig | None = None,
    ) -> None:

        self.session = session

        self.config = (
            config
            if config is not None
            else TeamFeatureConfig()
        )

    # ==============================================================
    # DATA LOADING
    # ==============================================================

    def load_games(self) -> pd.DataFrame:
        """
        Load completed games with valid scores.
        """

        stmt = select(Game).where(
            Game.completed.is_(True)
        )

        games = self.session.scalars(stmt).all()

        rows: list[dict[str, Any]] = []

        for game in games:

            if (
                game.home_points is None
                or game.away_points is None
            ):
                continue

            rows.append(
                {
                    "game_id": game.id,
                    "cfbd_id": game.cfbd_id,
                    "season": game.season,
                    "week": game.week,
                    "start_date": game.start_date,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_points": game.home_points,
                    "away_points": game.away_points,
                    "neutral_site": game.neutral_site,
                    "conference_game": game.conference_game,
                }
            )

        return pd.DataFrame(rows)

    def load_team_stats(self) -> pd.DataFrame:
        """
        Load all team-game statistics.
        """

        stmt = (
            select(
                GameTeamStats,
                Team.school,
            )
            .join(
                Team,
                Team.id == GameTeamStats.team_id,
            )
        )

        results = self.session.execute(stmt).all()

        rows: list[dict[str, Any]] = []

        for stats, school in results:

            rows.append(
                {
                    "game_id": stats.game_id,
                    "season": stats.season,
                    "week": stats.week,
                    "team_id": stats.team_id,
                    "team": school,
                    "home_away": stats.home_away,

                    "points": stats.points,

                    "rushing_tds": stats.rushing_tds,
                    "rushing_attempts": stats.rushing_attempts,
                    "rushing_yards": stats.rushing_yards,
                    "yards_per_rush": stats.yards_per_rush,

                    "passing_tds": stats.passing_tds,
                    "passing_completions": (
                        stats.passing_completions
                    ),
                    "passing_attempts": (
                        stats.passing_attempts
                    ),
                    "net_passing_yards": (
                        stats.net_passing_yards
                    ),
                    "yards_per_pass": (
                        stats.yards_per_pass
                    ),

                    "total_yards": stats.total_yards,
                    "first_downs": stats.first_downs,

                    "third_down_made": getattr(
                        stats,
                        "third_down_made",
                        None,
                    ),

                    "third_down_attempts": getattr(
                        stats,
                        "third_down_attempts",
                        None,
                    ),

                    "fourth_down_made": getattr(
                        stats,
                        "fourth_down_made",
                        None,
                    ),

                    "fourth_down_attempts": getattr(
                        stats,
                        "fourth_down_attempts",
                        None,
                    ),

                    "turnovers": stats.turnovers,
                    "interceptions": stats.interceptions,
                    "passes_intercepted": (
                        stats.passes_intercepted
                    ),

                    "fumbles_lost": stats.fumbles_lost,
                    "fumbles_recovered": (
                        stats.fumbles_recovered
                    ),

                    "kicking_points": stats.kicking_points,

                    "punt_returns": stats.punt_returns,
                    "punt_return_yards": (
                        stats.punt_return_yards
                    ),
                    "punt_return_tds": (
                        stats.punt_return_tds
                    ),

                    "kick_returns": stats.kick_returns,
                    "kick_return_yards": (
                        stats.kick_return_yards
                    ),
                    "kick_return_tds": (
                        stats.kick_return_tds
                    ),

                    "possession_time": (
                        stats.possession_time
                    ),

                    "penalties_yards": (
                        stats.penalties_yards
                    ),
                }
            )

        return pd.DataFrame(rows)

    # ==============================================================
    # HELPERS
    # ==============================================================

    @staticmethod
    def safe_mean(
        series: pd.Series,
    ) -> float:
        """
        Safely calculate a numeric mean.
        """

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        numeric = numeric.dropna()

        if numeric.empty:
            return 0.0

        return float(
            numeric.mean()
        )

    @staticmethod
    def safe_int(value: Any) -> int:
        """
        Safely convert a value to int.
        """

        try:
            if pd.isna(value):
                return 0

            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return 0

    @staticmethod
    def calculate_historical_weight(
        current_games: int,
        historical_weights: tuple[float, ...],
    ) -> float:
        """
        Calculate historical-season weight.
        """

        if current_games <= 0:
            return historical_weights[0]

        index = min(
            current_games,
            len(historical_weights) - 1,
        )

        return historical_weights[index]

    # ==============================================================
    # BUILD TEAM HISTORY
    # ==============================================================

    def build_team_history(
        self,
        stats_df: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine team statistics with game information.

        Produces one row per team per completed game.

        IMPORTANT:

        This method also creates opponent offensive statistics.

        That allows us to correctly calculate defensive statistics:

            points_allowed
            passing_yards_allowed
            rushing_yards_allowed
            total_yards_allowed
            yards_per_play_allowed

        from the actual opponent's performance.
        """

        if stats_df.empty:
            return pd.DataFrame()

        games_lookup = games_df[
            [
                "game_id",
                "home_team",
                "away_team",
                "season",
                "week",
                "start_date",
                "home_points",
                "away_points",
                "neutral_site",
                "conference_game",
            ]
        ].copy()

        df = stats_df.merge(
            games_lookup,
            on="game_id",
            how="left",
            suffixes=("", "_game"),
        )

        # ----------------------------------------------------------
        # NUMERIC CONVERSIONS
        # ----------------------------------------------------------

        numeric_columns = [
            "points",
            "rushing_attempts",
            "passing_attempts",
            "rushing_yards",
            "net_passing_yards",
            "total_yards",
            "yards_per_rush",
            "yards_per_pass",
            "passing_completions",
            "third_down_made",
            "third_down_attempts",
            "fourth_down_made",
            "fourth_down_attempts",
            "turnovers",
            "interceptions",
            "fumbles_lost",
            "first_downs",
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        # ----------------------------------------------------------
        # OPPONENT
        # ----------------------------------------------------------

        df["opponent"] = df.apply(
            lambda row: (
                row["away_team"]
                if str(row["home_away"]).lower() == "home"
                else row["home_team"]
            ),
            axis=1,
        )

        # ----------------------------------------------------------
        # TEAM SCORE
        # ----------------------------------------------------------

        df["team_points"] = pd.to_numeric(
            df["points"],
            errors="coerce",
        )

        # ----------------------------------------------------------
        # OPPONENT SCORE
        # ----------------------------------------------------------

        df["opponent_points"] = df.apply(
            lambda row: (
                row["away_points"]
                if str(row["home_away"]).lower() == "home"
                else row["home_points"]
            ),
            axis=1,
        )

        df["opponent_points"] = pd.to_numeric(
            df["opponent_points"],
            errors="coerce",
        )

        # ----------------------------------------------------------
        # POINT DIFFERENTIAL
        # ----------------------------------------------------------

        df["point_differential"] = (
            df["team_points"]
            - df["opponent_points"]
        )

        # ----------------------------------------------------------
        # WIN
        # ----------------------------------------------------------

        df["win"] = (
            df["point_differential"] > 0
        ).astype(int)

        # ----------------------------------------------------------
        # YARDS PER PLAY
        # ----------------------------------------------------------

        total_plays = (
            df["rushing_attempts"].fillna(0)
            +
            df["passing_attempts"].fillna(0)
        )

        df["yards_per_play"] = (
            df["total_yards"]
            /
            total_plays.replace(
                0,
                pd.NA,
            )
        )

        # ----------------------------------------------------------
        # PASS COMPLETION %
        # ----------------------------------------------------------

        df["pass_completion_pct"] = (
            df["passing_completions"]
            /
            df["passing_attempts"].replace(
                0,
                pd.NA,
            )
        )

        # ----------------------------------------------------------
        # THIRD DOWN %
        # ----------------------------------------------------------

        df["third_down_pct"] = (
            df["third_down_made"]
            /
            df["third_down_attempts"].replace(
                0,
                pd.NA,
            )
        )

        # ----------------------------------------------------------
        # FOURTH DOWN %
        # ----------------------------------------------------------

        df["fourth_down_pct"] = (
            df["fourth_down_made"]
            /
            df["fourth_down_attempts"].replace(
                0,
                pd.NA,
            )
        )

        # ----------------------------------------------------------
        # CREATE OPPONENT OFFENSIVE SNAPSHOT
        # ----------------------------------------------------------
        #
        # Every game has two rows:
        #
        # Team A
        # Team B
        #
        # We merge the opposing row so that Team A's row contains
        # Team B's actual offensive production.
        #
        # This is what allows us to correctly calculate:
        #
        # Team A defensive performance =
        # what Team B produced against Team A.
        #

        opponent_stats = df[
            [
                "game_id",
                "team_id",
                "team",
                "team_points",
                "rushing_yards",
                "net_passing_yards",
                "total_yards",
                "yards_per_play",
            ]
        ].copy()

        opponent_stats = opponent_stats.rename(
            columns={
                "team_id": "_opponent_team_id",
                "team": "_opponent_team_name",
                "team_points": "_opponent_scored",
                "rushing_yards": "_opponent_rushing_yards",
                "net_passing_yards": (
                    "_opponent_passing_yards"
                ),
                "total_yards": "_opponent_total_yards",
                "yards_per_play": (
                    "_opponent_yards_per_play"
                ),
            }
        )

        df = df.merge(
            opponent_stats,
            on="game_id",
            how="left",
        )

        # ----------------------------------------------------------
        # REMOVE SELF-MATCH IF DATABASE EVER CONTAINS DUPLICATES
        # ----------------------------------------------------------

        if "_opponent_team_id" in df.columns:

            same_team = (
                df["_opponent_team_id"]
                == df["team_id"]
            )

            df.loc[
                same_team,
                [
                    "_opponent_team_id",
                    "_opponent_team_name",
                    "_opponent_scored",
                    "_opponent_rushing_yards",
                    "_opponent_passing_yards",
                    "_opponent_total_yards",
                    "_opponent_yards_per_play",
                ],
            ] = pd.NA

        # ----------------------------------------------------------
        # DEFENSIVE METRICS
        # ----------------------------------------------------------

        df["points_allowed"] = pd.to_numeric(
            df["_opponent_scored"],
            errors="coerce",
        )

        df["passing_yards_allowed"] = pd.to_numeric(
            df["_opponent_passing_yards"],
            errors="coerce",
        )

        df["rushing_yards_allowed"] = pd.to_numeric(
            df["_opponent_rushing_yards"],
            errors="coerce",
        )

        df["total_yards_allowed"] = pd.to_numeric(
            df["_opponent_total_yards"],
            errors="coerce",
        )

        df["yards_per_play_allowed"] = pd.to_numeric(
            df["_opponent_yards_per_play"],
            errors="coerce",
        )

        # ----------------------------------------------------------
        # SORT
        # ----------------------------------------------------------

        df = df.sort_values(
            [
                "season",
                "week",
                "game_id",
                "team_id",
            ]
        ).reset_index(drop=True)

        return df

    # ==============================================================
    # PRIOR GAMES
    # ==============================================================

    def get_prior_team_games(
        self,
        team: str,
        season: int,
        week: int,
        game_id: Any,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return games available to a team BEFORE a target game.

        We deliberately DO NOT use same-week games.

        This is safer than assuming that game_id represents
        chronological kickoff order.

        Previous seasons are fully available.
        """

        if history.empty:
            return history.copy()

        prior = history[
            (history["team"] == team)
            &
            (
                (history["season"] < season)
                |
                (
                    (history["season"] == season)
                    &
                    (
                        history["week"].fillna(0)
                        < week
                    )
                )
            )
        ].copy()

        if prior.empty:
            return prior

        return prior.sort_values(
            [
                "season",
                "week",
                "game_id",
            ]
        ).reset_index(drop=True)

    # ==============================================================
    # HISTORICAL FEATURES
    # ==============================================================

    def calculate_historical_features(
        self,
        team: str,
        season: int,
        before_week: int,
        history: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Calculate historical performance using previous seasons only.
        """

        historical = history[
            (history["team"] == team)
            &
            (history["season"] < season)
        ].copy()

        if historical.empty:
            return self.empty_features()

        historical = historical.sort_values(
            [
                "season",
                "week",
                "game_id",
            ]
        )

        historical = historical.tail(
            self.config.historical_games
        )

        return self.aggregate_features(
            historical
        )

    # ==============================================================
    # CURRENT SEASON FEATURES
    # ==============================================================

    def calculate_current_features(
        self,
        team: str,
        season: int,
        before_week: int,
        history: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Calculate opponent-adjusted current-season performance.

        ONLY games before the target week are included.
        """

        current = history[
            (history["team"] == team)
            &
            (history["season"] == season)
            &
            (
                history["week"].fillna(0)
                < before_week
            )
        ].copy()

        if current.empty:
            return self.empty_features()

        current = current.sort_values(
            [
                "week",
                "game_id",
            ]
        ).reset_index(drop=True)

        adjusted_games: list[dict[str, Any]] = []

        for _, game in current.iterrows():

            adjusted_game = (
                self.adjust_game_for_opponent(
                    game=game,
                    history=history,
                )
            )

            adjusted_games.append(
                adjusted_game
            )

        adjusted_df = pd.DataFrame(
            adjusted_games
        )

        return self.aggregate_features(
            adjusted_df
        )

    # ==============================================================
    # OPPONENT DEFENSIVE SNAPSHOT
    # ==============================================================

    def get_opponent_defensive_snapshot(
        self,
        opponent: str,
        season: int,
        week: int,
        game_id: Any,
        history: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Calculate opponent defensive strength BEFORE a game.

        Defensive statistics come from what the opponent actually
        allowed in its previous games.

        Example:

            Opponent allowed:
                170 passing yards/game

            League allowed:
                235 passing yards/game

            That represents a strong pass defence.

        The target game is never included.
        """

        prior = self.get_prior_team_games(
            team=opponent,
            season=season,
            week=week,
            game_id=game_id,
            history=history,
        )

        if prior.empty:
            return {
                "points_allowed": 0.0,
                "passing_yards_allowed": 0.0,
                "rushing_yards_allowed": 0.0,
                "total_yards_allowed": 0.0,
                "yards_per_play_allowed": 0.0,
                "games": 0.0,
            }

        # Use the most recent defensive games.
        prior = prior.tail(
            self.config.opponent_strength_games
        )

        return {
            "points_allowed": self.safe_mean(
                prior["points_allowed"]
            ),

            "passing_yards_allowed": self.safe_mean(
                prior["passing_yards_allowed"]
            ),

            "rushing_yards_allowed": self.safe_mean(
                prior["rushing_yards_allowed"]
            ),

            "total_yards_allowed": self.safe_mean(
                prior["total_yards_allowed"]
            ),

            "yards_per_play_allowed": self.safe_mean(
                prior["yards_per_play_allowed"]
            ),

            "games": float(
                len(prior)
            ),
        }

    # ==============================================================
    # LEAGUE DEFENSIVE BASELINE
    # ==============================================================

    def get_league_defensive_baseline(
        self,
        season: int,
        week: int,
        game_id: Any,
        history: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Calculate the league-wide defensive baseline available
        BEFORE the target game.

        Only previous seasons and previous weeks of the current
        season are included.
        """

        if history.empty:
            return {
                "points_allowed": 0.0,
                "passing_yards_allowed": 0.0,
                "rushing_yards_allowed": 0.0,
                "total_yards_allowed": 0.0,
                "yards_per_play_allowed": 0.0,
            }

        prior = history[
            (
                history["season"] < season
            )
            |
            (
                (history["season"] == season)
                &
                (
                    history["week"].fillna(0)
                    < week
                )
            )
        ].copy()

        if prior.empty:
            return {
                "points_allowed": 0.0,
                "passing_yards_allowed": 0.0,
                "rushing_yards_allowed": 0.0,
                "total_yards_allowed": 0.0,
                "yards_per_play_allowed": 0.0,
            }

        return {
            "points_allowed": self.safe_mean(
                prior["points_allowed"]
            ),

            "passing_yards_allowed": self.safe_mean(
                prior["passing_yards_allowed"]
            ),

            "rushing_yards_allowed": self.safe_mean(
                prior["rushing_yards_allowed"]
            ),

            "total_yards_allowed": self.safe_mean(
                prior["total_yards_allowed"]
            ),

            "yards_per_play_allowed": self.safe_mean(
                prior["yards_per_play_allowed"]
            ),
        }

    # ==============================================================
    # OPPONENT ADJUSTMENT
    # ==============================================================

    def opponent_adjusted_value(
        self,
        actual: float,
        opponent_allowed: float,
        league_allowed: float,
    ) -> float:
        """
        Adjust an observed offensive statistic according to
        opponent defensive strength.

        Additive adjustment:

            adjusted =
                actual
                +
                (league_allowed - opponent_allowed)
                * strength

        Example:

            League allows 235 passing yards.

            Opponent allows 180.

            Team throws for 200.

            Adjustment:

                200 + (235 - 180) * 0.75
                = 241.25

        This makes the performance look better because the team
        faced a stronger-than-average defence.
        """

        if pd.isna(actual):
            return 0.0

        if (
            pd.isna(opponent_allowed)
            or pd.isna(league_allowed)
        ):
            return float(actual)

        if opponent_allowed == 0 or league_allowed == 0:
            return float(actual)

        adjustment = (
            league_allowed
            - opponent_allowed
        )

        adjusted = (
            actual
            +
            (
                adjustment
                *
                self.config.opponent_adjustment_strength
            )
        )

        return float(adjusted)

    # ==============================================================
    # ADJUST INDIVIDUAL GAME
    # ==============================================================

    def adjust_game_for_opponent(
        self,
        game: pd.Series,
        history: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Create an opponent-adjusted version of one game.

        ONLY pre-game information is used to determine opponent
        defensive strength.
        """

        opponent = game["opponent"]

        season = self.safe_int(
            game["season"]
        )

        week = self.safe_int(
            game["week"]
        )

        game_id = game["game_id"]

        opponent_defense = (
            self.get_opponent_defensive_snapshot(
                opponent=opponent,
                season=season,
                week=week,
                game_id=game_id,
                history=history,
            )
        )

        league = (
            self.get_league_defensive_baseline(
                season=season,
                week=week,
                game_id=game_id,
                history=history,
            )
        )

        adjusted = game.to_dict()

        # ----------------------------------------------------------
        # OFFENSIVE VOLUME
        # ----------------------------------------------------------

        adjusted["rushing_yards"] = (
            self.opponent_adjusted_value(
                actual=game["rushing_yards"],
                opponent_allowed=(
                    opponent_defense[
                        "rushing_yards_allowed"
                    ]
                ),
                league_allowed=(
                    league[
                        "rushing_yards_allowed"
                    ]
                ),
            )
        )

        adjusted["net_passing_yards"] = (
            self.opponent_adjusted_value(
                actual=game["net_passing_yards"],
                opponent_allowed=(
                    opponent_defense[
                        "passing_yards_allowed"
                    ]
                ),
                league_allowed=(
                    league[
                        "passing_yards_allowed"
                    ]
                ),
            )
        )

        adjusted["total_yards"] = (
            self.opponent_adjusted_value(
                actual=game["total_yards"],
                opponent_allowed=(
                    opponent_defense[
                        "total_yards_allowed"
                    ]
                ),
                league_allowed=(
                    league[
                        "total_yards_allowed"
                    ]
                ),
            )
        )

        # ----------------------------------------------------------
        # SCORING
        # ----------------------------------------------------------

        adjusted["team_points"] = (
            self.opponent_adjusted_value(
                actual=game["team_points"],
                opponent_allowed=(
                    opponent_defense[
                        "points_allowed"
                    ]
                ),
                league_allowed=(
                    league[
                        "points_allowed"
                    ]
                ),
            )
        )

        # ----------------------------------------------------------
        # YARDS PER PLAY
        # ----------------------------------------------------------

        adjusted["yards_per_play"] = (
            self.opponent_adjusted_value(
                actual=game["yards_per_play"],
                opponent_allowed=(
                    opponent_defense[
                        "yards_per_play_allowed"
                    ]
                ),
                league_allowed=(
                    league[
                        "yards_per_play_allowed"
                    ]
                ),
            )
        )

        # ----------------------------------------------------------
        # POINT DIFFERENTIAL
        # ----------------------------------------------------------
        #
        # IMPORTANT:
        #
        # We use adjusted offensive points but the REAL opponent
        # score. We never change the actual game result.
        #

        adjusted["point_differential"] = (
            adjusted["team_points"]
            -
            game["opponent_points"]
        )

        # ----------------------------------------------------------
        # ACTUAL EFFICIENCY METRICS
        # ----------------------------------------------------------
        #
        # These remain based on what actually happened.
        #

        passing_attempts = pd.to_numeric(
            game["passing_attempts"],
            errors="coerce",
        )

        completions = pd.to_numeric(
            game["passing_completions"],
            errors="coerce",
        )

        third_down_made = pd.to_numeric(
            game["third_down_made"],
            errors="coerce",
        )

        third_down_attempts = pd.to_numeric(
            game["third_down_attempts"],
            errors="coerce",
        )

        fourth_down_made = pd.to_numeric(
            game["fourth_down_made"],
            errors="coerce",
        )

        fourth_down_attempts = pd.to_numeric(
            game["fourth_down_attempts"],
            errors="coerce",
        )

        if (
            pd.notna(completions)
            and pd.notna(passing_attempts)
            and passing_attempts > 0
        ):
            adjusted["pass_completion_pct"] = (
                completions
                /
                passing_attempts
            )

        if (
            pd.notna(third_down_made)
            and pd.notna(third_down_attempts)
            and third_down_attempts > 0
        ):
            adjusted["third_down_pct"] = (
                third_down_made
                /
                third_down_attempts
            )

        if (
            pd.notna(fourth_down_made)
            and pd.notna(fourth_down_attempts)
            and fourth_down_attempts > 0
        ):
            adjusted["fourth_down_pct"] = (
                fourth_down_made
                /
                fourth_down_attempts
            )

        # ----------------------------------------------------------
        # PRESERVE REAL RESULT
        # ----------------------------------------------------------

        adjusted["win"] = int(
            game["win"]
        )

        # ----------------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------------

        adjusted["_opponent_games"] = (
            opponent_defense["games"]
        )

        adjusted["_opponent_points_allowed"] = (
            opponent_defense[
                "points_allowed"
            ]
        )

        adjusted["_opponent_passing_yards_allowed"] = (
            opponent_defense[
                "passing_yards_allowed"
            ]
        )

        adjusted["_opponent_rushing_yards_allowed"] = (
            opponent_defense[
                "rushing_yards_allowed"
            ]
        )

        adjusted["_opponent_total_yards_allowed"] = (
            opponent_defense[
                "total_yards_allowed"
            ]
        )

        adjusted["_league_points_allowed"] = (
            league[
                "points_allowed"
            ]
        )

        adjusted["_league_passing_yards_allowed"] = (
            league[
                "passing_yards_allowed"
            ]
        )

        return adjusted

    # ==============================================================
    # AGGREGATION
    # ==============================================================

    def aggregate_features(
        self,
        games: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Aggregate team performance over a group of games.
        """

        if games.empty:
            return self.empty_features()

        recent = games.tail(
            self.config.recent_games
        )

        return {
            # ------------------------------------------------------
            # SCORING
            # ------------------------------------------------------

            "points_for": self.safe_mean(
                games["team_points"]
            ),

            "points_against": self.safe_mean(
                games["opponent_points"]
            ),

            "point_diff": self.safe_mean(
                games["point_differential"]
            ),

            # ------------------------------------------------------
            # OFFENCE
            # ------------------------------------------------------

            "rushing_yards": self.safe_mean(
                games["rushing_yards"]
            ),

            "passing_yards": self.safe_mean(
                games["net_passing_yards"]
            ),

            "total_yards": self.safe_mean(
                games["total_yards"]
            ),

            "yards_per_rush": self.safe_mean(
                games["yards_per_rush"]
            ),

            "yards_per_pass": self.safe_mean(
                games["yards_per_pass"]
            ),

            "yards_per_play": self.safe_mean(
                games["yards_per_play"]
            ),

            # ------------------------------------------------------
            # BALL SECURITY
            # ------------------------------------------------------

            "turnovers": self.safe_mean(
                games["turnovers"]
            ),

            "interceptions": self.safe_mean(
                games["interceptions"]
            ),

            "fumbles_lost": self.safe_mean(
                games["fumbles_lost"]
            ),

            # ------------------------------------------------------
            # EFFICIENCY
            # ------------------------------------------------------

            "first_downs": self.safe_mean(
                games["first_downs"]
            ),

            "third_down_pct": self.safe_mean(
                games["third_down_pct"]
            ),

            "fourth_down_pct": self.safe_mean(
                games["fourth_down_pct"]
            ),

            "completion_pct": self.safe_mean(
                games["pass_completion_pct"]
            ),

            # ------------------------------------------------------
            # WINNING
            # ------------------------------------------------------

            "wins": self.safe_mean(
                games["win"]
            ),

            # ------------------------------------------------------
            # RECENT FORM
            # ------------------------------------------------------

            "recent_points_for": self.safe_mean(
                recent["team_points"]
            ),

            "recent_points_against": self.safe_mean(
                recent["opponent_points"]
            ),

            "recent_point_diff": self.safe_mean(
                recent["point_differential"]
            ),

            "recent_win_rate": self.safe_mean(
                recent["win"]
            ),

            # ------------------------------------------------------
            # SAMPLE SIZE
            # ------------------------------------------------------

            "games_played": float(
                len(games)
            ),
        }

    # ==============================================================
    # EMPTY FEATURES
    # ==============================================================

    @staticmethod
    def empty_features() -> dict[str, float]:
        """
        Return a complete zero-valued feature set.
        """

        return {
            "points_for": 0.0,
            "points_against": 0.0,
            "point_diff": 0.0,

            "rushing_yards": 0.0,
            "passing_yards": 0.0,
            "total_yards": 0.0,

            "yards_per_rush": 0.0,
            "yards_per_pass": 0.0,
            "yards_per_play": 0.0,

            "turnovers": 0.0,
            "interceptions": 0.0,
            "fumbles_lost": 0.0,

            "first_downs": 0.0,
            "third_down_pct": 0.0,
            "fourth_down_pct": 0.0,
            "completion_pct": 0.0,

            "wins": 0.0,

            "recent_points_for": 0.0,
            "recent_points_against": 0.0,
            "recent_point_diff": 0.0,
            "recent_win_rate": 0.0,

            "games_played": 0.0,
        }

    # ==============================================================
    # HISTORICAL + CURRENT BLEND
    # ==============================================================

    def blend_features(
        self,
        historical: dict[str, float],
        current: dict[str, float],
        current_games: int,
    ) -> dict[str, float]:
        """
        Blend historical and current-season performance.

        Historical/current weighting:

            0 games -> 100% historical
            1 game  -> 90% historical
            2 games -> 75% historical
            3 games -> 60% historical
            4 games -> 50% historical
            5 games -> 35% historical
            6 games -> 20% historical
            7+ games -> 10% historical
        """

        historical_weight = (
            self.calculate_historical_weight(
                current_games=current_games,
                historical_weights=(
                    self.config.historical_weights
                ),
            )
        )

        current_weight = (
            1.0
            - historical_weight
        )

        blended: dict[str, float] = {}

        keys = (
            set(historical.keys())
            |
            set(current.keys())
        )

        for key in keys:

            historical_value = historical.get(
                key,
                0.0,
            )

            current_value = current.get(
                key,
                0.0,
            )

            if current_games <= 0:

                blended[key] = (
                    historical_value
                )

            else:

                blended[key] = (
                    historical_value
                    * historical_weight
                    +
                    current_value
                    * current_weight
                )

        blended["historical_weight"] = (
            historical_weight
        )

        blended["current_weight"] = (
            current_weight
        )

        blended["current_games"] = float(
            current_games
        )

        return blended

    # ==============================================================
    # TEAM SNAPSHOT
    # ==============================================================

    def get_team_snapshot(
        self,
        team: str,
        season: int,
        week: int,
        history: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Produce a leakage-safe pre-game feature snapshot.

        Example:

            Team   = Ohio State
            Season = 2026
            Week   = 4

        Historical:

            Previous seasons only.

        Current:

            2026 Weeks 1-3 only.

        Opponent adjustments:

            Each previous 2026 game is adjusted using the defensive
            strength that its opponent had BEFORE that game.

        Week 4 is NEVER included.
        """

        historical = (
            self.calculate_historical_features(
                team=team,
                season=season,
                before_week=week,
                history=history,
            )
        )

        current = (
            self.calculate_current_features(
                team=team,
                season=season,
                before_week=week,
                history=history,
            )
        )

        current_games = int(
            current.get(
                "games_played",
                0.0,
            )
        )

        return self.blend_features(
            historical=historical,
            current=current,
            current_games=current_games,
        )

