from __future__ import annotations

from typing import Any

import pandas as pd


class MatchupFeatureBuilder:
    """
    Converts two team snapshots into matchup-level model features.
    """

    def build(
        self,
        home_team: str,
        away_team: str,
        home_features: dict[str, float],
        away_features: dict[str, float],
        neutral_site: bool = False,
    ) -> dict[str, Any]:
        """
        Build pre-game matchup features.
        """

        features: dict[str, Any] = {}

        # --------------------------------------------------------------
        # TEAM IDENTIFIERS
        # --------------------------------------------------------------

        features["home_team"] = home_team
        features["away_team"] = away_team

        features["neutral_site"] = int(
            neutral_site
        )

        # --------------------------------------------------------------
        # RAW TEAM FEATURES
        # --------------------------------------------------------------

        for key, value in home_features.items():

            features[
                f"home_{key}"
            ] = value

        for key, value in away_features.items():

            features[
                f"away_{key}"
            ] = value

        # --------------------------------------------------------------
        # DIFFERENTIAL FEATURES
        # --------------------------------------------------------------

        differential_features = [
            "points_for",
            "points_against",
            "point_diff",
            "rushing_yards",
            "passing_yards",
            "total_yards",
            "yards_per_rush",
            "yards_per_pass",
            "yards_per_play",
            "turnovers",
            "wins",
            "first_downs",
            "third_down_pct",
            "fourth_down_pct",
            "completion_pct",
            "recent_points_for",
            "recent_points_against",
            "recent_point_diff",
            "recent_win_rate",
        ]

        for key in differential_features:

            home_value = home_features.get(
                key,
                0.0,
            )

            away_value = away_features.get(
                key,
                0.0,
            )

            features[
                f"{key}_diff"
            ] = (
                home_value
                - away_value
            )

        # --------------------------------------------------------------
        # CURRENT FORM DIFFERENCES
        # --------------------------------------------------------------

        features[
            "current_weight_diff"
        ] = (
            home_features.get(
                "current_weight",
                0.0,
            )
            -
            away_features.get(
                "current_weight",
                0.0,
            )
        )

        features[
            "recent_form_diff"
        ] = (
            home_features.get(
                "recent_win_rate",
                0.0,
            )
            -
            away_features.get(
                "recent_win_rate",
                0.0,
            )
        )

        # --------------------------------------------------------------
        # HOME FIELD
        # --------------------------------------------------------------

        features[
            "home_field_advantage"
        ] = 0 if neutral_site else 1

        return features