from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import select

from database.database import SessionLocal
from database.models import Team


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "2026_prediction_history.csv"
)


def main() -> None:
    print()
    print("=" * 70)
    print("CHECKING 2026 PREDICTION HISTORY")
    print("=" * 70)

    if not HISTORY_PATH.exists():
        print(f"❌ History file not found: {HISTORY_PATH}")
        return

    history = pd.read_csv(HISTORY_PATH)

    print()
    print(f"History file: {HISTORY_PATH}")
    print(f"Total predictions: {len(history):,}")

    session = SessionLocal()

    try:
        teams = session.scalars(
            select(Team)
        ).all()
    finally:
        session.close()

    team_classifications = {}

    for team in teams:
        if team.school:
            team_classifications[
                team.school.strip().casefold()
            ] = team.classification

    history["home_classification"] = (
        history["home_team"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .map(team_classifications)
    )

    history["away_classification"] = (
        history["away_team"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .map(team_classifications)
    )

    history["fbs_involved"] = (
        history["home_classification"].eq("fbs")
        | history["away_classification"].eq("fbs")
    )

    unmatched = history[
        history["home_classification"].isna()
        | history["away_classification"].isna()
    ].copy()

    fcs_only = history[
        ~history["fbs_involved"]
        & history["home_classification"].notna()
        & history["away_classification"].notna()
    ].copy()

    fbs_involved = history[
        history["fbs_involved"]
    ].copy()

    print()
    print("-" * 70)
    print("CLASSIFICATION SUMMARY")
    print("-" * 70)

    print(
        f"FBS-involved predictions: "
        f"{len(fbs_involved):,}"
    )

    print(
        f"FCS-only predictions: "
        f"{len(fcs_only):,}"
    )

    print(
        f"Predictions with unmatched teams: "
        f"{len(unmatched):,}"
    )

    if not fcs_only.empty:
        print()
        print("-" * 70)
        print("FCS-ONLY PREDICTIONS")
        print("-" * 70)

        print(
            fcs_only[
                [
                    "cfbd_id",
                    "week",
                    "home_team",
                    "home_classification",
                    "away_team",
                    "away_classification",
                ]
            ].to_string(index=False)
        )

    if not unmatched.empty:
        print()
        print("-" * 70)
        print("UNMATCHED TEAMS")
        print("-" * 70)

        print(
            unmatched[
                [
                    "cfbd_id",
                    "week",
                    "home_team",
                    "home_classification",
                    "away_team",
                    "away_classification",
                ]
            ].to_string(index=False)
        )

    print()
    print("-" * 70)
    print("FBS-INVOLVED PREDICTIONS")
    print("-" * 70)

    if fbs_involved.empty:
        print("None found.")
    else:
        print(
            fbs_involved[
                [
                    "cfbd_id",
                    "week",
                    "home_team",
                    "away_team",
                    "home_classification",
                    "away_classification",
                ]
            ].to_string(index=False)
        )

    print()
    print("=" * 70)
    print("HISTORY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()