from __future__ import annotations

from pathlib import Path
import shutil

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

BACKUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "2026_prediction_history_backup_before_fbs_cleanup.csv"
)


def normalise_team_name(value: object) -> str:
    """Normalise a team name for reliable matching."""
    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .casefold()
        .replace("’", "'")
    )


def load_team_classifications() -> dict[str, str | None]:
    """Load team classifications from the database."""

    session = SessionLocal()

    try:
        teams = session.scalars(
            select(Team)
        ).all()
    finally:
        session.close()

    classifications: dict[str, str | None] = {}

    for team in teams:
        if not team.school:
            continue

        classifications[
            normalise_team_name(team.school)
        ] = team.classification

    return classifications


def main() -> None:
    print()
    print("=" * 70)
    print("2026 PREDICTION HISTORY — FBS CLEANUP")
    print("=" * 70)

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Prediction history not found:\n{HISTORY_PATH}"
        )

    print()
    print("Loading prediction history...")

    history = pd.read_csv(HISTORY_PATH)

    print(
        f"✓ Existing history records: "
        f"{len(history):,}"
    )

    required_columns = {
        "cfbd_id",
        "home_team",
        "away_team",
    }

    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise ValueError(
            "Prediction history is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print()
    print("Creating backup...")

    shutil.copy2(
        HISTORY_PATH,
        BACKUP_PATH,
    )

    print(
        f"✓ Backup created:\n"
        f"  {BACKUP_PATH}"
    )

    print()
    print("Loading team classifications...")

    classifications = load_team_classifications()

    print(
        f"✓ Team classifications loaded: "
        f"{len(classifications):,}"
    )

    history["home_classification"] = (
        history["home_team"]
        .apply(normalise_team_name)
        .map(classifications)
    )

    history["away_classification"] = (
        history["away_team"]
        .apply(normalise_team_name)
        .map(classifications)
    )

    history["fbs_involved"] = (
        history["home_classification"].eq("fbs")
        | history["away_classification"].eq("fbs")
    )

    unmatched = history[
        history["home_classification"].isna()
        | history["away_classification"].isna()
    ].copy()

    fbs_history = history[
        history["fbs_involved"]
    ].copy()

    removed = history[
        ~history["fbs_involved"]
    ].copy()

    print()
    print("-" * 70)
    print("CLEANUP SUMMARY")
    print("-" * 70)

    print(
        f"Original records:        {len(history):,}"
    )

    print(
        f"FBS-involved records:     {len(fbs_history):,}"
    )

    print(
        f"Records to remove:        {len(removed):,}"
    )

    print(
        f"Unmatched records:        {len(unmatched):,}"
    )

    print()
    print("-" * 70)
    print("FBS PREDICTIONS BEING RETAINED")
    print("-" * 70)

    print(
        fbs_history[
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
    print("-" * 70)
    print("REMOVING NON-FBS HISTORY")
    print("-" * 70)

    print(
        "The following historical records will be removed "
        "because neither team is classified as FBS:"
    )

    if removed.empty:
        print("None.")
    else:
        print(
            removed[
                [
                    "cfbd_id",
                    "home_team",
                    "away_team",
                    "home_classification",
                    "away_classification",
                ]
            ].to_string(index=False)
        )

    # Remove diagnostic columns before saving.
    clean_history = fbs_history.drop(
        columns=[
            "home_classification",
            "away_classification",
            "fbs_involved",
        ],
        errors="ignore",
    ).copy()

    # Preserve one record per CFBD game.
    clean_history = clean_history.drop_duplicates(
        subset=["cfbd_id"],
        keep="first",
    )

    clean_history = clean_history.reset_index(
        drop=True
    )

    clean_history.to_csv(
        HISTORY_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("FBS HISTORY CLEANUP COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Original history: "
        f"{len(history):,}"
    )

    print(
        f"Retained FBS history: "
        f"{len(clean_history):,}"
    )

    print(
        f"Removed non-FBS history: "
        f"{len(history) - len(clean_history):,}"
    )

    print()
    print(
        "Backup preserved at:"
    )

    print(
        f"  {BACKUP_PATH}"
    )

    print()
    print(
        "Clean history saved at:"
    )

    print(
        f"  {HISTORY_PATH}"
    )

    print()
    print(
        "The official FBS prediction history is now clean."
    )


if __name__ == "__main__":
    main()