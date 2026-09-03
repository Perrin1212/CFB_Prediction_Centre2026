from database.database import SessionLocal, init_db
from ingestion.stats import StatsIngestor


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Historical seasons to load.
#
# We use three completed seasons so the model can learn:
#
#   2023 → long-term historical baseline
#   2024 → recent historical baseline
#   2025 → most recent completed season
#
# This will give the 2026 model much more historical information.
HISTORICAL_SEASONS = [
    2023,
    2024,
    2025,
]


# Regular season weeks.
#
# Week 1-15 covers the normal CFB regular season.
# Week 16 catches conference championship games where applicable.
REGULAR_SEASON_WEEKS = range(1, 17)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def process_week(
    ingestor: StatsIngestor,
    season: int,
    week: int,
) -> int:
    """Process one regular-season week."""

    try:
        count = ingestor.run_week(
            season=season,
            week=week,
        )

        return count

    except Exception as exc:

        print(
            f"  ⚠ Week {week} failed: {exc}"
        )

        return 0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:

    print("🏈 CFB Prediction Centre")
    print("=" * 40)

    print()
    print("[1/3] Checking database...")

    init_db()

    print("✓ Database ready")

    total_processed = 0

    # ------------------------------------------------------------------
    # Historical seasons
    # ------------------------------------------------------------------

    for season in HISTORICAL_SEASONS:

        print()
        print("=" * 40)
        print(
            f"HISTORICAL STATISTICS: {season}"
        )
        print("=" * 40)

        session = SessionLocal()

        try:

            ingestor = StatsIngestor(
                session
            )

            # ----------------------------------------------------------
            # Regular season
            # ----------------------------------------------------------

            print()
            print(
                "Loading regular-season statistics..."
            )

            regular_total = 0

            for week in REGULAR_SEASON_WEEKS:

                print()
                print(
                    f"--- {season} Regular Season Week {week} ---"
                )

                count = process_week(
                    ingestor=ingestor,
                    season=season,
                    week=week,
                )

                regular_total += count

            print()
            print(
                f"✓ {season} regular-season records processed: "
                f"{regular_total}"
            )

            total_processed += regular_total

        finally:

            session.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 40)
    print(
        "HISTORICAL STATISTICS COMPLETE"
    )
    print("=" * 40)

    print()
    print(
        f"✓ Total team-stat records processed: "
        f"{total_processed}"
    )

    print()
    print(
        "Historical statistics loaded for:"
    )

    for season in HISTORICAL_SEASONS:

        print(
            f"  ✓ {season}"
        )

    print()
    print(
        "Next step: validate the expanded database "
        "before rebuilding features."
    )


if __name__ == "__main__":
    main()