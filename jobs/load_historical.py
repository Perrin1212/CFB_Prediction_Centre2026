from database.database import SessionLocal, init_db
from ingestion.games import GameIngestor
from ingestion.stats import StatsIngestor


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Historical seasons to load.
#
# We need the GAME records first because the statistics ingestor
# requires each CFBD game ID to already exist in the database.
HISTORICAL_SEASONS = [
    2023,
    2024,
    2025,
]


# Regular-season weeks.
#
# Week 1-15 covers the normal CFB regular season.
# Week 16 catches conference championship games where applicable.
REGULAR_SEASON_WEEKS = range(1, 17)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:

    print("🏈 CFB Prediction Centre")
    print("=" * 60)
    print("HISTORICAL DATA LOADER")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Initialise database
    # ------------------------------------------------------------------

    print()
    print("[1/3] Checking database...")

    init_db()

    print("✓ Database ready")


    # ==================================================================
    # STEP 1 — LOAD HISTORICAL GAMES
    # ==================================================================

    print()
    print("=" * 60)
    print("STEP 1: LOADING HISTORICAL GAMES")
    print("=" * 60)

    game_total = 0

    for season in HISTORICAL_SEASONS:

        print()
        print(f"--- Loading {season} games ---")

        session = SessionLocal()

        try:

            ingestor = GameIngestor(session)

            count = ingestor.run(
                season=season
            )

            game_total += count

            print(
                f"✓ {season}: {count} games processed"
            )

        except Exception as exc:

            print(
                f"⚠ Failed to load {season} games:"
                f" {exc}"
            )

        finally:

            session.close()


    print()
    print(
        f"✓ Total historical games processed: "
        f"{game_total}"
    )


    # ==================================================================
    # STEP 2 — LOAD HISTORICAL TEAM STATISTICS
    # ==================================================================

    print()
    print("=" * 60)
    print("STEP 2: LOADING HISTORICAL TEAM STATISTICS")
    print("=" * 60)

    total_processed = 0

    for season in HISTORICAL_SEASONS:

        print()
        print("=" * 60)
        print(
            f"HISTORICAL STATISTICS: {season}"
        )
        print("=" * 60)

        session = SessionLocal()

        try:

            ingestor = StatsIngestor(session)

            season_total = 0

            # ----------------------------------------------------------
            # Regular season
            # ----------------------------------------------------------

            print()
            print(
                "Loading regular-season statistics..."
            )

            for week in REGULAR_SEASON_WEEKS:

                print()
                print(
                    f"--- {season} Regular Season "
                    f"Week {week} ---"
                )

                try:

                    count = ingestor.run_week(
                        season=season,
                        week=week,
                    )

                    season_total += count

                    print(
                        f"  ✓ Team-stat records processed: "
                        f"{count}"
                    )

                except Exception as exc:

                    print(
                        f"  ⚠ Week {week} failed: "
                        f"{exc}"
                    )


            print()
            print(
                f"✓ {season} regular-season records "
                f"processed: {season_total}"
            )

            total_processed += season_total

        finally:

            session.close()


    # ==================================================================
    # SUMMARY
    # ==================================================================

    print()
    print("=" * 60)
    print("HISTORICAL DATA LOAD COMPLETE")
    print("=" * 60)

    print()
    print(
        f"✓ Historical games processed: "
        f"{game_total}"
    )

    print(
        f"✓ Historical team-stat records processed: "
        f"{total_processed}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "  python -m jobs.validate_data"
    )


if __name__ == "__main__":
    main()