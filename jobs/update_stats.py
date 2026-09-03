from database.database import SessionLocal
from ingestion.stats import StatsIngestor


def main() -> None:
    print("🏈 CFB Prediction Centre")
    print("=" * 40)

    season = 2025
    week = 1

    session = SessionLocal()

    try:
        ingestor = StatsIngestor(session)

        count = ingestor.run_week(
            season=season,
            week=week,
        )

        print()
        print(
            f"✓ Total team-stat records processed: "
            f"{count}"
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()