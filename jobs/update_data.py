from config.settings import CURRENT_SEASON
from database.database import SessionLocal, init_db
from ingestion.games import GameIngestor
from ingestion.teams import TeamIngestor


def main() -> None:
    print("🏈 CFB Prediction Centre")
    print("=" * 40)

    # --------------------------------------------------------------
    # 1. Database
    # --------------------------------------------------------------

    print("\n[1/3] Checking database...")

    init_db()

    print("✓ Database ready")

    # --------------------------------------------------------------
    # 2. Teams
    # --------------------------------------------------------------

    print("\n[2/3] Updating teams...")

    session = SessionLocal()

    try:
        team_ingestor = TeamIngestor(session)

        team_count = team_ingestor.run(
            CURRENT_SEASON
        )

        print(
            f"✓ Teams processed: {team_count}"
        )

    finally:
        session.close()

    # --------------------------------------------------------------
    # 3. Games
    # --------------------------------------------------------------

    print("\n[3/3] Updating games...")

    session = SessionLocal()

    try:
        game_ingestor = GameIngestor(session)

        game_count = game_ingestor.run(
            CURRENT_SEASON
        )

        print(
            f"✓ Games processed: {game_count}"
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()