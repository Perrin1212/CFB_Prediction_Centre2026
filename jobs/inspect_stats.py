from pprint import pprint

from ingestion.cfbd_api import CFBDClient


def main() -> None:
    print("🏈 CFB Prediction Centre")
    print("=" * 40)

    client = CFBDClient()

    # We use a completed season temporarily so that
    # we can inspect the API response structure.
    season = 2025
    week = 1

    print("\nFetching historical team game statistics...")
    print(f"Season: {season}")
    print(f"Week: {week}")

    stats = client.get(
        "games/teams",
        params={
            "year": season,
            "week": week,
        },
    )

    print(f"\n✓ Records returned: {len(stats)}")

    if stats:
        print("\nFirst record:")
        pprint(
            stats[0],
            sort_dicts=False,
        )
    else:
        print("\n⚠ No statistics returned.")


if __name__ == "__main__":
    main()