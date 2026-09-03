from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.database import SessionLocal
from database.models import Game, Team
from features.matchups import MatchupFeatureBuilder
from features.team_form import TeamFormEngine
from ingestion.cfbd_api import CFBDClient


# ============================================================
# CONFIGURATION
# ============================================================

SEASON = 2026

# A prediction becomes official when the game is <= 24 hours
# from kickoff.
LOCK_HOURS = 24

# Betting data is captured once, at the moment the prediction
# enters the official lock window.
CAPTURE_LOCK_MARKET = True

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "cfb_home_win_model.joblib"
)

FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "feature_columns.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "2026_predictions.csv"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "2026_prediction_history.csv"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp_string() -> str:
    return utc_now().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def parse_datetime_utc(value: Any):
    if pd.isna(value):
        return pd.NaT

    try:
        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        return timestamp

    except (TypeError, ValueError):
        return pd.NaT


def normalise_team_name(value: object) -> str:
    """Normalise team names for reliable matching."""

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .casefold()
        .replace("’", "'")
    )


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def american_odds_valid(value: Any) -> bool:
    odds = safe_float(value)

    if odds is None:
        return False

    return odds != 0


# ============================================================
# MODEL
# ============================================================

def load_model():
    print()
    print("Loading trained model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print(
        f"✓ Model loaded: {MODEL_PATH}"
    )

    return model


def load_feature_columns() -> list[str]:
    print()
    print("Loading model feature columns...")

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Feature column file not found: "
            f"{FEATURE_COLUMNS_PATH}"
        )

    with open(
        FEATURE_COLUMNS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    if not isinstance(feature_columns, list):
        raise ValueError(
            "feature_columns.json must contain a list."
        )

    if not feature_columns:
        raise ValueError(
            "No model feature columns found."
        )

    print(
        f"✓ Model features: "
        f"{len(feature_columns)}"
    )

    return feature_columns


# ============================================================
# TEAM UNIVERSE
# ============================================================

def load_fbs_team_universe(
    session,
) -> dict[str, str | None]:
    """
    Build a mapping of team name -> classification.

    Prediction universe:

        FBS vs FBS
        FBS vs FCS

    Excluded:

        FCS vs FCS
        lower division only games
    """

    print()
    print("Loading FBS team universe...")

    stmt = (
        select(
            Team.school,
            Team.classification,
        )
        .where(
            Team.classification.is_not(None)
        )
    )

    rows = session.execute(stmt).all()

    classifications: dict[str, str | None] = {}

    for school, classification in rows:

        if not school:
            continue

        classifications[
            normalise_team_name(school)
        ] = (
            classification.strip().casefold()
            if classification
            else None
        )

    fbs_count = sum(
        1
        for classification
        in classifications.values()
        if classification == "fbs"
    )

    print(
        f"✓ FBS teams available: "
        f"{fbs_count:,}"
    )

    return classifications


# ============================================================
# UPCOMING GAMES
# ============================================================

def load_upcoming_games(
    session,
    team_classifications: dict[str, str | None],
) -> pd.DataFrame:
    """
    Load upcoming games involving at least one FBS team.

    Included:

        FBS vs FBS
        FBS vs FCS

    Excluded:

        FCS vs FCS
        lower-division only games
    """

    print()
    print(
        f"Loading upcoming {SEASON} games..."
    )

    stmt = (
        select(Game)
        .where(
            Game.season == SEASON,
            Game.completed.is_(False),
        )
        .order_by(
            Game.start_date,
            Game.week,
            Game.id,
        )
    )

    games = session.scalars(stmt).all()

    total_upcoming = len(games)

    rows: list[dict] = []

    excluded_non_fbs = 0
    unmatched_games = 0

    unmatched_team_names: set[str] = set()

    for game in games:

        home_key = normalise_team_name(
            game.home_team
        )

        away_key = normalise_team_name(
            game.away_team
        )

        home_classification = (
            team_classifications.get(
                home_key
            )
        )

        away_classification = (
            team_classifications.get(
                away_key
            )
        )

        home_is_fbs = (
            home_classification == "fbs"
        )

        away_is_fbs = (
            away_classification == "fbs"
        )

        if not home_is_fbs and not away_is_fbs:

            excluded_non_fbs += 1

            if home_classification is None:
                unmatched_team_names.add(
                    game.home_team
                )

            if away_classification is None:
                unmatched_team_names.add(
                    game.away_team
                )

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
                "neutral_site": game.neutral_site,
                "conference_game": game.conference_game,
                "venue": game.venue,
                "home_classification": home_classification,
                "away_classification": away_classification,
            }
        )

        if (
            home_classification is None
            or away_classification is None
        ):

            unmatched_games += 1

            if home_classification is None:
                unmatched_team_names.add(
                    game.home_team
                )

            if away_classification is None:
                unmatched_team_names.add(
                    game.away_team
                )

    df = pd.DataFrame(rows)

    if not df.empty:

        df["cfbd_id"] = pd.to_numeric(
            df["cfbd_id"],
            errors="coerce",
        )

        df["start_date_utc"] = (
            df["start_date"]
            .apply(parse_datetime_utc)
        )

    print()
    print(
        f"✓ Total non-completed 2026 games: "
        f"{total_upcoming:,}"
    )

    print(
        f"✓ Upcoming FBS-involved games: "
        f"{len(df):,}"
    )

    print(
        f"✓ Non-FBS-only games excluded: "
        f"{excluded_non_fbs:,}"
    )

    if unmatched_games:

        print()
        print(
            f"⚠ FBS-involved games with "
            f"unmatched team classification: "
            f"{unmatched_games:,}"
        )

    if unmatched_team_names:

        print()
        print(
            "⚠ Unmatched team names encountered:"
        )

        for team_name in sorted(
            unmatched_team_names
        ):
            print(
                f"  - {team_name}"
            )

    df = df.drop(
        columns=[
            "home_classification",
            "away_classification",
        ],
        errors="ignore",
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_prediction_features(
    games: pd.DataFrame,
    engine: TeamFormEngine,
    history: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print(
        "Building dynamic 2026 prediction features..."
    )

    matchup_builder = MatchupFeatureBuilder()

    feature_rows: list[dict] = []

    total_games = len(games)

    for index, (_, game) in enumerate(
        games.iterrows(),
        start=1,
    ):

        season = int(
            game["season"]
        )

        week = int(
            game["week"]
            if pd.notna(game["week"])
            else 0
        )

        home_team = game["home_team"]
        away_team = game["away_team"]

        home_features = (
            engine.get_team_snapshot(
                team=home_team,
                season=season,
                week=week,
                history=history,
            )
        )

        away_features = (
            engine.get_team_snapshot(
                team=away_team,
                season=season,
                week=week,
                history=history,
            )
        )

        matchup = matchup_builder.build(
            home_team=home_team,
            away_team=away_team,
            home_features=home_features,
            away_features=away_features,
            neutral_site=bool(
                game["neutral_site"]
            ),
        )

        matchup["game_id"] = int(
            game["game_id"]
        )

        matchup["cfbd_id"] = int(
            game["cfbd_id"]
        )

        matchup["season"] = season
        matchup["week"] = week

        matchup["start_date"] = (
            game["start_date"]
        )

        matchup["home_team"] = home_team
        matchup["away_team"] = away_team

        matchup["neutral_site"] = bool(
            game["neutral_site"]
        )

        matchup["conference_game"] = bool(
            game["conference_game"]
        )

        matchup["venue"] = game["venue"]

        feature_rows.append(matchup)

        if (
            index % 25 == 0
            or index == total_games
        ):

            print(
                f"  Processed "
                f"{index:,}/{total_games:,}"
            )

    if not feature_rows:
        raise RuntimeError(
            "No prediction features were generated."
        )

    return pd.DataFrame(
        feature_rows
    )


def validate_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Prediction dataset is missing model "
            f"features: {missing}"
        )

    print()
    print(
        "✓ Feature validation passed."
    )

    print(
        f"  Required features: "
        f"{len(feature_columns)}"
    )


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def generate_predictions(
    feature_df: pd.DataFrame,
    model,
    feature_columns: list[str],
) -> pd.DataFrame:

    print()
    print(
        "Generating 2026 predictions..."
    )

    X = feature_df[
        feature_columns
    ].copy()

    probabilities = model.predict_proba(
        X
    )[:, 1]

    output = feature_df[
        [
            "game_id",
            "cfbd_id",
            "season",
            "week",
            "start_date",
            "home_team",
            "away_team",
            "neutral_site",
            "conference_game",
            "venue",
        ]
    ].copy()

    output[
        "home_win_probability"
    ] = probabilities

    output[
        "away_win_probability"
    ] = (
        1.0
        - output[
            "home_win_probability"
        ]
    )

    output[
        "predicted_winner"
    ] = output.apply(
        lambda row: (
            row["home_team"]
            if row[
                "home_win_probability"
            ] >= 0.5
            else row["away_team"]
        ),
        axis=1,
    )

    output[
        "prediction_probability"
    ] = output[
        [
            "home_win_probability",
            "away_win_probability",
        ]
    ].max(axis=1)

    output["confidence"] = pd.cut(
        output[
            "prediction_probability"
        ],
        bins=[
            0.0,
            0.60,
            0.70,
            0.80,
            1.00,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
        include_lowest=True,
    )

    output[
        "prediction_timestamp"
    ] = utc_timestamp_string()

    now = pd.Timestamp.now(
        tz="UTC"
    )

    output[
        "start_date_utc"
    ] = output[
        "start_date"
    ].apply(
        parse_datetime_utc
    )

    output[
        "hours_to_kickoff"
    ] = (
        (
            output[
                "start_date_utc"
            ]
            - now
        )
        .dt.total_seconds()
        / 3600
    )

    output[
        "prediction_status"
    ] = output[
        "hours_to_kickoff"
    ].apply(
        lambda hours: (
            "LOCK_WINDOW"
            if (
                pd.notna(hours)
                and hours <= LOCK_HOURS
            )
            else "FORECAST"
        )
    )

    output = output.sort_values(
        [
            "week",
            "start_date",
            "prediction_probability",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return output


# ============================================================
# PREDICTION HISTORY
# ============================================================

def load_prediction_history() -> pd.DataFrame:

    if not HISTORY_PATH.exists():

        print()
        print(
            "No prediction history found."
        )

        return pd.DataFrame()

    history = pd.read_csv(
        HISTORY_PATH
    )

    if history.empty:
        return pd.DataFrame()

    if "cfbd_id" not in history.columns:
        raise ValueError(
            "Existing prediction history does not "
            "contain cfbd_id."
        )

    history[
        "cfbd_id"
    ] = pd.to_numeric(
        history["cfbd_id"],
        errors="coerce",
    )

    return history


# ============================================================
# MARKET DATA
# ============================================================

def extract_numeric(
    data: dict[str, Any],
    *keys: str,
) -> float | None:

    for key in keys:

        value = data.get(key)

        if value is None:
            continue

        parsed = safe_float(value)

        if parsed is not None:
            return parsed

    return None


def choose_market_line(
    lines: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Select one usable market record.

    Preference:

        1. Provider record with both moneylines.
        2. Provider record with a spread.
        3. First usable provider record.

    This is deliberately deterministic.

    We store the selected provider alongside the
    captured values so the snapshot is auditable.
    """

    if not lines:
        return None

    candidates: list[dict[str, Any]] = []

    for line in lines:

        if not isinstance(line, dict):
            continue

        provider = (
            line.get("provider")
            or line.get("name")
            or "Unknown"
        )

        home_ml = extract_numeric(
            line,
            "homeMoneyline",
            "home_moneyline",
        )

        away_ml = extract_numeric(
            line,
            "awayMoneyline",
            "away_moneyline",
        )

        spread = extract_numeric(
            line,
            "spread",
            "homeSpread",
            "home_spread",
        )

        over_under = extract_numeric(
            line,
            "overUnder",
            "over_under",
        )

        candidate = {
            "provider": str(provider),
            "home_moneyline": home_ml,
            "away_moneyline": away_ml,
            "spread": spread,
            "over_under": over_under,
        }

        candidates.append(candidate)

    if not candidates:
        return None

    for candidate in candidates:

        if (
            american_odds_valid(
                candidate["home_moneyline"]
            )
            and american_odds_valid(
                candidate["away_moneyline"]
            )
        ):
            return candidate

    for candidate in candidates:

        if candidate["spread"] is not None:
            return candidate

    return candidates[0]


def capture_lock_market(
    game_id: int,
    home_team: str,
    away_team: str,
) -> dict[str, Any]:
    """
    Capture betting data at the exact point that
    the prediction is being locked.

    This data is stored permanently with the
    official prediction.

    If the API cannot provide a line, we return
    an unavailable snapshot rather than inventing
    anything.
    """

    captured_at = utc_timestamp_string()

    snapshot: dict[str, Any] = {
        "lock_market_captured": False,
        "lock_market_timestamp": captured_at,
        "lock_provider": None,
        "lock_home_moneyline": None,
        "lock_away_moneyline": None,
        "lock_spread": None,
        "lock_spread_favourite": None,
        "lock_over_under": None,
    }

    if not CAPTURE_LOCK_MARKET:
        return snapshot

    print()
    print(
        f"  Capturing lock-time market "
        f"for {away_team} @ {home_team}..."
    )

    try:

        client = CFBDClient()

        response = client.get(
            "lines",
            params={
                "gameId": game_id,
            },
        )

        if not isinstance(response, list):
            print(
                "  ⚠ CFBD lines response was not a list."
            )
            return snapshot

        market = choose_market_line(
            response
        )

        if market is None:
            print(
                "  ⚠ No usable market line returned."
            )
            return snapshot

        home_ml = market[
            "home_moneyline"
        ]

        away_ml = market[
            "away_moneyline"
        ]

        spread = market[
            "spread"
        ]

        snapshot[
            "lock_provider"
        ] = market[
            "provider"
        ]

        snapshot[
            "lock_home_moneyline"
        ] = home_ml

        snapshot[
            "lock_away_moneyline"
        ] = away_ml

        snapshot[
            "lock_spread"
        ] = spread

        snapshot[
            "lock_over_under"
        ] = market[
            "over_under"
        ]

        if spread is not None:

            if spread < 0:

                snapshot[
                    "lock_spread_favourite"
                ] = home_team

            elif spread > 0:

                snapshot[
                    "lock_spread_favourite"
                ] = away_team

            else:

                snapshot[
                    "lock_spread_favourite"
                ] = None

        snapshot[
            "lock_market_captured"
        ] = True

        print(
            "  ✓ Lock market captured"
        )

        print(
            f"    Provider: "
            f"{snapshot['lock_provider']}"
        )

        print(
            f"    Home ML: "
            f"{snapshot['lock_home_moneyline']}"
        )

        print(
            f"    Away ML: "
            f"{snapshot['lock_away_moneyline']}"
        )

        print(
            f"    Spread: "
            f"{snapshot['lock_spread']}"
        )

    except Exception as exc:

        print(
            "  ⚠ Unable to capture lock-time "
            f"market: {exc}"
        )

    return snapshot


def add_model_selected_lock_odds(
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Determine the moneyline attached to the team's
    model selected at lock.
    """

    home = row.get(
        "home_team"
    )

    away = row.get(
        "away_team"
    )

    pick = row.get(
        "predicted_winner"
    )

    home_ml = safe_float(
        row.get(
            "lock_home_moneyline"
        )
    )

    away_ml = safe_float(
        row.get(
            "lock_away_moneyline"
        )
    )

    selected_odds = None
    bet_side = None

    if pick == home:

        selected_odds = home_ml
        bet_side = "HOME"

    elif pick == away:

        selected_odds = away_ml
        bet_side = "AWAY"

    row[
        "lock_moneyline"
    ] = selected_odds

    row[
        "lock_moneyline_odds"
    ] = selected_odds

    row[
        "bet_side"
    ] = bet_side

    return row


# ============================================================
# LOCK OFFICIAL PREDICTIONS
# ============================================================

def update_prediction_history(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add newly eligible official predictions.

    Existing official predictions are immutable.

    A prediction is locked once when it first enters
    the 24-hour window.

    At that exact point we capture the market snapshot.

    Existing records are never recalculated or overwritten.
    """

    print()
    print(
        "Updating official prediction history..."
    )

    existing = load_prediction_history()

    predictions = predictions.copy()

    predictions[
        "cfbd_id"
    ] = pd.to_numeric(
        predictions["cfbd_id"],
        errors="coerce",
    )

    if existing.empty:

        existing_ids: set[int] = set()

    else:

        existing[
            "cfbd_id"
        ] = pd.to_numeric(
            existing["cfbd_id"],
            errors="coerce",
        )

        existing_ids = set(
            existing[
                "cfbd_id"
            ]
            .dropna()
            .astype(int)
        )

    lock_candidates = predictions[
        predictions[
            "prediction_status"
        ] == "LOCK_WINDOW"
    ].copy()

    new_locked = lock_candidates[
        ~lock_candidates[
            "cfbd_id"
        ].isin(existing_ids)
    ].copy()

    if not new_locked.empty:

        print()
        print(
            f"🔒 {len(new_locked):,} "
            "new prediction(s) entering "
            "official lock..."
        )

        locked_rows: list[dict] = []

        for _, prediction in new_locked.iterrows():

            row = prediction.to_dict()

            cfbd_id = int(
                prediction[
                    "cfbd_id"
                ]
            )

            home_team = str(
                prediction[
                    "home_team"
                ]
            )

            away_team = str(
                prediction[
                    "away_team"
                ]
            )

            print()
            print(
                f"Locking: "
                f"{away_team} @ {home_team}"
            )

            snapshot = capture_lock_market(
                game_id=cfbd_id,
                home_team=home_team,
                away_team=away_team,
            )

            row.update(
                snapshot
            )

            row[
                "official_prediction_timestamp"
            ] = utc_timestamp_string()

            row[
                "prediction_status"
            ] = "LOCKED"

            row = (
                add_model_selected_lock_odds(
                    row
                )
            )

            locked_rows.append(
                row
            )

        new_locked = pd.DataFrame(
            locked_rows
        )

        if existing.empty:

            history = new_locked.copy()

        else:

            history = pd.concat(
                [
                    existing,
                    new_locked,
                ],
                ignore_index=True,
            )

        print()
        print(
            f"✓ New official predictions locked: "
            f"{len(new_locked):,}"
        )

    else:

        history = existing.copy()

        if history.empty:

            print(
                "✓ No games are currently inside "
                f"the {LOCK_HOURS}-hour lock window."
            )

        else:

            print(
                "✓ No new predictions entered "
                f"the {LOCK_HOURS}-hour lock window."
            )

    if not history.empty:

        history = history.drop_duplicates(
            subset=["cfbd_id"],
            keep="first",
        )

        history[
            "prediction_status"
        ] = "LOCKED"

        if (
            "official_prediction_timestamp"
            not in history.columns
        ):

            history[
                "official_prediction_timestamp"
            ] = history.get(
                "prediction_timestamp",
                "",
            )

        history["_sort_date"] = (
            pd.to_datetime(
                history[
                    "start_date"
                ],
                errors="coerce",
            )
        )

        history = (
            history
            .sort_values(
                [
                    "_sort_date",
                    "week",
                ]
            )
            .drop(
                columns="_sort_date"
            )
            .reset_index(
                drop=True
            )
        )

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        HISTORY_PATH,
        index=False,
    )

    print()
    print(
        "✓ Official prediction history saved:"
    )

    print(
        f"  {HISTORY_PATH}"
    )

    print(
        f"✓ Total official predictions: "
        f"{len(history):,}"
    )

    return history


# ============================================================
# CURRENT PREDICTION FILE
# ============================================================

def build_current_predictions(
    fresh_predictions: pd.DataFrame,
    official_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine dynamic forecasts with immutable
    official predictions.

    Locked games use their original model prediction
    AND their original lock-time market snapshot.
    """

    current = fresh_predictions.copy()

    if official_history.empty:
        return current

    official = official_history.copy()

    official[
        "cfbd_id"
    ] = pd.to_numeric(
        official["cfbd_id"],
        errors="coerce",
    )

    official_columns = [
        "cfbd_id",

        # Official model values
        "home_win_probability",
        "away_win_probability",
        "predicted_winner",
        "prediction_probability",
        "confidence",
        "prediction_timestamp",
        "official_prediction_timestamp",
        "prediction_status",

        # Frozen market snapshot
        "lock_market_captured",
        "lock_market_timestamp",
        "lock_provider",
        "lock_home_moneyline",
        "lock_away_moneyline",
        "lock_moneyline",
        "lock_moneyline_odds",
        "lock_spread",
        "lock_spread_favourite",
        "lock_over_under",
        "bet_side",
    ]

    official_columns = [
        column
        for column in official_columns
        if column in official.columns
    ]

    official = official[
        official_columns
    ].copy()

    official = official.drop_duplicates(
        subset=["cfbd_id"],
        keep="first",
    )

    rename_map = {}

    for column in official.columns:

        if column == "cfbd_id":
            continue

        rename_map[
            column
        ] = f"official_{column}"

    official = official.rename(
        columns=rename_map
    )

    current = current.merge(
        official,
        on="cfbd_id",
        how="left",
    )

    locked_mask = (
        current[
            "official_prediction_status"
        ].notna()
    )

    # --------------------------------------------------------
    # Immutable official model fields
    # --------------------------------------------------------

    model_columns = [
        "home_win_probability",
        "away_win_probability",
        "predicted_winner",
        "prediction_probability",
        "confidence",
        "prediction_timestamp",
    ]

    for column in model_columns:

        official_column = (
            f"official_{column}"
        )

        if (
            official_column
            not in current.columns
        ):
            continue

        current.loc[
            locked_mask,
            column,
        ] = current.loc[
            locked_mask,
            official_column,
        ]

    # --------------------------------------------------------
    # Immutable lock-market fields
    # --------------------------------------------------------

    market_columns = [
        "lock_market_captured",
        "lock_market_timestamp",
        "lock_provider",
        "lock_home_moneyline",
        "lock_away_moneyline",
        "lock_moneyline",
        "lock_moneyline_odds",
        "lock_spread",
        "lock_spread_favourite",
        "lock_over_under",
        "bet_side",
    ]

    for column in market_columns:

        official_column = (
            f"official_{column}"
        )

        if (
            official_column
            not in current.columns
        ):
            continue

        current.loc[
            locked_mask,
            column,
        ] = current.loc[
            locked_mask,
            official_column,
        ]

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    current[
        "prediction_status"
    ] = "FORECAST"

    current.loc[
        locked_mask,
        "prediction_status",
    ] = "LOCKED"

    # Preserve official lock timestamp.
    if (
        "official_official_prediction_timestamp"
        in current.columns
    ):

        current[
            "official_prediction_timestamp"
        ] = current[
            "official_official_prediction_timestamp"
        ]

    # --------------------------------------------------------
    # Remove merge helper columns
    # --------------------------------------------------------

    temporary_columns = [
        column
        for column in current.columns
        if column.startswith(
            "official_"
        )
    ]

    current = current.drop(
        columns=temporary_columns,
        errors="ignore",
    )

    current = current.sort_values(
        [
            "week",
            "start_date",
            "prediction_probability",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return current


def save_current_predictions(
    predictions: pd.DataFrame,
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "✓ Current predictions saved:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print(
        f"  Upcoming games: "
        f"{len(predictions):,}"
    )

    if (
        "prediction_status"
        in predictions.columns
    ):

        status_counts = (
            predictions[
                "prediction_status"
            ]
            .value_counts()
            .to_dict()
        )

        forecast_count = (
            status_counts.get(
                "FORECAST",
                0,
            )
        )

        locked_count = (
            status_counts.get(
                "LOCKED",
                0,
            )
        )

        print(
            f"  Dynamic forecasts: "
            f"{forecast_count:,}"
        )

        print(
            f"  Official locked: "
            f"{locked_count:,}"
        )


# ============================================================
# DISPLAY
# ============================================================

def display_predictions(
    predictions: pd.DataFrame,
) -> None:

    print()
    print_header(
        "CURRENT 2026 PREDICTIONS"
    )

    if predictions.empty:

        print(
            "No predictions available."
        )

        return

    display_df = predictions[
        [
            "week",
            "home_team",
            "away_team",
            "home_win_probability",
            "away_win_probability",
            "predicted_winner",
            "confidence",
            "prediction_status",
            "hours_to_kickoff",
        ]
    ].copy()

    display_df[
        "home_win_probability"
    ] = (
        display_df[
            "home_win_probability"
        ]
        .astype(float)
        .mul(100)
        .round(1)
    )

    display_df[
        "away_win_probability"
    ] = (
        display_df[
            "away_win_probability"
        ]
        .astype(float)
        .mul(100)
        .round(1)
    )

    display_df[
        "hours_to_kickoff"
    ] = (
        display_df[
            "hours_to_kickoff"
        ]
        .round(1)
    )

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "CFB PREDICTION CENTRE — "
        "DYNAMIC FORECAST + OFFICIAL LOCK"
    )

    print()
    print(
        f"Official prediction lock window: "
        f"{LOCK_HOURS} hours before kickoff"
    )

    print()
    print(
        "Prediction universe: "
        "FBS-involved games"
    )

    print()
    print(
        "Lock-time market capture: "
        + (
            "ENABLED"
            if CAPTURE_LOCK_MARKET
            else "DISABLED"
        )
    )

    model = load_model()

    feature_columns = (
        load_feature_columns()
    )

    session = SessionLocal()

    try:

        # ----------------------------------------------------
        # Historical data
        # ----------------------------------------------------

        print()
        print(
            "Loading completed-game history..."
        )

        engine = TeamFormEngine(
            session
        )

        history_games = (
            engine.load_games()
        )

        history_stats = (
            engine.load_team_stats()
        )

        if history_games.empty:

            raise RuntimeError(
                "No completed games found."
            )

        if history_stats.empty:

            raise RuntimeError(
                "No team statistics found."
            )

        print(
            f"✓ Completed games: "
            f"{len(history_games):,}"
        )

        print(
            f"✓ Team-stat records: "
            f"{len(history_stats):,}"
        )

        print()
        print(
            "Building team history..."
        )

        history = (
            engine.build_team_history(
                stats_df=history_stats,
                games_df=history_games,
            )
        )

        print(
            f"✓ Team history rows: "
            f"{len(history):,}"
        )

        # ----------------------------------------------------
        # Teams / games
        # ----------------------------------------------------

        team_classifications = (
            load_fbs_team_universe(
                session
            )
        )

        upcoming = (
            load_upcoming_games(
                session=session,
                team_classifications=(
                    team_classifications
                ),
            )
        )

        if upcoming.empty:

            print()
            print(
                "No upcoming FBS games found."
            )

            return

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        feature_df = (
            build_prediction_features(
                games=upcoming,
                engine=engine,
                history=history,
            )
        )

        validate_features(
            df=feature_df,
            feature_columns=(
                feature_columns
            ),
        )

        # ----------------------------------------------------
        # Fresh predictions
        # ----------------------------------------------------

        fresh_predictions = (
            generate_predictions(
                feature_df=feature_df,
                model=model,
                feature_columns=(
                    feature_columns
                ),
            )
        )

        # ----------------------------------------------------
        # Official locking
        # ----------------------------------------------------

        official_history = (
            update_prediction_history(
                predictions=fresh_predictions,
            )
        )

        # ----------------------------------------------------
        # Combine dynamic + immutable predictions
        # ----------------------------------------------------

        current_predictions = (
            build_current_predictions(
                fresh_predictions=(
                    fresh_predictions
                ),
                official_history=(
                    official_history
                ),
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_current_predictions(
            current_predictions
        )

        display_predictions(
            current_predictions
        )

        forecast_count = (
            current_predictions[
                "prediction_status"
            ]
            .eq("FORECAST")
            .sum()
        )

        locked_count = (
            current_predictions[
                "prediction_status"
            ]
            .eq("LOCKED")
            .sum()
        )

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "2026 PREDICTION UPDATE COMPLETE"
        )
        print("=" * 70)

        print()
        print(
            f"Dynamic forecasts: "
            f"{forecast_count:,}"
        )

        print(
            f"Official locked: "
            f"{locked_count:,}"
        )

        print()
        print(
            "Future games will continue to "
            "update until they enter the "
            f"{LOCK_HOURS}-hour lock window."
        )

        print()
        print(
            "Once locked, the model prediction "
            "and lock-time market snapshot "
            "are immutable."
        )

        print()
        print(
            "The tracker can now grade the "
            "£100 betting simulation using "
            "the frozen lock-time odds."
        )

    finally:

        session.close()


if __name__ == "__main__":
    main()