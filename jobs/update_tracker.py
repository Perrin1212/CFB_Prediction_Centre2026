from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "2026_prediction_history.csv"
)

TRACKER_FILE = (
    PROJECT_ROOT
    / "data"
    / "2026_prediction_tracker.csv"
)

load_dotenv(
    PROJECT_ROOT / ".env"
)

API_KEY = os.getenv(
    "CFBD_API_KEY"
)

BASE_URL = os.getenv(
    "CFBD_BASE_URL",
    "https://api.collegefootballdata.com",
)

SEASON = 2026

# Standard assumed ATS price when CFBD does not provide
# a separate spread price.
ATS_AMERICAN_ODDS = -110

# Standard simulation stake.
BET_STAKE = 100.0


# ============================================================
# HELPERS
# ============================================================

def normalise_team_name(name: str) -> str:

    if pd.isna(name):
        return ""

    name = str(name).strip().lower()

    replacements = {
        "hawai'i": "hawaii",
        "hawaiʻi": "hawaii",
        "st. thomas (mn)": "st thomas mn",
        "ut rio grande valley": "utrgv",
        "ut permian basin": "ut permian basin",
        "se louisiana": "se louisiana",
        "southeast missouri state": "southeast missouri state",
        "miami (oh)": "miami ohio",
        "miami (fl)": "miami florida",
        "nc state": "nc state",
        "uconn": "connecticut",
        "uab": "uab",
    }

    return replacements.get(
        name,
        name,
    )


def is_missing(value) -> bool:

    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def safe_float(value) -> float | None:

    if is_missing(value):
        return None

    try:

        if isinstance(value, str):
            value = (
                value
                .strip()
                .replace("%", "")
                .replace("£", "")
                .replace(",", "")
            )

        number = float(value)

        if pd.isna(number):
            return None

        return number

    except (
        TypeError,
        ValueError,
    ):
        return None


def parse_probability(value) -> float:

    number = safe_float(value)

    if number is None:
        return 0.0

    if number > 1:
        number = number / 100

    return max(
        0.0,
        min(
            1.0,
            number,
        ),
    )


def confidence_bucket(
    probability: float,
) -> str:

    probability = parse_probability(
        probability
    )

    if probability >= 0.90:
        return "Elite"

    if probability >= 0.80:
        return "Very Strong"

    if probability >= 0.70:
        return "Strong"

    if probability >= 0.60:
        return "Moderate"

    if probability >= 0.55:
        return "Lean"

    return "Coin Flip"


def american_odds_profit(
    odds: float,
    stake: float = BET_STAKE,
) -> float:

    """
    Calculate profit on an American-odds bet.

    +150:
        £100 stake -> £150 profit

    -150:
        £100 stake -> £66.67 profit
    """

    if odds > 0:

        return stake * (
            odds / 100.0
        )

    if odds < 0:

        return stake * (
            100.0 / abs(odds)
        )

    return 0.0


def calculate_moneyline_profit(
    odds,
    bet_won,
    bet_push=False,
) -> float | None:

    odds = safe_float(odds)

    if odds is None:
        return None

    if bet_push:
        return 0.0

    if bet_won is True:
        return round(
            american_odds_profit(
                odds
            ),
            2,
        )

    if bet_won is False:
        return -BET_STAKE

    return None


def calculate_ats_profit(
    ats_result: str | None,
) -> float | None:

    if ats_result is None:
        return None

    result = str(
        ats_result
    ).strip().upper()

    if result == "WIN":
        return round(
            american_odds_profit(
                ATS_AMERICAN_ODDS
            ),
            2,
        )

    if result == "LOSS":
        return -BET_STAKE

    if result == "PUSH":
        return 0.0

    return None


# ============================================================
# LOAD PREDICTION HISTORY
# ============================================================

def get_prediction_history() -> pd.DataFrame:

    if not HISTORY_FILE.exists():

        raise FileNotFoundError(
            "Prediction history not found:\n"
            f"{HISTORY_FILE}\n\n"
            "Run jobs.predict_2026 first."
        )

    df = pd.read_csv(
        HISTORY_FILE
    )

    if df.empty:

        raise RuntimeError(
            "Prediction history is empty."
        )

    required = [
        "cfbd_id",
        "week",
        "start_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "away_win_probability",
        "predicted_winner",
        "prediction_probability",
        "confidence",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns in prediction history: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Normalise game ID.
    # --------------------------------------------------------

    df["cfbd_id"] = pd.to_numeric(
        df["cfbd_id"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Probability columns.
    # --------------------------------------------------------

    for column in [
        "home_win_probability",
        "away_win_probability",
        "prediction_probability",
    ]:

        df[column] = df[column].apply(
            parse_probability
        )

    return df


# ============================================================
# CFBD COMPLETED GAMES
# ============================================================

def get_completed_games() -> pd.DataFrame:

    if not API_KEY:

        raise RuntimeError(
            "CFBD_API_KEY not found in .env"
        )

    print()
    print(
        "Retrieving completed 2026 games..."
    )

    url = (
        f"{BASE_URL}/games"
    )

    headers = {
        "Authorization":
            f"Bearer {API_KEY}"
    }

    params = {
        "year": SEASON,
        "seasonType": "regular",
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    rows = []

    for game in data:

        if not game.get(
            "completed",
            False,
        ):
            continue

        cfbd_id = game.get(
            "id"
        )

        home_team = game.get(
            "homeTeam"
        )

        away_team = game.get(
            "awayTeam"
        )

        home_points = game.get(
            "homePoints"
        )

        away_points = game.get(
            "awayPoints"
        )

        if (
            cfbd_id is None
            or home_team is None
            or away_team is None
        ):
            continue

        rows.append(
            {
                "cfbd_id": cfbd_id,
                "home_team": home_team,
                "away_team": away_team,
                "final_home_score": home_points,
                "final_away_score": away_points,
            }
        )

    games = pd.DataFrame(
        rows
    )

    if not games.empty:

        games["cfbd_id"] = pd.to_numeric(
            games["cfbd_id"],
            errors="coerce",
        )

        games["final_home_score"] = pd.to_numeric(
            games["final_home_score"],
            errors="coerce",
        ).astype("Int64")

        games["final_away_score"] = pd.to_numeric(
            games["final_away_score"],
            errors="coerce",
        ).astype("Int64")

    print(
        f"✓ Completed games retrieved: "
        f"{len(games):,}"
    )

    return games


# ============================================================
# INITIALISE RESULT COLUMNS
# ============================================================

def initialise_result_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    string_columns = [
        "actual_winner",
        "ats_result",
        "moneyline_result",
        "moneyline_bet_side",
        "bet_side",
        "lock_provider",
        "model_vs_market",
        "confidence_bucket",
    ]

    for column in string_columns:

        if column not in df.columns:

            df[column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="string",
            )

        else:

            df[column] = (
                df[column]
                .astype("string")
            )

    integer_columns = [
        "final_home_score",
        "final_away_score",
    ]

    for column in integer_columns:

        if column not in df.columns:

            df[column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="Int64",
            )

        else:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype("Int64")

    boolean_columns = [
        "prediction_correct",
        "model_agrees_with_market",
        "moneyline_bet_valid",
        "ats_bet_valid",
    ]

    for column in boolean_columns:

        if column not in df.columns:

            df[column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="boolean",
            )

        else:

            df[column] = (
                df[column]
                .astype("string")
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                    }
                )
                .astype("boolean")
            )

    numeric_columns = [
        "moneyline_odds",
        "moneyline_stake",
        "moneyline_profit_loss",
        "moneyline_return",
        "ats_spread",
        "ats_stake",
        "ats_profit_loss",
        "ats_return",
        "cumulative_moneyline_profit",
        "cumulative_ats_profit",
    ]

    for column in numeric_columns:

        if column not in df.columns:

            df[column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="Float64",
            )

        else:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype("Float64")

    return df


# ============================================================
# GRADE GAME RESULTS
# ============================================================

def grade_predictions(
    predictions: pd.DataFrame,
    completed_games: pd.DataFrame,
) -> pd.DataFrame:

    df = initialise_result_columns(
        predictions
    )

    if completed_games.empty:
        return df

    completed_games = completed_games.copy()

    completed_games["cfbd_id"] = pd.to_numeric(
        completed_games["cfbd_id"],
        errors="coerce",
    )

    completed_games["final_home_score"] = pd.to_numeric(
        completed_games["final_home_score"],
        errors="coerce",
    ).astype("Int64")

    completed_games["final_away_score"] = pd.to_numeric(
        completed_games["final_away_score"],
        errors="coerce",
    ).astype("Int64")

    result_lookup = (
        completed_games
        .drop_duplicates(
            subset=["cfbd_id"],
            keep="last",
        )
        .set_index("cfbd_id")
    )

    for index, row in df.iterrows():

        cfbd_id = row["cfbd_id"]

        if pd.isna(cfbd_id):
            continue

        if cfbd_id not in result_lookup.index:
            continue

        result = result_lookup.loc[
            cfbd_id
        ]

        home_score = result[
            "final_home_score"
        ]

        away_score = result[
            "final_away_score"
        ]

        if (
            pd.isna(home_score)
            or pd.isna(away_score)
        ):
            continue

        home_score_num = int(
            home_score
        )

        away_score_num = int(
            away_score
        )

        home_team = str(
            row["home_team"]
        ).strip()

        away_team = str(
            row["away_team"]
        ).strip()

        df.at[
            index,
            "final_home_score"
        ] = home_score_num

        df.at[
            index,
            "final_away_score"
        ] = away_score_num

        # ----------------------------------------------------
        # Actual winner.
        # ----------------------------------------------------

        if home_score_num > away_score_num:

            actual_winner = home_team

        elif away_score_num > home_score_num:

            actual_winner = away_team

        else:

            actual_winner = "Tie"

        df.at[
            index,
            "actual_winner"
        ] = actual_winner

        # ----------------------------------------------------
        # Model result.
        # ----------------------------------------------------

        predicted_winner = str(
            row["predicted_winner"]
        ).strip()

        if actual_winner == "Tie":

            df.at[
                index,
                "prediction_correct"
            ] = pd.NA

        elif (
            normalise_team_name(
                predicted_winner
            )
            ==
            normalise_team_name(
                actual_winner
            )
        ):

            df.at[
                index,
                "prediction_correct"
            ] = True

        else:

            df.at[
                index,
                "prediction_correct"
            ] = False

    return df


# ============================================================
# LOCK-TIME MARKET NORMALISATION
# ============================================================

def prepare_lock_market(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Ensure lock-time columns exist.
    #
    # These MUST come from predict_2026.py.
    # We deliberately do NOT query current betting lines here.
    # --------------------------------------------------------

    lock_columns = [
        "lock_market_captured",
        "lock_market_timestamp",
        "lock_provider",
        "lock_home_moneyline",
        "lock_away_moneyline",
        "lock_spread",
        "lock_spread_favourite",
        "lock_over_under",
        "lock_moneyline",
        "lock_moneyline_odds",
        "bet_side",
    ]

    for column in lock_columns:

        if column not in df.columns:

            if column == "lock_market_captured":

                df[column] = False

            else:

                df[column] = pd.NA

    # --------------------------------------------------------
    # Numeric lock market fields.
    # --------------------------------------------------------

    for column in [
        "lock_home_moneyline",
        "lock_away_moneyline",
        "lock_spread",
        "lock_over_under",
        "lock_moneyline",
        "lock_moneyline_odds",
    ]:

        df[column] = df[column].apply(
            safe_float
        )

    # --------------------------------------------------------
    # Selected model moneyline.
    #
    # Prefer the value explicitly captured by predict_2026.py.
    # If it isn't present, derive it ONLY from the frozen
    # lock-time home/away moneyline fields.
    # --------------------------------------------------------

    for index, row in df.iterrows():

        existing_odds = safe_float(
            row.get(
                "lock_moneyline_odds"
            )
        )

        if existing_odds is not None:
            continue

        predicted_winner = str(
            row["predicted_winner"]
        ).strip()

        home_team = str(
            row["home_team"]
        ).strip()

        away_team = str(
            row["away_team"]
        ).strip()

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

        if (
            normalise_team_name(
                predicted_winner
            )
            ==
            normalise_team_name(
                home_team
            )
        ):

            selected_odds = home_ml
            bet_side = "HOME"

        elif (
            normalise_team_name(
                predicted_winner
            )
            ==
            normalise_team_name(
                away_team
            )
        ):

            selected_odds = away_ml
            bet_side = "AWAY"

        if selected_odds is not None:

            df.at[
                index,
                "lock_moneyline_odds"
            ] = selected_odds

            df.at[
                index,
                "lock_moneyline"
            ] = selected_odds

            df.at[
                index,
                "bet_side"
            ] = bet_side

    return df


# ============================================================
# MODEL VS LOCK-TIME MARKET
# ============================================================

def calculate_market_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    for index, row in df.iterrows():

        market_favourite = row.get(
            "lock_spread_favourite"
        )

        predicted_winner = str(
            row["predicted_winner"]
        ).strip()

        if (
            is_missing(
                market_favourite
            )
            or str(
                market_favourite
            ).strip() == ""
        ):

            df.at[
                index,
                "model_vs_market"
            ] = "No Lock Line"

            df.at[
                index,
                "model_agrees_with_market"
            ] = pd.NA

            continue

        agrees = (
            normalise_team_name(
                predicted_winner
            )
            ==
            normalise_team_name(
                str(
                    market_favourite
                ).strip()
            )
        )

        df.at[
            index,
            "model_agrees_with_market"
        ] = bool(agrees)

        df.at[
            index,
            "model_vs_market"
        ] = (
            "Agrees"
            if agrees
            else "DISAGREES"
        )

    return df


# ============================================================
# MONEYLINE BETTING SIMULATION
# ============================================================

def calculate_moneyline_betting(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    for index, row in df.iterrows():

        odds = safe_float(
            row.get(
                "lock_moneyline_odds"
            )
        )

        actual_winner = row.get(
            "actual_winner"
        )

        predicted_winner = str(
            row["predicted_winner"]
        ).strip()

        # ----------------------------------------------------
        # No genuine lock-time odds = no simulated bet.
        # ----------------------------------------------------

        if odds is None:

            df.at[
                index,
                "moneyline_bet_valid"
            ] = False

            df.at[
                index,
                "moneyline_result"
            ] = "No Lock Odds"

            continue

        # ----------------------------------------------------
        # We have odds, so the £100 bet is valid.
        # ----------------------------------------------------

        df.at[
            index,
            "moneyline_bet_valid"
        ] = True

        df.at[
            index,
            "moneyline_stake"
        ] = BET_STAKE

        df.at[
            index,
            "moneyline_odds"
        ] = odds

        df.at[
            index,
            "moneyline_bet_side"
        ] = row.get(
            "bet_side"
        )

        # ----------------------------------------------------
        # Game not completed yet.
        # ----------------------------------------------------

        if (
            is_missing(
                actual_winner
            )
            or str(
                actual_winner
            ).strip() == ""
        ):

            df.at[
                index,
                "moneyline_result"
            ] = "Pending"

            continue

        actual_winner_text = str(
            actual_winner
        ).strip()

        # ----------------------------------------------------
        # Tie = no winner bet / push.
        # ----------------------------------------------------

        if actual_winner_text == "Tie":

            df.at[
                index,
                "moneyline_result"
            ] = "Push"

            df.at[
                index,
                "moneyline_profit_loss"
            ] = 0.0

            df.at[
                index,
                "moneyline_return"
            ] = BET_STAKE

            continue

        # ----------------------------------------------------
        # Win.
        # ----------------------------------------------------

        if (
            normalise_team_name(
                predicted_winner
            )
            ==
            normalise_team_name(
                actual_winner_text
            )
        ):

            profit = american_odds_profit(
                odds
            )

            df.at[
                index,
                "moneyline_result"
            ] = "WIN"

            df.at[
                index,
                "moneyline_profit_loss"
            ] = round(
                profit,
                2,
            )

            df.at[
                index,
                "moneyline_return"
            ] = round(
                BET_STAKE + profit,
                2,
            )

        # ----------------------------------------------------
        # Loss.
        # ----------------------------------------------------

        else:

            df.at[
                index,
                "moneyline_result"
            ] = "LOSS"

            df.at[
                index,
                "moneyline_profit_loss"
            ] = -BET_STAKE

            df.at[
                index,
                "moneyline_return"
            ] = 0.0

    return df


# ============================================================
# ATS GRADING
# ============================================================

def calculate_ats_betting(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    for index, row in df.iterrows():

        spread = safe_float(
            row.get(
                "lock_spread"
            )
        )

        actual_winner = row.get(
            "actual_winner"
        )

        predicted_winner = str(
            row["predicted_winner"]
        ).strip()

        home_team = str(
            row["home_team"]
        ).strip()

        away_team = str(
            row["away_team"]
        ).strip()

        # ----------------------------------------------------
        # No frozen spread.
        # ----------------------------------------------------

        if spread is None:

            df.at[
                index,
                "ats_bet_valid"
            ] = False

            df.at[
                index,
                "ats_result"
            ] = "No Lock Spread"

            continue

        df.at[
            index,
            "ats_bet_valid"
        ] = True

        df.at[
            index,
            "ats_spread"
        ] = spread

        df.at[
            index,
            "ats_stake"
        ] = BET_STAKE

        # ----------------------------------------------------
        # Pending game.
        # ----------------------------------------------------

        if (
            is_missing(
                actual_winner
            )
            or str(
                actual_winner
            ).strip() == ""
        ):

            df.at[
                index,
                "ats_result"
            ] = "Pending"

            continue

        # ----------------------------------------------------
        # Need final scores.
        # ----------------------------------------------------

        home_score = safe_float(
            row.get(
                "final_home_score"
            )
        )

        away_score = safe_float(
            row.get(
                "final_away_score"
            )
        )

        if (
            home_score is None
            or away_score is None
        ):

            df.at[
                index,
                "ats_result"
            ] = "Pending"

            continue

        # ----------------------------------------------------
        # Convert model pick into a side.
        # ----------------------------------------------------

        if (
            normalise_team_name(
                predicted_winner
            )
            ==
            normalise_team_name(
                home_team
            )
        ):

            model_picked_home = True

        elif (
            normalise_team_name(
                predicted_winner
            )
            ==
            normalise_team_name(
                away_team
            )
        ):

            model_picked_home = False

        else:

            df.at[
                index,
                "ats_result"
            ] = "No Model Side"

            continue

        # ----------------------------------------------------
        # Spread is signed relative to home team.
        #
        # Example:
        #
        # Home -7:
        #     home_score - away_score - 7
        #
        # Home +7:
        #     home_score - away_score + 7
        # ----------------------------------------------------

        adjusted_home_margin = (
            home_score
            - away_score
            + spread
        )

        # ----------------------------------------------------
        # Model picked home.
        # ----------------------------------------------------

        if model_picked_home:

            if adjusted_home_margin > 0:

                result = "WIN"

            elif adjusted_home_margin < 0:

                result = "LOSS"

            else:

                result = "PUSH"

        # ----------------------------------------------------
        # Model picked away.
        # ----------------------------------------------------

        else:

            if adjusted_home_margin < 0:

                result = "WIN"

            elif adjusted_home_margin > 0:

                result = "LOSS"

            else:

                result = "PUSH"

        df.at[
            index,
            "ats_result"
        ] = result

        profit = calculate_ats_profit(
            result
        )

        if profit is not None:

            df.at[
                index,
                "ats_profit_loss"
            ] = profit

            df.at[
                index,
                "ats_return"
            ] = round(
                BET_STAKE + profit,
                2,
            )

    return df


# ============================================================
# CUMULATIVE PROFIT
# ============================================================

def calculate_cumulative_profit(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Sort by kickoff date first.
    # --------------------------------------------------------

    df["_sort_date"] = pd.to_datetime(
        df["start_date"],
        errors="coerce",
    )

    df = df.sort_values(
        [
            "_sort_date",
            "week",
            "cfbd_id",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Moneyline cumulative P/L.
    # --------------------------------------------------------

    moneyline_running = 0.0
    moneyline_values = []

    for value in df[
        "moneyline_profit_loss"
    ]:

        number = safe_float(
            value
        )

        if number is not None:
            moneyline_running += number

        moneyline_values.append(
            round(
                moneyline_running,
                2,
            )
        )

    df[
        "cumulative_moneyline_profit"
    ] = moneyline_values

    # --------------------------------------------------------
    # ATS cumulative P/L.
    # --------------------------------------------------------

    ats_running = 0.0
    ats_values = []

    for value in df[
        "ats_profit_loss"
    ]:

        number = safe_float(
            value
        )

        if number is not None:
            ats_running += number

        ats_values.append(
            round(
                ats_running,
                2,
            )
        )

    df[
        "cumulative_ats_profit"
    ] = ats_values

    df = df.drop(
        columns=[
            "_sort_date"
        ]
    )

    return df


# ============================================================
# PRESERVE LEGACY TRACKER DATA
# ============================================================

def preserve_existing_tracker(
    tracker: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:

    if previous.empty:
        return tracker

    if "cfbd_id" not in previous.columns:
        return tracker

    if "cfbd_id" not in tracker.columns:
        return tracker

    previous = previous.copy()

    previous["cfbd_id"] = pd.to_numeric(
        previous["cfbd_id"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Only preserve old fields that are NOT supposed to be
    # recalculated from the immutable lock-time snapshot.
    #
    # The new lock-time betting fields intentionally do NOT
    # come from the previous tracker.
    # --------------------------------------------------------

    preserve_columns = [
        "provider",
        "spread",
        "spread_favourite",
        "home_moneyline",
        "away_moneyline",
        "over_under",
    ]

    available_columns = [
        column
        for column in preserve_columns
        if column in previous.columns
    ]

    if not available_columns:
        return tracker

    previous = previous[
        [
            "cfbd_id"
        ]
        + available_columns
    ]

    previous = previous.drop_duplicates(
        subset=[
            "cfbd_id"
        ],
        keep="last",
    )

    tracker = tracker.merge(
        previous,
        on="cfbd_id",
        how="left",
        suffixes=(
            "",
            "_previous",
        ),
    )

    for column in available_columns:

        previous_column = (
            f"{column}_previous"
        )

        if previous_column not in tracker.columns:
            continue

        if column not in tracker.columns:

            tracker[column] = tracker[
                previous_column
            ]

        else:

            current = tracker[
                column
            ]

            current = current.replace(
                "",
                pd.NA,
            )

            tracker[column] = (
                current
                .fillna(
                    tracker[
                        previous_column
                    ]
                )
            )

        tracker = tracker.drop(
            columns=[
                previous_column
            ]
        )

    return tracker


# ============================================================
# BUILD TRACKER
# ============================================================

def build_tracker() -> pd.DataFrame:

    predictions = (
        get_prediction_history()
    )

    completed_games = (
        get_completed_games()
    )

    # --------------------------------------------------------
    # Previous tracker is only used for legacy market fields.
    # --------------------------------------------------------

    if TRACKER_FILE.exists():

        try:

            previous_tracker = pd.read_csv(
                TRACKER_FILE
            )

        except Exception as error:

            print(
                "⚠ Could not read previous tracker: "
                f"{error}"
            )

            previous_tracker = (
                pd.DataFrame()
            )

    else:

        previous_tracker = (
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # Grade games.
    # --------------------------------------------------------

    tracker = grade_predictions(
        predictions,
        completed_games,
    )

    # --------------------------------------------------------
    # Confidence.
    # --------------------------------------------------------

    tracker[
        "confidence_bucket"
    ] = tracker[
        "prediction_probability"
    ].apply(
        confidence_bucket
    )

    # --------------------------------------------------------
    # Preserve legacy market fields for compatibility.
    #
    # These are NOT used for the new betting simulation.
    # --------------------------------------------------------

    tracker = preserve_existing_tracker(
        tracker,
        previous_tracker,
    )

    # --------------------------------------------------------
    # Prepare frozen lock-time market.
    # --------------------------------------------------------

    tracker = prepare_lock_market(
        tracker
    )

    # --------------------------------------------------------
    # Compare model against the market that existed at lock.
    # --------------------------------------------------------

    tracker = calculate_market_comparison(
        tracker
    )

    # --------------------------------------------------------
    # £100 MONEYLINE SIMULATION
    # --------------------------------------------------------

    tracker = calculate_moneyline_betting(
        tracker
    )

    # --------------------------------------------------------
    # £100 ATS SIMULATION
    # --------------------------------------------------------

    tracker = calculate_ats_betting(
        tracker
    )

    # --------------------------------------------------------
    # Cumulative P/L.
    # --------------------------------------------------------

    tracker = calculate_cumulative_profit(
        tracker
    )

    # --------------------------------------------------------
    # Tracker timestamp.
    # --------------------------------------------------------

    tracker[
        "tracker_updated"
    ] = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    # --------------------------------------------------------
    # Preferred column order.
    # --------------------------------------------------------

    preferred_order = [
        # Identity
        "cfbd_id",
        "week",
        "start_date",
        "home_team",
        "away_team",

        # Model
        "predicted_winner",
        "prediction_probability",
        "confidence",
        "confidence_bucket",
        "home_win_probability",
        "away_win_probability",

        # Frozen lock-time market
        "lock_market_captured",
        "lock_market_timestamp",
        "lock_provider",
        "lock_home_moneyline",
        "lock_away_moneyline",
        "lock_spread",
        "lock_spread_favourite",
        "lock_over_under",
        "lock_moneyline",
        "lock_moneyline_odds",
        "bet_side",

        # Model vs market
        "model_vs_market",
        "model_agrees_with_market",

        # Moneyline simulation
        "moneyline_bet_valid",
        "moneyline_bet_side",
        "moneyline_stake",
        "moneyline_odds",
        "moneyline_result",
        "moneyline_profit_loss",
        "moneyline_return",
        "cumulative_moneyline_profit",

        # ATS simulation
        "ats_bet_valid",
        "ats_spread",
        "ats_stake",
        "ats_result",
        "ats_profit_loss",
        "ats_return",
        "cumulative_ats_profit",

        # Result
        "actual_winner",
        "final_home_score",
        "final_away_score",
        "prediction_correct",

        # Legacy/current compatibility fields
        "provider",
        "spread",
        "spread_favourite",
        "home_moneyline",
        "away_moneyline",
        "over_under",

        # Metadata
        "prediction_timestamp",
        "official_prediction_timestamp",
        "prediction_status",
        "tracker_updated",
    ]

    remaining = [
        column
        for column in tracker.columns
        if column not in preferred_order
    ]

    tracker = tracker[
        [
            column
            for column in preferred_order
            if column in tracker.columns
        ]
        + remaining
    ]

    # --------------------------------------------------------
    # Final chronological sort.
    # --------------------------------------------------------

    tracker["_sort_date"] = pd.to_datetime(
        tracker["start_date"],
        errors="coerce",
    )

    tracker = tracker.sort_values(
        [
            "_sort_date",
            "week",
            "cfbd_id",
        ],
        na_position="last",
    ).drop(
        columns="_sort_date"
    )

    tracker = tracker.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    TRACKER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracker.to_csv(
        TRACKER_FILE,
        index=False,
    )

    return tracker


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "CFB PREDICTION TRACKER"
    )
    print("=" * 60)

    try:

        tracker = build_tracker()

    except Exception as error:

        print()
        print(
            "❌ Tracker update failed:"
        )
        print(
            f"{type(error).__name__}: {error}"
        )
        raise

    print()
    print(
        f"Predictions tracked: "
        f"{len(tracker):,}"
    )

    # ========================================================
    # PREDICTION PERFORMANCE
    # ========================================================

    completed_mask = (
        tracker[
            "actual_winner"
        ]
        .astype("string")
        .str.strip()
        .fillna("")
        .ne("")
    )

    completed_count = int(
        completed_mask.sum()
    )

    correct_count = int(
        tracker[
            "prediction_correct"
        ]
        .astype("string")
        .str.lower()
        .eq("true")
        .sum()
    )

    wrong_count = int(
        tracker[
            "prediction_correct"
        ]
        .astype("string")
        .str.lower()
        .eq("false")
        .sum()
    )

    print()
    print(
        f"Completed predictions: "
        f"{completed_count:,}"
    )

    print(
        f"Correct predictions: "
        f"{correct_count:,}"
    )

    print(
        f"Wrong predictions: "
        f"{wrong_count:,}"
    )

    if completed_count > 0:

        accuracy = (
            correct_count
            / completed_count
        ) * 100

        print(
            f"Prediction accuracy: "
            f"{accuracy:.1f}%"
        )

    # ========================================================
    # MONEYLINE PERFORMANCE
    # ========================================================

    valid_ml = (
        tracker[
            "moneyline_bet_valid"
        ]
        .astype("boolean")
        .fillna(False)
    )

    settled_ml = (
        valid_ml
        &
        tracker[
            "moneyline_profit_loss"
        ].notna()
    )

    ml_bets = int(
        settled_ml.sum()
    )

    ml_profit = (
        pd.to_numeric(
            tracker.loc[
                settled_ml,
                "moneyline_profit_loss"
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    ml_staked = (
        ml_bets
        * BET_STAKE
    )

    print()
    print(
        "-" * 60
    )
    print(
        "£100 MONEYLINE SIMULATION"
    )
    print(
        "-" * 60
    )

    print(
        f"Valid lock-time bets: "
        f"{int(valid_ml.sum()):,}"
    )

    print(
        f"Settled bets: "
        f"{ml_bets:,}"
    )

    print(
        f"Total staked: "
        f"£{ml_staked:,.2f}"
    )

    print(
        f"Net profit/loss: "
        f"£{ml_profit:,.2f}"
    )

    if ml_staked > 0:

        ml_roi = (
            ml_profit
            / ml_staked
        ) * 100

        print(
            f"ROI: "
            f"{ml_roi:.2f}%"
        )

    else:

        print(
            "ROI: N/A"
        )

    # ========================================================
    # ATS PERFORMANCE
    # ========================================================

    valid_ats = (
        tracker[
            "ats_bet_valid"
        ]
        .astype("boolean")
        .fillna(False)
    )

    settled_ats = (
        valid_ats
        &
        tracker[
            "ats_profit_loss"
        ].notna()
    )

    ats_bets = int(
        settled_ats.sum()
    )

    ats_profit = (
        pd.to_numeric(
            tracker.loc[
                settled_ats,
                "ats_profit_loss"
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    ats_staked = (
        ats_bets
        * BET_STAKE
    )

    print()
    print(
        "-" * 60
    )
    print(
        "£100 ATS SIMULATION"
    )
    print(
        "-" * 60
    )

    print(
        f"Valid lock-time spreads: "
        f"{int(valid_ats.sum()):,}"
    )

    print(
        f"Settled bets: "
        f"{ats_bets:,}"
    )

    print(
        f"Total staked: "
        f"£{ats_staked:,.2f}"
    )

    print(
        f"Net profit/loss: "
        f"£{ats_profit:,.2f}"
    )

    if ats_staked > 0:

        ats_roi = (
            ats_profit
            / ats_staked
        ) * 100

        print(
            f"ROI: "
            f"{ats_roi:.2f}%"
        )

    else:

        print(
            "ROI: N/A"
        )

    # ========================================================
    # LOCK-TIME MARKET COVERAGE
    # ========================================================

    captured_count = int(
        tracker[
            "lock_market_captured"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    print()
    print(
        "-" * 60
    )
    print(
        "LOCK-TIME MARKET COVERAGE"
    )
    print(
        "-" * 60
    )

    print(
        f"Predictions with frozen market: "
        f"{captured_count:,}"
    )

    print(
        f"Predictions without frozen market: "
        f"{len(tracker) - captured_count:,}"
    )

    print()
    print(
        "Tracker saved to:"
    )

    print(
        TRACKER_FILE
    )