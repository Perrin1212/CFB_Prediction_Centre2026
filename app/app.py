
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="CFB Prediction Centre 2026",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "2026_predictions.csv"
)

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


# ============================================================
# PREMIUM SPORTS UI
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(42, 94, 145, 0.22),
                transparent 32%
            ),
            radial-gradient(
                circle at 10% 20%,
                rgba(31, 57, 91, 0.20),
                transparent 30%
            ),
            linear-gradient(
                180deg,
                #07111f 0%,
                #0a1525 45%,
                #060d18 100%
            );
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #07101c;
        border-right: 1px solid rgba(110, 150, 190, 0.18);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        color: #f5f8fc !important;
    }

    p, label {
        color: #b9c7d8;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] h2 {
        color: #f5f8fc !important;
        font-weight: 800;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #b9c7d8;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: 10px;
        border: 1px solid rgba(76, 157, 230, 0.38);
        background: #102945;
        color: #f5f9ff;
        font-weight: 750;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #4fa9f5;
        background: #15385d;
        color: white;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(17, 36, 61, 0.96),
                rgba(9, 22, 39, 0.96)
            );
        border: 1px solid rgba(104, 143, 181, 0.20);
        border-radius: 15px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #8fa4bb !important;
    }

    [data-testid="stMetricValue"] {
        color: #f4f8fd !important;
    }


    /* ========================================================
       DATAFRAMES
       ======================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-box {
        background:
            linear-gradient(
                135deg,
                rgba(17, 46, 78, 0.98),
                rgba(8, 21, 37, 0.98)
            );
        border: 1px solid rgba(78, 157, 228, 0.28);
        border-radius: 22px;
        padding: 34px;
        margin-bottom: 22px;
        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.25);
    }

    .hero-kicker {
        color: #59b4ff;
        font-size: 13px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        margin-bottom: 8px;
    }

    .hero-heading {
        color: #ffffff;
        font-size: 42px;
        line-height: 1.05;
        font-weight: 850;
        margin-bottom: 12px;
    }

    .hero-description {
        color: #aebed0;
        font-size: 16px;
        line-height: 1.6;
        max-width: 850px;
    }


    /* ========================================================
       SECTION TEXT
       ======================================================== */

    .section-heading {
        color: #f3f7fb;
        font-size: 25px;
        font-weight: 820;
        margin-top: 25px;
        margin-bottom: 5px;
    }

    .section-subheading {
        color: #8295aa;
        font-size: 14px;
        margin-bottom: 17px;
    }


    /* ========================================================
       GAME VISUAL DIVIDER
       ======================================================== */

    .blue-line {
        height: 4px;
        width: 100%;
        border-radius: 999px;
        background:
            linear-gradient(
                90deg,
                #173b61,
                #55b5ff,
                #d6ad55,
                #55b5ff,
                #173b61
            );
        margin: 10px 0 25px 0;
    }


    /* ========================================================
       INFO PANELS
       ======================================================== */

    .info-panel {
        background:
            linear-gradient(
                145deg,
                rgba(16, 34, 56, 0.96),
                rgba(8, 19, 33, 0.96)
            );
        border: 1px solid rgba(104, 143, 181, 0.20);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .info-title {
        color: #f4f8fd;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .info-text {
        color: #94a9bd;
        font-size: 14px;
        line-height: 1.5;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 800px) {

        .block-container {
            padding: 1rem;
        }

        .hero-box {
            padding: 23px;
        }

        .hero-heading {
            font-size: 30px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    return df


def first_value(
    row: pd.Series,
    columns: list[str],
    default: Any = None,
) -> Any:

    for column in columns:

        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return default


def parse_probability(value: Any) -> float | None:

    if value is None:
        return None

    try:

        value = float(value)

        if value > 1:
            value /= 100

        return max(0.0, min(1.0, value))

    except (TypeError, ValueError):

        return None


def probability(row: pd.Series) -> float | None:

    return parse_probability(
        first_value(
            row,
            [
                "prediction_probability",
                "predicted_probability",
                "model_probability",
                "win_probability",
            ],
        )
    )


def probability_text(value: Any) -> str:

    probability_value = parse_probability(value)

    if probability_value is None:
        return "—"

    return f"{probability_value:.1%}"


def confidence_label(value: Any) -> str:

    probability_value = parse_probability(value)

    if probability_value is None:
        return "Unknown"

    if probability_value >= 0.70:
        return "High"

    if probability_value >= 0.55:
        return "Medium"

    return "Low"


def home_team(row: pd.Series) -> str:

    return str(
        first_value(
            row,
            ["home_team", "home", "homeTeam"],
            "Home",
        )
    )


def away_team(row: pd.Series) -> str:

    return str(
        first_value(
            row,
            ["away_team", "away", "awayTeam"],
            "Away",
        )
    )


def prediction(row: pd.Series) -> str:

    return str(
        first_value(
            row,
            [
                "predicted_winner",
                "prediction",
                "model_pick",
                "pick",
            ],
            "—",
        )
    )


def game_date(row: pd.Series) -> Any:

    return first_value(
        row,
        [
            "start_date",
            "start_date_utc",
            "startDate",
            "game_date",
            "date",
        ],
    )


def game_id(row: pd.Series) -> Any:

    return first_value(
        row,
        [
            "cfbd_id",
            "game_id",
            "id",
        ],
    )


def prediction_status(row: pd.Series) -> str:

    return str(
        first_value(
            row,
            [
                "prediction_status",
                "status",
            ],
            "FORECAST",
        )
    ).upper()


def result_text(row: pd.Series) -> str:

    result = first_value(
        row,
        [
            "prediction_result",
            "result",
            "outcome",
            "grade",
        ],
    )

    if result is not None:
        return str(result)

    correct = first_value(
        row,
        [
            "correct",
            "prediction_correct",
            "is_correct",
        ],
    )

    if correct is not None:

        if str(correct).lower() in {
            "true",
            "1",
            "yes",
            "correct",
        }:
            return "Correct"

        if str(correct).lower() in {
            "false",
            "0",
            "no",
            "wrong",
        }:
            return "Wrong"

    return "Pending"


def result_is_correct(row: pd.Series) -> bool | None:

    result = result_text(row).lower()

    if result in {
        "correct",
        "win",
        "won",
        "w",
        "right",
    }:
        return True

    if result in {
        "wrong",
        "loss",
        "lost",
        "l",
        "incorrect",
    }:
        return False

    return None


def official_locked(row: pd.Series) -> bool:

    return prediction_status(row) == "LOCKED"


def safe_date_sort(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    output = df.copy()

    output["_sort_date"] = pd.to_datetime(
        output.apply(game_date, axis=1),
        errors="coerce",
        utc=True,
    )

    output = output.sort_values(
        "_sort_date",
        na_position="last",
    )

    return output.drop(
        columns=["_sort_date"],
        errors="ignore",
    )


def normalise_bool(value: Any) -> bool:

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def lock_moneyline(row: pd.Series) -> float | None:

    direct = first_value(
        row,
        [
            "lock_moneyline",
            "lock_moneyline_odds",
            "locked_moneyline",
            "locked_moneyline_odds",
        ],
    )

    try:

        if direct is not None:
            return float(direct)

    except (TypeError, ValueError):
        pass

    pick = prediction(row)

    home = home_team(row)
    away = away_team(row)

    if pick == home:

        value = first_value(
            row,
            ["lock_home_moneyline"],
        )

    elif pick == away:

        value = first_value(
            row,
            ["lock_away_moneyline"],
        )

    else:

        return None

    try:
        return float(value)

    except (TypeError, ValueError):

        return None


def lock_spread(row: pd.Series) -> float | None:

    value = first_value(
        row,
        [
            "lock_spread",
            "locked_spread",
        ],
    )

    try:
        return float(value)

    except (TypeError, ValueError):

        return None


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(ttl=30)
def load_predictions() -> pd.DataFrame:

    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame()

    try:

        return clean_dataframe(
            pd.read_csv(PREDICTIONS_FILE)
        )

    except Exception:

        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_history() -> pd.DataFrame:

    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    try:

        return clean_dataframe(
            pd.read_csv(HISTORY_FILE)
        )

    except Exception:

        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_tracker() -> pd.DataFrame:

    if not TRACKER_FILE.exists():
        return pd.DataFrame()

    try:

        return clean_dataframe(
            pd.read_csv(TRACKER_FILE)
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD
# ============================================================

predictions = load_predictions()
history = load_history()
tracker = load_tracker()

if (
    predictions.empty
    and history.empty
    and tracker.empty
):

    st.error(
        "No prediction data is available. "
        "Run the prediction pipeline first."
    )

    st.stop()


# ============================================================
# CURRENT GAME UNIVERSE
# ============================================================

if not predictions.empty:

    games = predictions.copy()

else:

    games = history.copy()


games = clean_dataframe(games)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if "selected_game_id" not in st.session_state:
    st.session_state.selected_game_id = None


def open_game(game_id_value: Any) -> None:

    st.session_state.selected_game_id = str(
        game_id_value
    )

    st.session_state.page = "🎯 Game Detail"


def back_home() -> None:

    st.session_state.selected_game_id = None
    st.session_state.page = "🏠 Home"


# ============================================================
# SIDEBAR
# ============================================================

NAV_OPTIONS = [
    "🏠 Home",
    "🏫 Team Search",
    "🏈 All Games",
    "📊 Model Tracker",
]

with st.sidebar:

    st.markdown("## 🏈 CFB Prediction Centre")

    st.caption("2026 SEASON")

    st.divider()

    # --------------------------------------------------------
    # IMPORTANT:
    # Game Detail is a contextual page.
    #
    # When a user clicks "View Game", Streamlit reruns the
    # entire script. The sidebar must NOT overwrite the
    # Game Detail session state during that rerun.
    # --------------------------------------------------------

    if st.session_state.page == "🎯 Game Detail":

        page = "🎯 Game Detail"

    else:

        page = st.radio(
            "Navigation",
            NAV_OPTIONS,
            index=NAV_OPTIONS.index(
                st.session_state.page
            ),
            label_visibility="collapsed",
        )

        st.session_state.page = page

    st.divider()

    st.caption("PREDICTION LIFECYCLE")

    st.caption(
        "Forecast → 24-hour lock → "
        "official prediction → result"
    )


# ============================================================
# GAME DETAIL
# ============================================================

if st.session_state.page == "🎯 Game Detail":

    if st.session_state.selected_game_id is None:

        st.warning("No game selected.")

        if st.button("← Back to Home"):

            back_home()
            st.rerun()

        st.stop()

    selected_id = str(
        st.session_state.selected_game_id
    )

    matches = games[
        games.apply(
            lambda row:
                str(game_id(row)) == selected_id,
            axis=1,
        )
    ]

    if matches.empty and not history.empty:

        matches = history[
            history.apply(
                lambda row:
                    str(game_id(row)) == selected_id,
                axis=1,
            )
        ]

    if matches.empty:

        st.error("Game could not be found.")

        if st.button("← Back"):

            back_home()
            st.rerun()

        st.stop()

    game = matches.iloc[0]

    home = home_team(game)
    away = away_team(game)
    pick = prediction(game)

    prob = probability(game)

    if prob is None:
        prob = 0.5

    home_probability = parse_probability(
        first_value(
            game,
            ["home_win_probability"],
        )
    )

    away_probability = parse_probability(
        first_value(
            game,
            ["away_win_probability"],
        )
    )

    if home_probability is None or away_probability is None:

        if pick == home:

            home_probability = prob
            away_probability = 1 - prob

        else:

            away_probability = prob
            home_probability = 1 - prob

    status = prediction_status(game)

    # --------------------------------------------------------
    # BACK
    # --------------------------------------------------------

    if st.button("← Back"):

        back_home()
        st.rerun()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="hero-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"### 🏈 WEEK {game.get('week', '—')} • {status}"
    )

    st.markdown(
        f"# {away} @ {home}"
    )

    st.caption(
        str(game_date(game))
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # WIN PROBABILITIES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">'
        '🎯 Model Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        away,
        f"{away_probability:.1%}",
    )

    c2.metric(
        "MODEL PICK",
        pick,
    )

    c3.metric(
        home,
        f"{home_probability:.1%}",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Confidence",
        confidence_label(prob),
    )

    c2.metric(
        "Probability",
        probability_text(prob),
    )

    c3.metric(
        "Prediction Status",
        status,
    )

    # --------------------------------------------------------
    # LOCKED MARKET
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">'
        '💷 Betting Market'
        '</div>',
        unsafe_allow_html=True,
    )

    odds = lock_moneyline(game)
    spread = lock_spread(game)

    lock_captured = normalise_bool(
        first_value(
            game,
            ["lock_market_captured"],
            False,
        )
    )

    if odds is not None and (
        lock_captured
        or "lock_market_captured" not in game.index
    ):

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Lock Moneyline",
            f"{odds:+.0f}",
        )

        c2.metric(
            "Lock Spread",
            (
                f"{spread:+.1f}"
                if spread is not None
                else "—"
            ),
        )

        c3.metric(
            "Provider",
            str(
                first_value(
                    game,
                    ["lock_provider"],
                    "Unavailable",
                )
            ),
        )

        st.success(
            "Verified lock-time market available."
        )

    else:

        st.warning(
            "No verified lock-time market is stored "
            "for this prediction."
        )

        st.caption(
            "This game is excluded from the £100 betting "
            "simulation until genuine lock-time odds exist."
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    tracker_match = pd.DataFrame()

    if (
        not tracker.empty
        and "cfbd_id" in tracker.columns
    ):

        tracker_match = tracker[
            tracker["cfbd_id"].astype(str)
            == selected_id
        ]

    if not tracker_match.empty:

        tracked = tracker_match.iloc[0]

        st.markdown(
            '<div class="section-heading">'
            '🏁 Result'
            '</div>',
            unsafe_allow_html=True,
        )

        result = result_text(tracked)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Prediction Result",
            result,
        )

        c2.metric(
            home,
            str(
                first_value(
                    tracked,
                    ["home_points"],
                    "—",
                )
            ),
        )

        c3.metric(
            away,
            str(
                first_value(
                    tracked,
                    ["away_points"],
                    "—",
                )
            ),
        )

        ml_profit = first_value(
            tracked,
            ["moneyline_profit_loss"],
        )

        if ml_profit is not None:

            try:

                st.metric(
                    "£100 Moneyline P/L",
                    f"£{float(ml_profit):+,.2f}",
                )

            except (TypeError, ValueError):

                pass

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    with st.expander("View prediction record"):

        details = pd.DataFrame(
            {
                "Field": list(game.index),
                "Value": [
                    game[column]
                    for column in game.index
                ],
            }
        )

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# HOME
# ============================================================

elif st.session_state.page == "🏠 Home":

    st.markdown(
        '<div class="hero-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### 🏈 2026 COLLEGE FOOTBALL"
    )

    st.markdown(
        "# CFB Prediction Centre"
    )

    st.markdown(
        "Model-driven college football predictions, "
        "official 24-hour locks and transparent performance "
        "tracking."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="blue-line"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_games = len(games)

    locked = sum(
        official_locked(row)
        for _, row in games.iterrows()
    )

    high_confidence = sum(
        (
            probability(row) is not None
            and probability(row) >= 0.70
        )
        for _, row in games.iterrows()
    )

    completed = 0
    correct = 0

    if not tracker.empty:

        for _, row in tracker.iterrows():

            result = result_is_correct(row)

            if result is not None:

                completed += 1

                if result:
                    correct += 1

    accuracy = (
        correct / completed
        if completed
        else None
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Games",
        f"{total_games:,}",
    )

    c2.metric(
        "Official Locks",
        f"{locked:,}",
    )

    c3.metric(
        "High Confidence",
        f"{high_confidence:,}",
    )

    c4.metric(
        "Lifetime Accuracy",
        (
            f"{accuracy:.1%}"
            if accuracy is not None
            else "—"
        ),
    )

    # --------------------------------------------------------
    # TOP FIVE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">'
        '🔥 Top Five Confidence Games'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subheading">'
        'The highest-confidence selections from the next '
        'upcoming game week.'
        '</div>',
        unsafe_allow_html=True,
    )

    upcoming = games.copy()

    upcoming["_date"] = pd.to_datetime(
        upcoming.apply(game_date, axis=1),
        errors="coerce",
        utc=True,
    )

    now = pd.Timestamp.now(tz="UTC")

    future = upcoming[
        upcoming["_date"] >= now
    ].copy()

    if future.empty:

        future = upcoming.copy()

    if not future.empty:

        future = future.sort_values(
            "_date",
            na_position="last",
        )

        valid_weeks = (
            pd.to_numeric(
                future.get(
                    "week",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            )
            .dropna()
        )

        if not valid_weeks.empty:

            next_week = int(
                valid_weeks.min()
            )

            top_games = future[
                pd.to_numeric(
                    future["week"],
                    errors="coerce",
                )
                == next_week
            ].copy()

        else:

            top_games = future.copy()

        top_games["_probability"] = (
            top_games.apply(
                probability,
                axis=1,
            )
        )

        top_games = (
            top_games
            .dropna(
                subset=["_probability"]
            )
            .sort_values(
                "_probability",
                ascending=False,
            )
            .head(5)
        )

    else:

        top_games = pd.DataFrame()

    if top_games.empty:

        st.info(
            "No upcoming games are currently available."
        )

    else:

        for _, row in top_games.iterrows():

            gid = game_id(row)

            home = home_team(row)
            away = away_team(row)
            pick = prediction(row)
            prob = probability(row)

            st.subheader(
                f"🏈 {away} @ {home}"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Week",
                row.get("week", "—"),
            )

            c2.metric(
                "Model Pick",
                pick,
            )

            c3.metric(
                "Confidence",
                confidence_label(prob),
            )

            c4.metric(
                "Probability",
                probability_text(prob),
            )

            st.caption(
                f"{game_date(row)} • "
                f"{prediction_status(row)}"
            )

            if st.button(
                "🎯 View Game",
                key=f"home_game_{gid}",
                use_container_width=True,
            ):

                open_game(gid)
                st.rerun()

            st.divider()


# ============================================================
# TEAM SEARCH
# ============================================================

elif st.session_state.page == "🏫 Team Search":

    st.markdown(
        '<div class="section-heading">'
        '🏫 Team Search'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subheading">'
        'Search for a single team and view its complete '
        '2026 prediction schedule.'
        '</div>',
        unsafe_allow_html=True,
    )

    team_names = set()

    for _, row in games.iterrows():

        team_names.add(
            home_team(row)
        )

        team_names.add(
            away_team(row)
        )

    team_names = sorted(
        team
        for team in team_names
        if team
        and team.lower() != "nan"
    )

    if not team_names:

        st.info("No teams available.")
        st.stop()

    search = st.text_input(
        "Search for a team",
        placeholder="Start typing a team name...",
    )

    if search:

        matches = [
            team
            for team in team_names
            if search.lower()
            in team.lower()
        ]

        if not matches:

            st.warning(
                "No teams matched your search."
            )

            st.stop()

        selected_team = st.selectbox(
            "Select team",
            matches,
        )

    else:

        selected_team = st.selectbox(
            "Select team",
            team_names,
        )

    team_mask = games.apply(
        lambda row:
            selected_team.lower()
            in {
                home_team(row).lower(),
                away_team(row).lower(),
            },
        axis=1,
    )

    schedule = games[
        team_mask
    ].copy()

    schedule = safe_date_sort(
        schedule
    )

    st.markdown(
        f"### 🏈 {selected_team}"
    )

    st.caption(
        f"{len(schedule)} games in the prediction universe."
    )

    # --------------------------------------------------------
    # TEAM RECORD
    # --------------------------------------------------------

    team_wins = 0
    team_losses = 0

    for _, row in schedule.iterrows():

        result = result_text(row).lower()

        pick = prediction(row)

        if result == "correct":

            if pick == selected_team:
                team_wins += 1

        elif result == "wrong":

            if pick == selected_team:
                team_losses += 1

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Games",
        len(schedule),
    )

    c2.metric(
        "Model Picks",
        sum(
            prediction(row) == selected_team
            for _, row in schedule.iterrows()
        ),
    )

    c3.metric(
        "Official Locks",
        sum(
            official_locked(row)
            for _, row in schedule.iterrows()
        ),
    )

    # --------------------------------------------------------
    # SCHEDULE
    # --------------------------------------------------------

    for _, row in schedule.iterrows():

        gid = game_id(row)

        home = home_team(row)
        away = away_team(row)
        pick = prediction(row)
        prob = probability(row)
        status = prediction_status(row)
        result = result_text(row)

        is_home = (
            selected_team.lower()
            == home.lower()
        )

        opponent = (
            away
            if is_home
            else home
        )

        location = (
            "vs"
            if is_home
            else "@"
        )

        st.markdown(
            f"### {selected_team} {location} {opponent}"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Week",
            row.get("week", "—"),
        )

        c2.metric(
            "Model Pick",
            pick,
        )

        c3.metric(
            "Probability",
            probability_text(prob),
        )

        c4.metric(
            "Status",
            status,
        )

        c5.metric(
            "Result",
            result,
        )

        if st.button(
            "🎯 View Game",
            key=f"team_game_{gid}",
            use_container_width=True,
        ):

            open_game(gid)
            st.rerun()

        st.divider()


# ============================================================
# ALL GAMES
# ============================================================

elif st.session_state.page == "🏈 All Games":

    st.markdown(
        '<div class="section-heading">'
        '🏈 All Games'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subheading">'
        'Browse and filter the complete 2026 prediction universe.'
        '</div>',
        unsafe_allow_html=True,
    )

    if games.empty:

        st.info("No games available.")
        st.stop()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        team_search = st.text_input(
            "Team",
            placeholder="e.g. Alabama",
        )

    with c2:

        weeks = sorted(
            pd.to_numeric(
                games.get(
                    "week",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )

        selected_week = st.selectbox(
            "Week",
            ["All"] + weeks,
        )

    with c3:

        confidence_filter = st.selectbox(
            "Confidence",
            [
                "All",
                "High",
                "Medium",
                "Low",
            ],
        )

    with c4:

        status_filter = st.selectbox(
            "Status",
            [
                "All",
                "FORECAST",
                "LOCK_WINDOW",
                "LOCKED",
            ],
        )

    filtered = games.copy()

    if team_search:

        search_lower = team_search.lower()

        filtered = filtered[
            filtered.apply(
                lambda row:
                    search_lower
                    in home_team(row).lower()
                    or
                    search_lower
                    in away_team(row).lower(),
                axis=1,
            )
        ]

    if selected_week != "All":

        filtered = filtered[
            pd.to_numeric(
                filtered["week"],
                errors="coerce",
            )
            == selected_week
        ]

    if confidence_filter != "All":

        filtered = filtered[
            filtered.apply(
                lambda row:
                    confidence_label(
                        probability(row)
                    )
                    == confidence_filter,
                axis=1,
            )
        ]

    if status_filter != "All":

        filtered = filtered[
            filtered.apply(
                prediction_status,
                axis=1,
            )
            == status_filter
        ]

    filtered = safe_date_sort(
        filtered
    )

    st.write(
        f"Showing **{len(filtered):,}** games."
    )

    # --------------------------------------------------------
    # GAME LIST
    # --------------------------------------------------------

    for _, row in filtered.iterrows():

        gid = game_id(row)

        home = home_team(row)
        away = away_team(row)
        pick = prediction(row)
        prob = probability(row)

        st.markdown(
            f"### {away} @ {home}"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Week",
            row.get("week", "—"),
        )

        c2.metric(
            "Model Pick",
            pick,
        )

        c3.metric(
            "Probability",
            probability_text(prob),
        )

        c4.metric(
            "Confidence",
            confidence_label(prob),
        )

        c5.metric(
            "Status",
            prediction_status(row),
        )

        st.caption(
            str(game_date(row))
        )

        if st.button(
            "🎯 View Game",
            key=f"all_game_{gid}",
            use_container_width=True,
        ):

            open_game(gid)
            st.rerun()

        st.divider()


# ============================================================
# MODEL TRACKER
# ============================================================

elif st.session_state.page == "📊 Model Tracker":

    st.markdown(
        '<div class="section-heading">'
        '📊 Model Tracker'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subheading">'
        'Lifetime prediction performance and the £100 flat-stake '
        'moneyline simulation.'
        '</div>',
        unsafe_allow_html=True,
    )

    if tracker.empty:

        st.info(
            "No tracker records are available yet."
        )
        st.stop()

    # --------------------------------------------------------
    # UPDATE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔄 Update Tracker",
    ):

        with st.spinner(
            "Updating tracker..."
        ):

            try:

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "jobs.update_tracker",
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )

                if result.returncode == 0:

                    load_tracker.clear()
                    load_history.clear()
                    load_predictions.clear()

                    st.success(
                        "Tracker updated successfully."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Tracker update failed."
                    )

                    st.code(
                        result.stderr
                        or result.stdout
                    )

            except Exception as exc:

                st.error(
                    f"Tracker update failed: {exc}"
                )

    # --------------------------------------------------------
    # LIFETIME RECORD
    # --------------------------------------------------------

    results = []

    for _, row in tracker.iterrows():

        result = result_is_correct(row)

        if result is not None:
            results.append(result)

    completed = len(results)
    correct = sum(results)
    wrong = completed - correct

    accuracy = (
        correct / completed
        if completed
        else None
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Predictions",
        len(tracker),
    )

    c2.metric(
        "Completed",
        completed,
    )

    c3.metric(
        "Correct",
        correct,
    )

    c4.metric(
        "Wrong",
        wrong,
    )

    c5.metric(
        "Accuracy",
        (
            f"{accuracy:.1%}"
            if accuracy is not None
            else "—"
        ),
    )

    # --------------------------------------------------------
    # £100 MONEYLINE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">'
        '💷 £100 Moneyline Performance'
        '</div>',
        unsafe_allow_html=True,
    )

    valid_bets = pd.DataFrame()

    if "moneyline_bet_valid" in tracker.columns:

        valid_mask = (
            tracker["moneyline_bet_valid"]
            .astype(str)
            .str.lower()
            .isin(
                ["true", "1", "yes"]
            )
        )

        valid_bets = tracker[
            valid_mask
        ].copy()

    settled_bets = pd.DataFrame()

    if (
        not valid_bets.empty
        and "moneyline_profit_loss"
        in valid_bets.columns
    ):

        settled_bets = valid_bets[
            pd.to_numeric(
                valid_bets[
                    "moneyline_profit_loss"
                ],
                errors="coerce",
            ).notna()
        ].copy()

    total_staked = 0.0
    total_profit = 0.0

    if not settled_bets.empty:

        total_staked = (
            pd.to_numeric(
                settled_bets.get(
                    "moneyline_stake",
                    pd.Series(
                        100.0,
                        index=settled_bets.index,
                    ),
                ),
                errors="coerce",
            )
            .fillna(100)
            .sum()
        )

        total_profit = (
            pd.to_numeric(
                settled_bets[
                    "moneyline_profit_loss"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    roi = (
        total_profit / total_staked
        if total_staked
        else None
    )

    ml_wins = 0
    ml_losses = 0
    ml_pushes = 0

    if (
        not settled_bets.empty
        and "moneyline_result"
        in settled_bets.columns
    ):

        ml_results = (
            settled_bets[
                "moneyline_result"
            ]
            .astype(str)
            .str.upper()
        )

        ml_wins = int(
            ml_results.eq("WIN").sum()
        )

        ml_losses = int(
            ml_results.eq("LOSS").sum()
        )

        ml_pushes = int(
            ml_results.eq("PUSH").sum()
        )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Valid Bets",
        len(valid_bets),
    )

    c2.metric(
        "Settled",
        len(settled_bets),
    )

    c3.metric(
        "Wins / Losses",
        f"{ml_wins} / {ml_losses}",
    )

    c4.metric(
        "Net P/L",
        f"£{total_profit:+,.2f}",
    )

    c5.metric(
        "ROI",
        (
            f"{roi:.2%}"
            if roi is not None
            else "N/A"
        ),
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">'
        '📈 Cumulative £100 P/L'
        '</div>',
        unsafe_allow_html=True,
    )

    if not settled_bets.empty:

        chart = settled_bets.copy()

        chart["_date"] = pd.to_datetime(
            chart.apply(
                game_date,
                axis=1,
            ),
            errors="coerce",
        )

        chart = chart.sort_values(
            "_date",
            na_position="last",
        )

        chart["P/L"] = pd.to_numeric(
            chart[
                "moneyline_profit_loss"
            ],
            errors="coerce",
        ).fillna(0)

        chart["Cumulative P/L"] = (
            chart["P/L"].cumsum()
        )

        chart_index = []

        for _, row in chart.iterrows():

            label = (
                f"W{row.get('week', '—')} "
                f"{away_team(row)} @ "
                f"{home_team(row)}"
            )

            chart_index.append(label)

        chart_data = pd.DataFrame(
            {
                "Cumulative P/L":
                    chart[
                        "Cumulative P/L"
                    ].values
            },
            index=chart_index,
        )

        st.line_chart(
            chart_data,
            height=400,
        )

        st.caption(
            "Each point represents the cumulative result "
            "after one settled £100 moneyline prediction."
        )

    else:

        st.info(
            "No settled bets with verified lock-time odds "
            "are available yet."
        )

    # --------------------------------------------------------
    # BETTING HISTORY
    # --------------------------------------------------------

    if not settled_bets.empty:

        st.markdown(
            '<div class="section-heading">'
            '📋 Betting History'
            '</div>',
            unsafe_allow_html=True,
        )

        columns = [
            "week",
            "start_date",
            "away_team",
            "home_team",
            "predicted_winner",
            "prediction_probability",
            "lock_moneyline",
            "moneyline_result",
            "moneyline_profit_loss",
            "cumulative_moneyline_profit",
        ]

        available = [
            column
            for column in columns
            if column in settled_bets.columns
        ]

        betting_history = (
            settled_bets[
                available
            ]
            .copy()
        )

        betting_history = betting_history.rename(
            columns={
                "week": "Week",
                "start_date": "Date",
                "away_team": "Away",
                "home_team": "Home",
                "predicted_winner": "Model Pick",
                "prediction_probability": "Probability",
                "lock_moneyline": "Lock Odds",
                "moneyline_result": "Result",
                "moneyline_profit_loss": "P/L",
                "cumulative_moneyline_profit":
                    "Cumulative P/L",
            }
        )

        st.dataframe(
            betting_history,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # MARKET COVERAGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">'
        '📡 Lock-Time Market Coverage'
        '</div>',
        unsafe_allow_html=True,
    )

    frozen_count = 0

    if "lock_market_captured" in tracker.columns:

        frozen_count = int(
            tracker[
                "lock_market_captured"
            ]
            .astype(str)
            .str.lower()
            .isin(
                ["true", "1", "yes"]
            )
            .sum()
        )

    c1, c2 = st.columns(2)

    c1.metric(
        "Frozen Markets",
        frozen_count,
    )

    c2.metric(
        "Missing Lock Markets",
        max(
            len(tracker) - frozen_count,
            0,
        ),
    )

    st.caption(
        "The betting simulation uses only the market "
        "snapshot captured when the prediction officially "
        "locked. Historical games without genuine lock-time "
        "odds are not backfilled."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏈 CFB Prediction Centre 2026 • "
    "Dynamic forecasts → 24-hour lock → "
    "official prediction → grading"
)

