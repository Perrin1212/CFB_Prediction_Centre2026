from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PREDICTIONS_PATH = Path(
    "data/models/logistic_regression_2025_predictions.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("CFB PREDICTION CENTRE")
print("2025 LEVEL / COMPETITION ANALYSIS")
print("=" * 60)

df = pd.read_csv(PREDICTIONS_PATH)

print()
print(f"✓ Predictions loaded: {len(df):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "home_team",
    "away_team",
    "target",
    "predicted_home_win_probability",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# BASIC MODEL METRICS
# ============================================================

df["predicted_home_win_probability"] = pd.to_numeric(
    df["predicted_home_win_probability"],
    errors="coerce",
)

df["target"] = pd.to_numeric(
    df["target"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "predicted_home_win_probability",
        "target",
    ]
).copy()

df["predicted_home_win"] = (
    df["predicted_home_win_probability"] >= 0.50
).astype(int)

df["correct"] = (
    df["predicted_home_win"]
    ==
    df["target"]
).astype(int)


# ============================================================
# CLASSIFY TEAM LEVEL
# ============================================================

# NOTE:
# This is deliberately a first-pass classification.
# We will inspect the results before making this part
# of the production model.

FBS_TEAMS = {
    # Major examples / known FBS teams.
    # The script also attempts to identify levels from
    # the team names in the data.
}


def classify_team(team: str) -> str:

    team = str(team).strip()

    # --------------------------------------------------------
    # FCS indicators
    # --------------------------------------------------------

    fcs_keywords = [
        "Prairie View",
        "Grambling",
        "Southern",
        "Alabama A&M",
        "Alabama State",
        "Bethune",
        "Florida A&M",
        "Jackson State",
        "Mississippi Valley",
        "Texas Southern",
        "North Carolina A&T",
        "North Carolina Central",
        "Morgan State",
        "Howard",
        "Norfolk State",
        "Delaware State",
        "South Carolina State",
        "Tennessee State",
        "Tennessee Tech",
        "Eastern Illinois",
        "Illinois State",
        "Indiana State",
        "Southern Illinois",
        "Western Illinois",
        "Youngstown State",
        "North Dakota State",
        "South Dakota State",
        "South Dakota",
        "North Dakota",
        "Montana",
        "Montana State",
        "Idaho",
        "Idaho State",
        "Weber State",
        "Eastern Washington",
        "Portland State",
        "Sacramento State",
        "UC Davis",
        "Cal Poly",
        "Northern Arizona",
        "Northern Colorado",
        "Eastern Kentucky",
        "Central Arkansas",
        "Austin Peay",
        "Murray State",
        "Tennessee-Martin",
        "SE Missouri State",
        "Samford",
        "Mercer",
        "Furman",
        "Wofford",
        "Chattanooga",
        "The Citadel",
        "VMI",
        "Elon",
        "Campbell",
        "Richmond",
        "William & Mary",
        "Villanova",
        "Delaware",
        "Monmouth",
        "Towson",
        "Rhode Island",
        "Maine",
        "New Hampshire",
        "Stony Brook",
        "Albany",
        "Dartmouth",
        "Harvard",
        "Yale",
        "Princeton",
        "Brown",
        "Columbia",
        "Cornell",
        "Penn",
        "Lafayette",
        "Lehigh",
        "Fordham",
        "Holy Cross",
        "Colgate",
        "Bucknell",
        "Georgetown",
        "Butler",
        "Davidson",
        "Drake",
        "Marist",
        "Valparaiso",
        "Morehead State",
        "Dayton",
        "Davidson",
        "Presbyterian",
        "Stetson",
        "Jacksonville",
        "Mercyhurst",
        "Robert Morris",
        "Duquesne",
        "Central Connecticut",
        "Wagner",
        "Sacred Heart",
        "LIU",
    ]

    for keyword in fcs_keywords:

        if keyword.lower() in team.lower():

            return "FCS"


    # --------------------------------------------------------
    # Lower divisions / non-NCAA top level indicators
    # --------------------------------------------------------

    lower_keywords = [
        "D-II",
        "D-III",
        "Division II",
        "Division III",
        "NAIA",
        "North Central",
        "Wisconsin-Whitewater",
        "Mount Union",
        "Mary Hardin-Baylor",
        "St. Thomas (MN)",
        "Augustana (SD)",
        "Sioux Falls",
        "Minnesota Duluth",
        "Grand Valley State",
        "Ferris State",
        "Valdosta State",
        "West Florida",
        "Delta State",
        "Harding",
        "North Greenville",
        "Erskine",
        "Chowan",
        "Miles",
        "Kentucky Christian",
        "Hanover",
        "Wheaton",
        "Wabash",
        "Trinity (TX)",
        "Johns Hopkins",
        "Muhlenberg",
        "Rowan",
        "Amherst",
        "Williams",
        "Middlebury",
        "Tufts",
        "Cortland",
        "Ithaca",
        "Mount St. Joseph",
        "Rose-Hulman",
        "Bluffton",
        "Hilbert",
        "Buffalo State",
        "Adrian",
        "Wheeling",
        "Walsh",
        "Savannah State",
        "Allen",
    ]

    for keyword in lower_keywords:

        if keyword.lower() in team.lower():

            return "Lower Division"


    # --------------------------------------------------------
    # Otherwise assume FBS
    # --------------------------------------------------------

    return "FBS"


df["home_level"] = df[
    "home_team"
].apply(classify_team)

df["away_level"] = df[
    "away_team"
].apply(classify_team)


# ============================================================
# GAME CATEGORY
# ============================================================

def classify_game(row):

    home = row["home_level"]
    away = row["away_level"]

    if home == "FBS" and away == "FBS":
        return "FBS vs FBS"

    if home == "FCS" and away == "FCS":
        return "FCS vs FCS"

    if home == "Lower Division" and away == "Lower Division":
        return "Lower vs Lower"

    if (
        home == "FBS"
        and away == "FCS"
    ) or (
        home == "FCS"
        and away == "FBS"
    ):
        return "FBS vs FCS"

    if "Lower Division" in [home, away]:
        return "Includes Lower Division"

    return "Other"


df["game_category"] = df.apply(
    classify_game,
    axis=1,
)


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

print()
print("=" * 60)
print("PERFORMANCE BY GAME CATEGORY")
print("=" * 60)

summary = (
    df.groupby(
        "game_category",
        observed=False,
    )
    .agg(
        games=("correct", "size"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        average_probability=(
            "predicted_home_win_probability",
            "mean",
        ),
    )
    .reset_index()
)

summary = summary.sort_values(
    "games",
    ascending=False,
)

print()

for _, row in summary.iterrows():

    print(
        f"{row['game_category']:<25} "
        f"Games: {int(row['games']):5d} | "
        f"Accuracy: {row['accuracy']:.2%} | "
        f"Avg Prob: {row['average_probability']:.2%}"
    )


# ============================================================
# CONFIDENCE BY CATEGORY
# ============================================================

print()
print("=" * 60)
print("HIGH-CONFIDENCE PERFORMANCE BY CATEGORY")
print("=" * 60)

for threshold in [
    0.70,
    0.80,
    0.90,
]:

    print()
    print(
        f"PROBABILITY >= {threshold:.0%}"
    )

    high_conf = df[
        df["predicted_home_win_probability"]
        >= threshold
    ].copy()

    if high_conf.empty:

        print("  No games.")

        continue

    category_summary = (
        high_conf.groupby(
            "game_category",
            observed=False,
        )
        .agg(
            games=("correct", "size"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
        .sort_values(
            "games",
            ascending=False,
        )
    )

    for _, row in category_summary.iterrows():

        print(
            f"  {row['game_category']:<23} "
            f"Games: {int(row['games']):4d} | "
            f"Accuracy: {row['accuracy']:.2%}"
        )


# ============================================================
# TEAM LEVEL COUNTS
# ============================================================

print()
print("=" * 60)
print("TEAM LEVEL COUNTS")
print("=" * 60)

print()
print("Home teams:")

print(
    df["home_level"]
    .value_counts()
    .to_string()
)

print()
print("Away teams:")

print(
    df["away_level"]
    .value_counts()
    .to_string()
)


# ============================================================
# FLAG EXTREME MISSES
# ============================================================

print()
print("=" * 60)
print("EXTREME CONFIDENCE MISSES")
print("=" * 60)

extreme_misses = df[
    (df["correct"] == 0)
    &
    (
        df["predicted_home_win_probability"]
        >= 0.90
    )
].copy()

extreme_misses = extreme_misses.sort_values(
    "predicted_home_win_probability",
    ascending=False,
)

columns = [
    "week",
    "home_team",
    "away_team",
    "home_level",
    "away_level",
    "game_category",
    "predicted_home_win_probability",
    "target",
]

print()

if extreme_misses.empty:

    print("No 90%+ misses.")

else:

    print(
        extreme_misses[
            columns
        ].head(50).to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

output_path = Path(
    "data/models/2025_level_analysis.csv"
)

summary.to_csv(
    output_path,
    index=False,
)

print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print()
print(
    f"✓ Saved: {output_path}"
)