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
print("2025 OUT-OF-SAMPLE PREDICTION ANALYSIS")
print("=" * 60)

df = pd.read_csv(PREDICTIONS_PATH)

print()
print(f"✓ Predictions loaded: {len(df):,}")

print()
print("COLUMNS")
print("-" * 60)

for column in df.columns:
    print(f"  - {column}")


# ============================================================
# IDENTIFY PROBABILITY COLUMN
# ============================================================

possible_probability_columns = [
    "predicted_home_win_probability",
    "home_win_probability",
    "predicted_probability",
    "prediction_probability",
    "probability",
    "home_win_prob",
    "pred_win_prob",
]

probability_column = next(
    (
        column
        for column in possible_probability_columns
        if column in df.columns
    ),
    None,
)

if probability_column is None:
    raise ValueError(
        "Could not identify the home-win probability column."
    )


# ============================================================
# IDENTIFY TARGET COLUMN
# ============================================================

possible_target_columns = [
    "target",
    "home_win",
    "actual",
    "actual_home_win",
]

target_column = next(
    (
        column
        for column in possible_target_columns
        if column in df.columns
    ),
    None,
)

if target_column is None:
    raise ValueError(
        "Could not identify the target column."
    )


print()
print(
    f"✓ Probability column: {probability_column}"
)

print(
    f"✓ Target column: {target_column}"
)


# ============================================================
# NORMALISE
# ============================================================

df[probability_column] = pd.to_numeric(
    df[probability_column],
    errors="coerce",
)

df[target_column] = pd.to_numeric(
    df[target_column],
    errors="coerce",
)

df = df.dropna(
    subset=[
        probability_column,
        target_column,
    ]
).copy()

df["prediction"] = (
    df[probability_column] >= 0.50
).astype(int)

df["correct"] = (
    df["prediction"]
    ==
    df[target_column]
).astype(int)


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

accuracy = df["correct"].mean()

print()
print("=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

print()
print(
    f"Games analysed : {len(df):,}"
)

print(
    f"Accuracy       : {accuracy:.4f}"
)

print(
    f"Accuracy       : {accuracy * 100:.2f}%"
)


# ============================================================
# CONFIDENCE BANDS
# ============================================================

print()
print("=" * 60)
print("CONFIDENCE BAND ANALYSIS")
print("=" * 60)

bins = [
    0.00,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.01,
]

labels = [
    "<50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90%+",
]

df["confidence_band"] = pd.cut(
    df[probability_column],
    bins=bins,
    labels=labels,
    right=False,
)

band_summary = (
    df.groupby(
        "confidence_band",
        observed=False,
    )
    .agg(
        games=("correct", "size"),
        wins=("target", "sum"),
        correct=("correct", "sum"),
        average_probability=(
            probability_column,
            "mean",
        ),
        actual_win_rate=(
            target_column,
            "mean",
        ),
    )
    .reset_index()
)

band_summary["accuracy"] = (
    band_summary["correct"]
    /
    band_summary["games"]
)

print()

for _, row in band_summary.iterrows():

    print(
        f"{str(row['confidence_band']):>8} | "
        f"Games: {int(row['games']):4d} | "
        f"Accuracy: {row['accuracy'] * 100:6.2f}% | "
        f"Pred Prob: {row['average_probability'] * 100:6.2f}% | "
        f"Actual Win: {row['actual_win_rate'] * 100:6.2f}%"
    )


# ============================================================
# HIGH-CONFIDENCE PICKS
# ============================================================

print()
print("=" * 60)
print("HIGH-CONFIDENCE PICKS")
print("=" * 60)

for threshold in [
    0.60,
    0.70,
    0.80,
    0.90,
]:

    subset = df[
        df[probability_column] >= threshold
    ]

    if subset.empty:
        continue

    accuracy = subset["correct"].mean()

    actual_win_rate = subset[
        target_column
    ].mean()

    print()
    print(
        f">= {threshold:.0%}"
    )

    print(
        f"  Games        : {len(subset):,}"
    )

    print(
        f"  Accuracy     : {accuracy:.2%}"
    )

    print(
        f"  Actual wins  : {actual_win_rate:.2%}"
    )


# ============================================================
# FAVOURITE VS UNDERDOG
# ============================================================

print()
print("=" * 60)
print("MODEL PICK DIRECTION")
print("=" * 60)

home_picks = df[
    df[probability_column] >= 0.50
]

away_picks = df[
    df[probability_column] < 0.50
]

if not home_picks.empty:

    print()
    print("HOME WIN PICKS")

    print(
        f"  Games    : {len(home_picks):,}"
    )

    print(
        f"  Accuracy : "
        f"{home_picks['correct'].mean():.2%}"
    )

if not away_picks.empty:

    print()
    print("AWAY WIN PICKS")

    print(
        f"  Games    : {len(away_picks):,}"
    )

    print(
        f"  Accuracy : "
        f"{away_picks['correct'].mean():.2%}"
    )


# ============================================================
# MOST CONFIDENT LOSSES
# ============================================================

print()
print("=" * 60)
print("MOST CONFIDENT WRONG PREDICTIONS")
print("=" * 60)

wrong = df[
    df["correct"] == 0
].copy()

wrong = wrong.sort_values(
    probability_column,
    ascending=False,
)

display_columns = [
    column
    for column in [
        "season",
        "week",
        "home_team",
        "away_team",
        probability_column,
        target_column,
    ]
    if column in wrong.columns
]

print()

if wrong.empty:

    print("No incorrect predictions.")

else:

    print(
        wrong[
            display_columns
        ].head(25).to_string(
            index=False
        )
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

output_path = Path(
    "data/models/2025_prediction_analysis.csv"
)

band_summary.to_csv(
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