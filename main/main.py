# Load libraries for data manipulation and model training/evaluation
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score

# Load the raw match data. Using index_col=0 drops the implicit CSV index column
matches = pd.read_csv('main/matches.csv', index_col=0)

# Quick visual check of the dataset (shape, columns, sample values)
print(matches)

# print(matches["team"].value_counts())
# print(matches[matches["team"] == "Liverpool"])
# print(matches["round"].value_counts())

# NOTE: DATA CLEANUP
# Convert and engineer features so the model can consume numeric inputs

# Must use numerical data for predictions
# print(matches.dtypes)

# 1) Convert date/time and categorical fields
matches["date"] = pd.to_datetime(matches["date"])

# Map string categories to integer codes
matches["venue_code"] = matches["venue"].astype("category").cat.codes

matches["opponent_code"] = matches["opponent"].astype("category").cat.codes

# Extract match start hour from "HH:MM" strings
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

# Day of week as numeric feature (Mon=0, Sun=6)
matches["day_code"] = matches["date"].dt.dayofweek

# Binary classification target: 1 if win (W), else 0
matches["target"] = (matches["result"] == "W").astype("int")

# print(matches)

# NOTE: TRAINING MACHINE LEARNING MODEL
# We train a RandomForest to predict whether a team wins a match

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# Temporal split: train on matches before 2022; test on matches after 2022-01-01
train = matches[matches["date"] < "2022-01-01"]

test = matches[matches["date"] > "2022-01-01"]

# Baseline predictors without rolling features
predictors = ["venue_code", "opponent_code", "hour", "day_code"]

rf.fit(train[predictors], train["target"])

preds = rf.predict(test[predictors])

# Basic accuracy for sanity check (not the primary metric for imbalanced data)
acc = accuracy_score(test["target"], preds) 

# print(f"Accuracy: {acc}")

# Confusion-like table for quick inspection, kept as an object (not printed)
combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))

pd.crosstab(index=combined["actual"], columns=combined["predicted"])

# Precision is more informative than accuracy if classes are imbalanced
precision = precision_score(test["target"], preds)

# print(f"Precision: {precision}")

# Group by team for rolling stats computed within each team
grouped_matches = matches.groupby("team")

# Compute rolling averages for recent-form features per team
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed="left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


# Base columns to roll (form stats from prior matches)
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]

# Names for the new rolling features
new_cols = [f"{col}_rolling" for col in cols]


# Apply rolling feature engineering per team and recombine
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))

# Remove the group key from the index and reset to simple RangeIndex
matches_rolling = matches_rolling.droplevel("team")

matches_rolling.index = range(matches_rolling.shape[0])

# Helper to train/predict using any given dataset and predictor set
def make_predictions(data, predictors):
    train = data[data["date"] < "2022-01-01"]
    test = data[data["date"] > "2022-01-01"]
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))
    precision = precision_score(test["target"], preds)
    return combined, precision

# Retrain with rolling features included to capture recent team form
combined, precision = make_predictions(matches_rolling, predictors + new_cols)
# Attach useful match metadata for interpretation/analysis
combined = combined.merge(matches_rolling[["team", "date", "opponent", "result"]], left_index=True, right_index=True)

# Mapping to normalize some team names for consistent joins
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}

mapping = MissingDict(**map_values)

# Normalize team names in preparation for a self-merge against opponent names
combined["new_team"] = combined["team"].map(mapping)

# Join predictions for both sides of the same fixture on date and opponent
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

# Filter for cases where home prediction is win (1) and opponent is not (0);
# then inspect how often that corresponds to an actual win for the first team.
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()
