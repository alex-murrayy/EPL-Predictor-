# Load libraries for data manipulation and model training/evaluation
import pandas as pd
import difflib
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

# --- On-demand single-match prediction helpers ---
# Team name normalization for user input (handles abbreviations and variations)
def _simplify_name(name: str) -> str:
    s = name.lower().strip()
    # Remove punctuation and collapse spaces
    for ch in ",.-_/\\()'\"":
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    # Common suffix removal
    for token in ["fc"]:
        s = s.replace(f" {token}", "")
    return s

# Build canonical set from data
_canonical_names = sorted(set(matches["team"].unique()) | set(matches["opponent"].unique()))
_simplified_to_canonical = { _simplify_name(n): n for n in _canonical_names }

# Handcrafted aliases → canonical names
_alias_map = {
    # Manchester United
    "manchester united": "Manchester United",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "man u": "Manchester United",
    # Manchester City
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "man c": "Manchester City",
    # Tottenham Hotspur
    "tottenham": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    # Wolverhampton Wanderers
    "wolves": "Wolverhampton Wanderers",
    # West Ham United
    "west ham": "West Ham United",
    # Newcastle United
    "newcastle": "Newcastle United",
    "newcastle utd": "Newcastle United",
    # Brighton
    "brighton": "Brighton and Hove Albion",
}

def canonicalize_team_name(user_input: str) -> str:
    """Return canonical team name from dataset, tolerating common variants.

    Strategy: alias lookup → exact simplified match → fuzzy match over known teams.
    """
    key = _simplify_name(user_input)
    # 1) Alias table
    if key in _alias_map:
        return _alias_map[key]
    # 2) Exact simplified match to a known team
    if key in _simplified_to_canonical:
        return _simplified_to_canonical[key]
    # 3) Fuzzy match on simplified keys
    choices = list(_simplified_to_canonical.keys())
    match = difflib.get_close_matches(key, choices, n=1, cutoff=0.8)
    if match:
        return _simplified_to_canonical[match[0]]
    # Fallback to original (will likely error later if unknown)
    return user_input
# Build code maps from the original encodings so we can encode new inputs consistently
venue_cats = matches["venue"].astype("category").cat.categories
venue_code_map = {cat: code for code, cat in enumerate(venue_cats)}

opponent_cats = matches["opponent"].astype("category").cat.categories
opponent_code_map = {cat: code for code, cat in enumerate(opponent_cats)}

def get_latest_rolling_features(team_name: str) -> pd.Series:
    """Return the most recent rolling feature vector for the given team."""
    team_rows = matches_rolling[matches_rolling["team"] == team_name]
    if team_rows.empty:
        raise ValueError(f"No rolling data available for team: {team_name}")
    latest = team_rows.sort_values("date").iloc[-1]
    return latest[new_cols]

def prepare_feature_row(team: str, opponent: str, venue: str, when_date: str | None = None) -> pd.DataFrame:
    """Create a single-row DataFrame with engineered features for prediction.

    - team/opponent: use original naming as in the CSV (mapping applied where useful)
    - venue: one of {"Home", "Away"}
    - when_date: optional ISO date (YYYY-MM-DD). If omitted, use the latest date in data.
    """
    # Normalize names from user input to the dataset's canonical names
    # Do NOT apply the short-name mapping here; rolling features and encoders
    # are built from canonical names present in the raw data.
    norm_team = canonicalize_team_name(team)
    norm_opp = canonicalize_team_name(opponent)

    # Date, hour, weekday features
    if when_date is None:
        ref_date = matches["date"].max()
    else:
        ref_date = pd.to_datetime(when_date)
    hour = 15  # typical kickoff fallback
    day_code = int(pd.to_datetime(ref_date).dayofweek)

    # Encode venue and opponent using learned category maps
    if venue not in venue_code_map:
        raise ValueError("venue must be either 'Home' or 'Away'")
    venue_code_val = int(venue_code_map[venue])
    if norm_opp not in opponent_code_map:
        raise ValueError(f"Unknown opponent name: {opponent}")
    opponent_code_val = int(opponent_code_map[norm_opp])

    # Latest rolling stats for the selected team
    rolling_vals = get_latest_rolling_features(norm_team)

    # Compose the full feature vector expected by the model
    data = {
        "venue_code": [venue_code_val],
        "opponent_code": [opponent_code_val],
        "hour": [hour],
        "day_code": [day_code],
        **{col: [float(rolling_vals[col])] for col in new_cols},
    }
    return pd.DataFrame(data)

def predict_match(team: str, opponent: str, venue: str, when_date: str | None = None) -> tuple[float, int]:
    """Return (prob_team_win, predicted_label) for team vs opponent at venue on date.

    predicted_label: 1 means team predicted to win, 0 otherwise.
    """
    features = prepare_feature_row(team, opponent, venue, when_date)
    # Ensure columns order matches the training predictors
    full_predictors = predictors + new_cols
    X = features[full_predictors]
    proba = rf.predict_proba(X)[0][1]
    label = int(proba >= 0.5)
    return float(proba), label

if __name__ == "__main__":
    # Optional interactive prediction flow
    try:
        use_cli = input("Make a prediction? (y/N): ").strip().lower() == "y"
        if use_cli:
            team_in = input("Team name: ").strip()
            opp_in = input("Opponent name: ").strip()
            venue_in = input("Venue (Home/Away): ").strip().title()
            date_in = input("Date (YYYY-MM-DD) [optional]: ").strip() or None
            prob, lbl = predict_match(team_in, opp_in, venue_in, date_in)
            outcome = "Win" if lbl == 1 else "Not Win"
            print(f"Prediction for {team_in} vs {opp_in} at {venue_in}: {outcome} (p_win={prob:.3f})")
    except Exception as e:
        print(f"Prediction error: {e}")
