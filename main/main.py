import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

matches = pd.read_csv('DataQuest Version/matches.csv', index_col=0)

print(matches)

# print(matches["team"].value_counts())
# print(matches[matches["team"] == "Liverpool"])
# print(matches["round"].value_counts())

# NOTE: DATA CLEANUP 

# Must use numerical data for predictions 
# print(matches.dtypes)

# 1. Convert categorical data to numerical data
matches["date"] = pd.to_datetime(matches["date"])

matches["venue_code"] = matches["venue"].astype("category").cat.codes

matches["opponent_code"] = matches["opponent"].astype("category").cat.codes

matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

matches["day_code"] = matches["date"].dt.dayofweek

matches["target"] = (matches["result"] == "W").astype("int")

# print(matches)

# NOTE: TRAINNIG MACHINE LEARNING MODEL

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = matches[matches["date"] < "2022-01-01"]

test = matches[matches["date"] > "2022-01-01"]

predictors = ["venue_code", "opponent_code", "hour", "day_code"]

rf.fit(train[predictors], train["target"])

preds = rf.predict(test[predictors])

acc = accuracy_score(test["target"], preds) 

# print(f"Accuracy: {acc}")