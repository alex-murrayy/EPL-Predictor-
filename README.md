## Football Predictor

Simple match outcome predictor for Premier League-style data. Trains a Random Forest on engineered features (categoricals, time, and recent-form rolling stats), and supports interactive single-match predictions from the CLI.

### 1) Requirements

- Python 3.12 (recommended via Homebrew on Apple Silicon)
- macOS or Linux

### 2) Setup

Create a virtual environment and install dependencies:

```bash
cd "./Football Predictor"
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If you use VS Code/Cursor, select the interpreter:

- Command Palette → “Python: Select Interpreter” → choose `.venv/bin/python`

### 3) Data

- The project expects a CSV at `main/matches.csv` (already present in this repo).
- The script will parse features and split data by date for training/test.

### 4) Train and evaluate

Run the script to train the model and compute metrics:

```bash
. .venv/bin/activate
python main/main.py
```

You’ll see the loaded data printed, then internal metrics (accuracy, precision calculations are present; some prints are commented to keep output tidy).

### 5) Interactive predictions (CLI)

When you run `main/main.py`, you’ll be prompted:

```
Make a prediction? (y/N): y
Team name: man united
Opponent name: spurs
Venue (Home/Away): Home
Date (YYYY-MM-DD) [optional]:
```

The script normalizes common team-name variants (e.g., “man united”, “man utd”, “spurs”). It then prints the predicted outcome for the selected team and the probability of winning.

Notes:

- Use venue exactly as `Home` or `Away`.
- Date is optional; if omitted, the latest date in the dataset is used to compute weekday features.
- If a team name isn’t recognized, try the dataset’s exact spelling as found in `main/matches.csv`.

### 6) Common issues

- Missing imports in the editor: ensure the interpreter is set to `.venv/bin/python`.
- File not found: if you changed the CSV location, update the path in `main/main.py` (`pd.read_csv('main/matches.csv', index_col=0)`).
- Unknown team name for predictions: the CLI uses alias + fuzzy matching. If it still fails, use the exact name from the dataset.

### 7) How it works (brief)

- Feature engineering: converts categorical fields (`venue`, `opponent`) to codes, extracts hour and weekday from time/date, and creates 3-match rolling averages (e.g., goals for/against, shots, etc.) within each team.
- Model: RandomForestClassifier trained on matches before 2022-01-01; evaluated on after.
- CLI prediction: builds a single feature row using the most recent rolling stats for the chosen team, encodes opponent and venue, and returns the win probability.

### 8) Next steps (improvements)

- Add probability calibration (`CalibratedClassifierCV`) for better-calibrated outputs.
- Use time-series cross-validation to tune hyperparameters.
- Add strength metrics (Elo), rest days, league table position, and head-to-head features.
- Try gradient boosting models (LightGBM/XGBoost) and compare PR AUC/ROC AUC.
