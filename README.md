# ⚽ EPL Season Predictor

A football analytics and machine learning project for predicting **English Premier League match outcomes** and estimating **end-of-season title, top-4, and relegation probabilities**.

The project combines historical Premier League statistics, match-level expected goals (xG), engineered team-strength features, machine learning, and Monte Carlo simulation.

---

## 🚀 What the Project Does

The project follows a pipeline:

```text
Historical EPL Data
        │
        ▼
Data Processing
        │
        ▼
Feature Engineering
        │
        ├── Team Strength
        ├── Recent Form
        ├── Goals / Shots
        ├── Expected Goals (xG)
        ├── Home Advantage
        └── Rest Days
        │
        ▼
Machine Learning Model
        │
        ▼
Match Outcome Probabilities
        │
        ▼
Remaining Fixtures
        │
        ▼
Monte Carlo Simulation
        │
        ▼
Season Outcome Probabilities
```

The final simulation produces:

* Current points
* Expected final points
* Title probability
* Top-4 probability
* Relegation probability

Example:

```text
team             points_now  exp_final_points  title_%  top4_%  relegation_%
Man City                  15                74     35.0    81.6           0.0
Arsenal                   12                73     31.1    80.1           0.0
Brighton                  10                65      9.9    50.7           0.3
Man United                 5                65      9.4    51.8           0.3
Liverpool                  9                64      8.2    45.2           0.5
```

---

## 🧠 Model

The model predicts the probability of three possible match outcomes:

```text
Away Win
Draw
Home Win
```

These probabilities are used by the Monte Carlo simulation to generate possible outcomes for the remaining fixtures.

Rather than directly predicting a final league table, the project simulates individual matches and aggregates the results into possible final tables.

---

## 📊 Historical Data

The project uses historical Premier League match and team statistics.

The collected data includes features such as:

* Possession
* Goals
* Assists
* Goals + assists
* Non-penalty goals
* Penalties
* Shots
* Shots on target
* Shot accuracy
* Goals per shot
* Goals per shot on target
* Cards
* Other performance statistics

Historical seasons used in the project include:

* 2019/20
* 2020/21
* 2021/22
* Additional seasons used by the model

---

## 📈 Expected Goals (xG)

The project also collects **match-level xG data from Understat**.

The xG collection script retrieves historical EPL matches and stores the data in:

```text
data/xg_raw.csv
```

Each record contains:

```text
Date
HomeTeam
AwayTeam
home_xg
away_xg
```

The xG information can then be incorporated into the project's feature set.

---

## 🎲 Monte Carlo Simulation

The remaining fixtures are simulated many times.

For each simulation:

1. A result is generated for every remaining fixture according to the model's predicted probabilities.
2. Points are awarded according to the simulated results.
3. Teams are ranked according to their simulated final points.
4. The process is repeated thousands of times.

For example:

```bash
python simulate.py --fixtures pl_2026_27_fixtures_all.csv --n 50000 --sigma 0.2
```

The simulation produces probabilities based on how frequently each team reaches a particular outcome across all simulations.

---

## 🌡️ Team-Strength Uncertainty

The simulation includes a `sigma` parameter to introduce variation in team strength between simulations.

Example:

```bash
--sigma 0.2
```

This allows different simulations to represent slightly different underlying team-strength scenarios rather than treating the estimated strengths as completely fixed.

---

## 🗓️ Fixture Rest

The project calculates the number of days between a team's scheduled league matches.

Rest information is incorporated into the features used when generating predictions for upcoming fixtures.

---

## 📁 Project Structure

```text
epl_predictor/
│
├── data/
│   ├── raw/
│   │   └── historical EPL data
│   ├── xg_raw.csv
│   └── title_odds.csv
│
├── data_fetch.py
├── features.py
├── fixtures.py
├── model.py
├── simulate.py
├── main.py
│
├── pl_2026_27_fixtures_all.csv
├── requirements.txt
└── README.md
```

### File Descriptions

| File                  | Purpose                                                                    |
| --------------------- | -------------------------------------------------------------------------- |
| `data_fetch.py`       | Collects match-level xG data and prepares the xG dataset                   |
| `features.py`         | Loads historical data, maintains team state, and generates model features  |
| `fixtures.py`         | Handles remaining fixtures and team-name mapping                           |
| `model.py`            | Handles the trained machine-learning model and match-outcome probabilities |
| `simulate.py`         | Runs the vectorised Monte Carlo simulation for the remaining season        |
| `main.py`             | Main project entry point                                                   |
| `requirements.txt`    | Python dependencies                                                        |
| `data/xg_raw.csv`     | Match-level xG dataset                                                     |
| `data/title_odds.csv` | Generated season simulation results                                        |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

The project uses packages including:

```text
pandas
numpy
scikit-learn
joblib
understatapi
```

---

## ▶️ Running the Project

### Fetch historical match data

```bash
python data_fetch.py
```

This retrieves historical EPL match data and writes:

```text
data/raw/
```

### Fetch xG data

```bash
python understat_fetch.py
```

This retrieves historical EPL xG data and writes:

```text
data/xg_raw.csv
```

### Train the model

```bash
python main.py
```

### Run the season simulation

```bash
python simulate.py --fixtures pl_2026_27_fixtures_all.csv --n 50000 --sigma 0.2
```

This generates the season simulation results in:

```text
data/title_odds.csv
```

### Tune the model

```bash
python tune.py
```

### Simulation arguments

| Argument     | Description                         |
| ------------ | ----------------------------------- |
| `--fixtures` | Fixture CSV used for the simulation |
| `--n`        | Number of Monte Carlo simulations   |
| `--sigma`    | Team-strength uncertainty           |
| `--seed`     | Random seed                         |

The simulation results are saved to:

```text
data/title_odds.csv
```

---

## 📄 Output

The generated CSV contains:

```text
team
points_now
exp_final_points
title_%
top4_%
relegation_%
```

### Output Columns

| Column             | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| `team`             | Team name                                                                  |
| `points_now`       | Current league points                                                      |
| `exp_final_points` | Average simulated final points                                             |
| `title_%`          | Percentage of simulations where the team finishes first                    |
| `top4_%`           | Percentage of simulations where the team finishes in the top four          |
| `relegation_%`     | Percentage of simulations where the team finishes in a relegation position |

The output is sorted by `title_%`, with `points_now` used as the secondary ordering when title percentages are equal.

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **scikit-learn**
* **Joblib**
* **Understat API**
* **Monte Carlo simulation**

---

## 🎯 Project Goals

The project explores how football prediction can combine:

* Statistical analysis
* Machine learning
* Football-specific features
* Expected goals
* Historical data
* Probabilistic simulation

Instead of producing only one deterministic prediction, the simulation generates a distribution of possible outcomes for the remainder of the season.

---

## ⚠️ Limitations

The predictions depend on the quality of the underlying data, model assumptions, and uncertainty in football results.

Factors that are difficult to capture completely include:

* Injuries and suspensions
* Transfers and squad changes
* Tactical changes
* Managerial changes
* Unexpected changes in team performance
* Data quality
* Model limitations
* Random variation in individual matches

The resulting probabilities should therefore be interpreted as **model estimates rather than guarantees**.

---

## 📌 Future Improvements

Potential extensions include:

* Player-level features
* Player availability and injury information
* More advanced xG features
* Home/away-specific team strength
* Recent xG form
* Opponent-adjusted statistics
* Probability calibration
* Bookmaker odds comparison
* SQL database integration
* Interactive dashboard
* Historical model backtesting

---

## 👤 Author

**Gavin Wahengbam**

Computer Science & Engineering Student

Sri Jayachamarajendra College of Engineering