"""
Elo + rolling-form features. Only information available BEFORE each match is used.

v5 changes:
  - xG now comes from understat_fetch.py (data/xg_raw.csv); FBref lost its
    xG data in Jan 2026
  - rest days capped at MAX_REST_DAYS so the first match of a season doesn't
    show ~90 days of "rest"
  - season PPG shrunk toward the league average (PPG_PRIOR_GAMES pseudo-games)
    so it isn't 0 / 1 / 3 after a single match
  - row_features accepts an optional rest override for the season simulator

v4: xG rolling form (home_form_xg / away_form_xg / home_form_xga / away_form_xga)
v3: recency-weighted form, rest days, season PPG, bundled `state` dict
Bookmaker implied probabilities are stored for benchmarking ONLY
(they are not in FEATURE_COLS).
"""
import glob
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
XG_PATH = os.path.join(os.path.dirname(__file__), "data", "xg_raw.csv")

K_FACTOR = 15              # tuned via tune.py grid search (val logloss 0.964 vs 0.966 at K=20)
HOME_ADV = 50              # tuned via tune.py grid search
FORM_WINDOW = 8            # widened from 5: a short hot/cold streak shouldn't dominate early in a season
FORM_DECAY = 0.90          # softened from 0.80: less overreaction to the last 1-2 results
SEASON_REGRESS = 0.45      # NOTE: raised by hand from the tuned 0.3 -- validate this in tune.py
PROMOTED_ELO = 1430.0      # start rating for promoted teams (tune this)
DEFAULT_REST_DAYS = 7.0    # assumed rest for a team's first match on record
MAX_REST_DAYS = 14         # cap so season-opening games don't show ~90 days of rest
PPG_PRIOR_GAMES = 5        # pseudo-games at league-average PPG (1.3) blended into season PPG

DEFAULTS = dict(pts=1.0, gf=1.2, ga=1.2, shots=10.0, sot=4.0, corners=5.0, xg=1.3, xga=1.3)

FEATURE_COLS = [
    "elo_diff", "home_elo", "away_elo",
    "home_form_pts", "away_form_pts",
    "home_form_gf", "home_form_ga", "away_form_gf", "away_form_ga",
    "home_form_shots", "away_form_shots",
    "home_form_sot", "away_form_sot",
    "home_form_corners", "away_form_corners",
    "home_form_xg", "away_form_xg", "home_form_xga", "away_form_xga",
    "home_rest_days", "away_rest_days",
    "home_season_ppg", "away_season_ppg",
]


def _load_xg() -> pd.DataFrame:
    """Loads understat_fetch.py's output if it exists. Returns an empty frame
    otherwise, so the merge below just no-ops."""
    cols = ["Date", "HomeTeam", "AwayTeam", "home_xg", "away_xg"]
    if not os.path.exists(XG_PATH):
        return pd.DataFrame(columns=cols)
    xg = pd.read_csv(XG_PATH)
    xg["Date"] = pd.to_datetime(xg["Date"], errors="coerce")
    return xg.dropna(subset=["Date", "HomeTeam", "AwayTeam"])


def load_raw(pattern: str = "E0_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW_DIR, pattern)))
    if not files:
        raise FileNotFoundError(f"No raw data in {RAW_DIR}. Run data_fetch.py first.")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin1")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"]).copy()   # .copy() avoids the fragmentation warning
        df["season_file"] = os.path.basename(f)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["Date"] = pd.to_datetime(all_df["Date"], dayfirst=True, format="mixed", errors="coerce")
    all_df = all_df.dropna(subset=["Date"])
    all_df = all_df.sort_values("Date", kind="stable").reset_index(drop=True)

    xg = _load_xg()
    if len(xg):
        all_df = all_df.merge(xg, on=["Date", "HomeTeam", "AwayTeam"], how="left")
        missing = all_df[all_df["home_xg"].isna()]
        print(f"  xG matched on {len(all_df) - len(missing)}/{len(all_df)} matches")
        if len(missing) > 20:
            print("  Many matches have no xG -- check NAME_MAP in understat_fetch.py. First few:")
            print(missing[["Date", "HomeTeam", "AwayTeam"]].head(5).to_string(index=False))
    else:
        all_df["home_xg"], all_df["away_xg"] = np.nan, np.nan
    return all_df


def form_avg(h) -> dict:
    """Exponentially recency-weighted average over the deque (most recent = highest weight)."""
    if not h:
        return dict(DEFAULTS)
    n = len(h)
    weights = [FORM_DECAY ** (n - 1 - i) for i in range(n)]  # oldest gets smallest weight
    out = {}
    for k, default in DEFAULTS.items():
        pairs = [(d[k], w) for d, w in zip(h, weights) if not pd.isna(d[k])]
        if not pairs:
            out[k] = default
        else:
            vals, ws = zip(*pairs)
            out[k] = float(np.average(vals, weights=ws))
    return out


def new_state():
    return {
        "elo": defaultdict(lambda: 1500.0),
        "history": defaultdict(lambda: deque(maxlen=FORM_WINDOW)),
        "last_date": {},                          # team -> last match Timestamp
        "season_pts": defaultdict(float),         # team -> points this season
        "season_played": defaultdict(int),        # team -> games played this season
    }


def rest_days(state, team, match_date) -> float:
    last = state["last_date"].get(team)
    if last is None:
        return DEFAULT_REST_DAYS
    return float(min(max((match_date - last).days, 0), MAX_REST_DAYS))


def row_features(home, away, match_date, state, rest=None) -> dict:
    """rest=(home_rest, away_rest) overrides the computed rest days (used by simulate.py)."""
    hf, af = form_avg(state["history"][home]), form_avg(state["history"][away])
    hr, ar = rest if rest is not None else (rest_days(state, home, match_date),
                                            rest_days(state, away, match_date))
    he, ae = state["elo"][home], state["elo"][away]
    home_played = state["season_played"][home]
    away_played = state["season_played"][away]
    home_ppg = (state["season_pts"][home] + 1.3 * PPG_PRIOR_GAMES) / (home_played + PPG_PRIOR_GAMES)
    away_ppg = (state["season_pts"][away] + 1.3 * PPG_PRIOR_GAMES) / (away_played + PPG_PRIOR_GAMES)
    return {
        "elo_diff": he - ae, "home_elo": he, "away_elo": ae,
        "home_form_pts": hf["pts"], "away_form_pts": af["pts"],
        "home_form_gf": hf["gf"], "home_form_ga": hf["ga"],
        "away_form_gf": af["gf"], "away_form_ga": af["ga"],
        "home_form_shots": hf["shots"], "away_form_shots": af["shots"],
        "home_form_sot": hf["sot"], "away_form_sot": af["sot"],
        "home_form_corners": hf["corners"], "away_form_corners": af["corners"],
        "home_form_xg": hf["xg"], "away_form_xg": af["xg"],
        "home_form_xga": hf["xga"], "away_form_xga": af["xga"],
        "home_rest_days": hr, "away_rest_days": ar,
        "home_season_ppg": home_ppg, "away_season_ppg": away_ppg,
    }


def _implied_probs(m) -> dict:
    odds = np.array([m.get("B365A", np.nan), m.get("B365D", np.nan), m.get("B365H", np.nan)], dtype=float)
    if np.isnan(odds).any() or (odds <= 1).any():
        return {"imp_A": np.nan, "imp_D": np.nan, "imp_H": np.nan}
    inv = 1 / odds
    inv = inv / inv.sum()
    return {"imp_A": inv[0], "imp_D": inv[1], "imp_H": inv[2]}


def _update(m, state):
    home, away, ftr, date = m["HomeTeam"], m["AwayTeam"], m["FTR"], m["Date"]
    elo, history = state["elo"], state["history"]
    he, ae = elo[home], elo[away]
    actual = {"H": 1.0, "D": 0.5, "A": 0.0}[ftr]
    expected = 1 / (1 + 10 ** (-((he + HOME_ADV) - ae) / 400))
    elo[home] = he + K_FACTOR * (actual - expected)
    elo[away] = ae + K_FACTOR * ((1 - actual) - (1 - expected))

    hp = {"H": 3, "D": 1, "A": 0}[ftr]
    ap = {"H": 0, "D": 1, "A": 3}[ftr]
    hg, ag = m.get("FTHG", np.nan), m.get("FTAG", np.nan)
    hxg, axg = m.get("home_xg", np.nan), m.get("away_xg", np.nan)
    history[home].append({"pts": hp, "gf": hg, "ga": ag,
                          "shots": m.get("HS", np.nan), "sot": m.get("HST", np.nan),
                          "corners": m.get("HC", np.nan), "xg": hxg, "xga": axg})
    history[away].append({"pts": ap, "gf": ag, "ga": hg,
                          "shots": m.get("AS", np.nan), "sot": m.get("AST", np.nan),
                          "corners": m.get("AC", np.nan), "xg": axg, "xga": hxg})

    state["season_pts"][home] += hp
    state["season_pts"][away] += ap
    state["season_played"][home] += 1
    state["season_played"][away] += 1
    state["last_date"][home] = date
    state["last_date"][away] = date


def _season_reset(state, new_teams, prev_teams):
    elo = state["elo"]
    for t in list(elo):
        if t in new_teams and t in prev_teams:              # stayed up: regress to mean
            elo[t] = 1500.0 + (elo[t] - 1500.0) * (1 - SEASON_REGRESS)
    for t in new_teams - prev_teams:                         # promoted / returning
        elo[t] = PROMOTED_ELO
        state["history"][t] = deque(maxlen=FORM_WINDOW)
    for t in new_teams:                                      # season-long PPG resets for everyone
        state["season_pts"][t] = 0.0
        state["season_played"][t] = 0


def replay(df: pd.DataFrame, record: bool = True):
    """Walk through matches chronologically. Returns (feature_df or None, state)."""
    state = new_state()
    season_teams = {f: set(g["HomeTeam"]) | set(g["AwayTeam"]) for f, g in df.groupby("season_file")}

    rows, cur_file, prev_teams = [], None, set()
    for m in df.to_dict("records"):
        f = m["season_file"]
        if f != cur_file:
            new_teams = season_teams[f]
            if cur_file is not None:
                _season_reset(state, new_teams, prev_teams)
            prev_teams, cur_file = new_teams, f

        if record:
            rows.append({
                "date": m["Date"], "season": f,
                "home_team": m["HomeTeam"], "away_team": m["AwayTeam"],
                **row_features(m["HomeTeam"], m["AwayTeam"], m["Date"], state),
                "result": m["FTR"], **_implied_probs(m),
            })
        _update(m, state)

    return (pd.DataFrame(rows) if record else None), state


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return replay(df, record=True)[0]


def replay_to_state(df: pd.DataFrame):
    """Returns the final `state` dict after replaying every match. Feed it into
    row_features(home, away, match_date, state) for simulation."""
    _, state = replay(df, record=False)
    return state


def current_standings(df: pd.DataFrame) -> pd.DataFrame:
    pts, played = defaultdict(int), defaultdict(int)
    for m in df.to_dict("records"):
        h, a, r = m["HomeTeam"], m["AwayTeam"], m["FTR"]
        played[h] += 1
        played[a] += 1
        pts[h] += {"H": 3, "D": 1, "A": 0}[r]
        pts[a] += {"H": 0, "D": 1, "A": 3}[r]
    table = pd.DataFrame({"team": list(played)})
    table["played"] = table["team"].map(played)
    table["points"] = table["team"].map(pts).astype(int)
    return table.sort_values("points", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    print(f"Loaded {len(raw)} raw matches")
    feats = build_features(raw)
    out_path = os.path.join(os.path.dirname(__file__), "data", "features.csv")
    feats.to_csv(out_path, index=False)
    print(f"Wrote {len(feats)} feature rows -> {out_path}")