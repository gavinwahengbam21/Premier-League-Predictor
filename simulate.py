"""
Vectorised Monte Carlo of the rest of the season, with per-season team-strength noise.

Usage:
    python data_fetch.py --current
    python simulate.py --fixtures pl_2026_27_fixtures_all.csv --n 50000 --sigma 0.2
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

from features import load_raw, replay_to_state, current_standings, row_features, MAX_REST_DAYS, DEFAULT_REST_DAYS
from fixtures import remaining_fixtures, NAME_MAP
from model import MODEL_PATH, predict_proba

HERE = os.path.dirname(__file__)


def schedule_rest(fixtures_csv):
    """(team, date) -> days since that team's previous scheduled league game (capped)."""
    fx = pd.read_csv(fixtures_csv, parse_dates=["date"])
    long = pd.concat([
        pd.DataFrame({"team": fx["home_team"].replace(NAME_MAP), "date": fx["date"]}),
        pd.DataFrame({"team": fx["away_team"].replace(NAME_MAP), "date": fx["date"]}),
    ]).sort_values(["team", "date"])
    long["rest"] = long.groupby("team")["date"].diff().dt.days.clip(upper=MAX_REST_DAYS)
    long["rest"] = long["rest"].fillna(DEFAULT_REST_DAYS)
    return {(r.team, r.date.normalize()): float(r.rest) for r in long.itertuples()}


def run_sims(probs, hi, ai, base_pts, n_sims, rng, sigma=0.2, chunk=2000):
    n_fx, T = len(probs), len(base_pts)
    Hm = np.zeros((n_fx, T)); Hm[np.arange(n_fx), hi] = 1
    Am = np.zeros((n_fx, T)); Am[np.arange(n_fx), ai] = 1
    pA, pD, pH = probs[:, 0], probs[:, 1], probs[:, 2]      # class order A, D, H

    title = np.zeros(T); top4 = np.zeros(T); rel = np.zeros(T); pts_sum = np.zeros(T)
    done = 0
    while done < n_sims:
        b = min(chunk, n_sims - done)
        z = rng.normal(0, sigma, (b, T))
        d = z[:, hi] - z[:, ai]
        a_ = pA * np.exp(-d); h_ = pH * np.exp(d); d_ = pD * np.ones_like(d)
        tot = a_ + d_ + h_
        c1, c2 = a_ / tot, (a_ + d_) / tot
        u = rng.random((b, n_fx))
        idx = (u > c1).astype(int) + (u > c2).astype(int)   # 0=A, 1=D, 2=H
        hp = np.select([idx == 2, idx == 1], [3, 1], 0)
        ap = np.select([idx == 0, idx == 1], [3, 1], 0)
        pts = base_pts + hp @ Hm + ap @ Am
        key = pts + rng.random(pts.shape) * 0.5
        rank = np.argsort(np.argsort(-key, axis=1), axis=1)
        title += (rank == 0).sum(0); top4 += (rank < 4).sum(0)
        rel += (rank >= T - 3).sum(0); pts_sum += pts.sum(0)
        done += b
    return title / n_sims, top4 / n_sims, rel / n_sims, pts_sum / n_sims


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=os.path.join(HERE, "pl_2026_27_fixtures_all.csv"))
    ap.add_argument("--season", default="E0_2627.csv")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = load_raw()
    played = raw[raw["season_file"] == args.season]
    print(f"Played so far this season: {len(played)} matches")

    state = replay_to_state(raw)
    table = current_standings(played).set_index("team")
    rem = remaining_fixtures(played, args.fixtures)
    rest = schedule_rest(args.fixtures)

    teams = sorted(set(rem["HomeTeam"]) | set(rem["AwayTeam"]))
    tix = {t: i for i, t in enumerate(teams)}
    base_pts = np.array([table["points"].get(t, 0) for t in teams], dtype=float)

    bundle = joblib.load(MODEL_PATH)
    rows = []
    for h, a, d in zip(rem["HomeTeam"], rem["AwayTeam"], pd.to_datetime(rem["date"])):
        d = d.normalize()
        rows.append(row_features(h, a, d, state,
                                 rest=(rest.get((h, d), DEFAULT_REST_DAYS), rest.get((a, d), DEFAULT_REST_DAYS))))
    probs = predict_proba(bundle, pd.DataFrame(rows))
    assert np.allclose(probs.sum(axis=1), 1)

    hi = rem["HomeTeam"].map(tix).values
    ai = rem["AwayTeam"].map(tix).values
    rng = np.random.default_rng(args.seed)
    title, top4, rel, exp_pts = run_sims(probs, hi, ai, base_pts, args.n, rng, sigma=args.sigma)

    out = pd.DataFrame({
        "team": teams,
        "points_now": base_pts.astype(int), "exp_final_points": np.rint(exp_pts).astype(int),
        "title_%": (title * 100).round(1), "top4_%": (top4 * 100).round(1),
        "relegation_%": (rel * 100).round(1),
    }).sort_values(
        ["title_%", "exp_final_points"],
        ascending=[False, False]
    ).reset_index(drop=True)

    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "data", "title_odds.csv"), index=False)
    print("\nSaved data/title_odds.csv")