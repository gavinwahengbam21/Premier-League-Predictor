"""Fetches match-level xG from Understat (EPL) and maps team names to football-data names."""
import glob
import os
import time

import pandas as pd
from understatapi import UnderstatClient

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(HERE, "data", "xg_raw.csv")
SEASON_STARTS = list(range(2016, 2027))     # 2016 = 2016/17 ... 2026 = 2026/27

NAME_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "West Bromwich Albion": "West Brom",
    "Sheffield United": "Sheffield United", "Leeds": "Leeds", "Leeds United": "Leeds",
    "Leicester": "Leicester", "Norwich": "Norwich", "Ipswich": "Ipswich",
    "Hull": "Hull", "Coventry": "Coventry",
}

rows = []
with UnderstatClient() as us:
    for yr in SEASON_STARTS:
        try:
            matches = us.league(league="EPL").get_match_data(season=str(yr))
        except Exception as e:
            print(f"  {yr}: failed ({e})")
            continue
        n = 0
        for m in matches:
            if not m.get("isResult"):
                continue                                # skip unplayed games
            rows.append({
                "Date": pd.to_datetime(m["datetime"]).normalize(),
                "HomeTeam": NAME_MAP.get(m["h"]["title"], m["h"]["title"]),
                "AwayTeam": NAME_MAP.get(m["a"]["title"], m["a"]["title"]),
                "home_xg": float(m["xG"]["h"]), "away_xg": float(m["xG"]["a"]),
            })
            n += 1
        print(f"  {yr}/{str(yr + 1)[2:]}: {n} played matches")
        time.sleep(2)                                   # be polite

xg = pd.DataFrame(rows).drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
xg.to_csv(OUT_PATH, index=False)
print(f"\nWrote {len(xg)} rows -> {OUT_PATH}")

# ---- list matches that still have no xG (load_raw already merges xg_raw.csv) ----
from features import load_raw
raw = load_raw()
if "home_xg" in raw.columns:
    miss = raw[raw["home_xg"].isna()]
    print(f"{len(miss)} matches without xG:")
    print(miss[["Date", "HomeTeam", "AwayTeam", "season_file"]].to_string(index=False))
else:
    print("Columns in raw:", [c for c in raw.columns if "xg" in c.lower()])