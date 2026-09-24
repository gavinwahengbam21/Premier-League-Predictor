"""
Downloads Premier League match data from football-data.co.uk.
Historical seasons are cached; the CURRENT season file is re-downloaded
every run so new results come in.

Usage:
    python data_fetch.py                # last 10 completed seasons + current
    python data_fetch.py --current      # only refresh the current season
"""
import argparse
import os
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
CURRENT_SEASON = "2627"   # bump each August
LAST_COMPLETED_END_YEAR = 26


def season_codes(n_seasons: int = 10) -> list[str]:
    codes = []
    for i in range(n_seasons):
        y2 = LAST_COMPLETED_END_YEAR - i
        codes.append(f"{y2 - 1:02d}{y2:02d}")
    return list(reversed(codes))


def fetch_season(season: str, force: bool = False) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"E0_{season}.csv")
    if os.path.exists(out_path) and not force:
        return out_path
    resp = requests.get(BASE_URL.format(season=season), timeout=30)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


def fetch_all(seasons: list[str]) -> list[str]:
    paths = []
    for s in seasons:
        try:
            paths.append(fetch_season(s))
            print(f"  fetched {s}")
        except Exception as e:
            print(f"  WARNING: could not fetch season {s}: {e}")
    return paths


def fetch_current() -> str | None:
    try:
        p = fetch_season(CURRENT_SEASON, force=True)
        print(f"  refreshed current season {CURRENT_SEASON} -> {p}")
        return p
    except Exception as e:
        print(f"  WARNING: could not refresh current season: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", action="store_true", help="only refresh the current season")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    if not args.current:
        fetch_all(season_codes(args.n))
    fetch_current()