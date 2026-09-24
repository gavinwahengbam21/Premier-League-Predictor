"""
Full pipeline. Run:  python main.py
Then, after each matchweek:  python data_fetch.py --current && python simulate.py
(retrain occasionally with python main.py)
"""
import os
from data_fetch import fetch_all, fetch_current, season_codes
from features import load_raw, build_features
from model import train

if __name__ == "__main__":
    print("Step 1/3: fetching data...")
    fetch_all(season_codes(10))
    fetch_current()

    print("\nStep 2/3: building features...")
    feats = build_features(load_raw())
    out_path = os.path.join(os.path.dirname(__file__), "data", "features.csv")
    feats.to_csv(out_path, index=False)
    print(f"  {len(feats)} matches -> {out_path}")

    print("\nStep 3/3: training model...")
    train(out_path)

    print("\nDone. Now run: python simulate.py --n 50000")