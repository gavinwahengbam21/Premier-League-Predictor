"""
Trains a H/D/A classifier, calibrates it, and benchmarks it against
(a) always predicting base rates and (b) Bet365 closing odds.

Chronological split: 70% train / 15% calibration / 15% test.
"""
import os

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss

from features import FEATURE_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.pkl")
EXCLUDE_SEASONS = set()    # e.g. {"E0_2021.csv"} to drop the no-crowd season from training


def _apply_iso(isos, p):
    out = np.column_stack([isos[k].predict(p[:, k]) for k in range(p.shape[1])])
    out = np.clip(out, 1e-4, None)
    return out / out.sum(axis=1, keepdims=True)


def predict_proba(bundle, X: pd.DataFrame) -> np.ndarray:
    """Class order is bundle['model'].classes_ == ['A','D','H']."""
    p = bundle["model"].predict_proba(X[bundle["feature_cols"]])
    return _apply_iso(bundle["isos"], p) if bundle["use_iso"] else p


def train(features_csv: str = None):
    features_csv = features_csv or os.path.join(os.path.dirname(__file__), "data", "features.csv")
    df = pd.read_csv(features_csv, parse_dates=["date"])
    df = df[~df["season"].isin(EXCLUDE_SEASONS)].sort_values("date", kind="stable").reset_index(drop=True)

    n = len(df)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    tr, ca, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    classes = ["A", "D", "H"]

    def make(name):
        if name == "logreg":
            return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        return GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.03,
                                          subsample=0.8, random_state=0)

    scores = {}
    for name in ("logreg", "gbm"):
        m = make(name).fit(tr[FEATURE_COLS], tr["result"])
        scores[name] = log_loss(ca["result"], m.predict_proba(ca[FEATURE_COLS]), labels=classes)
        print(f"Calibration-slice log loss  {name}: {scores[name]:.4f}")
    best = min(scores, key=scores.get)
    print(f"Chosen: {best}")

    # test-set check (model fit on train only)
    y = te["result"]
    m_te = make(best).fit(tr[FEATURE_COLS], tr["result"])
    p = m_te.predict_proba(te[FEATURE_COLS])
    base = tr["result"].value_counts(normalize=True).reindex(classes).values
    print(f"\nTest log loss  base rates: {log_loss(y, np.tile(base, (len(te), 1)), labels=classes):.4f}")
    print(f"Test log loss  model:      {log_loss(y, p, labels=classes):.4f}")
    print(f"Test accuracy:             {accuracy_score(y, m_te.predict(te[FEATURE_COLS])):.3f}")
    mask = te[["imp_A", "imp_D", "imp_H"]].notna().all(axis=1)
    if mask.any():
        pb = te.loc[mask, ["imp_A", "imp_D", "imp_H"]].values
        print(f"On {mask.sum()} matches with odds -> bookmaker {log_loss(y[mask], pb, labels=classes):.4f} "
              f"| model {log_loss(y[mask], p[mask.values], labels=classes):.4f}")

    # final model: refit on train + calibration
    final = make(best).fit(pd.concat([tr, ca])[FEATURE_COLS], pd.concat([tr, ca])["result"])
    assert list(final.classes_) == classes
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": final, "isos": {}, "use_iso": False, "feature_cols": FEATURE_COLS}, MODEL_PATH)
    print(f"Saved {best} -> {MODEL_PATH}")
    return final


if __name__ == "__main__":
    train()