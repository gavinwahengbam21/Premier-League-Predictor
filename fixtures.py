"""Maps my fixtures CSV onto football-data team names and removes games already played."""
from collections import Counter

import pandas as pd

NAME_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle", "Leeds United": "Leeds",
    "Hull City": "Hull", "Ipswich Town": "Ipswich", "Coventry City": "Coventry",
}


def remaining_fixtures(played: pd.DataFrame, fixtures_csv: str) -> pd.DataFrame:
    fx = pd.read_csv(fixtures_csv)

    fx["HomeTeam"] = fx["home_team"].replace(NAME_MAP)
    fx["AwayTeam"] = fx["away_team"].replace(NAME_MAP)

    # If this is an upcoming-only fixture file,
    # every row is already a remaining fixture.
    if len(fx) == 330:

        return fx[["matchweek", "date", "HomeTeam", "AwayTeam"]].reset_index(drop=True)

    # Otherwise assume it is a complete 380-fixture season
    if len(fx) != 380:
        print(
            f"WARNING: fixture file has {len(fx)} rows, "
            "expected either 330 upcoming or 380 full-season fixtures."
        )

    known = set(played["HomeTeam"]) | set(played["AwayTeam"])
    unknown = (set(fx["HomeTeam"]) | set(fx["AwayTeam"])) - known

    assert not unknown, (
        f"Team names not found in football-data file "
        f"(fix NAME_MAP): {unknown}"
    )

    fx_pairs = set(zip(fx["HomeTeam"], fx["AwayTeam"]))
    played_pairs = list(zip(played["HomeTeam"], played["AwayTeam"]))

    not_in_list = [p for p in played_pairs if p not in fx_pairs]

    if not_in_list:
        print("\nWARNING: played games with no matching fixture in my list:")
        for h, a in not_in_list:
            note = "  <- reversed in my list" if (a, h) in fx_pairs else ""
            print(f"   {h} v {a}{note}")

    done = set(played_pairs)

    rem = fx[
        [(h, a) not in done
         for h, a in zip(fx["HomeTeam"], fx["AwayTeam"])]
    ].reset_index(drop=True)

    print(
        f"Fixture list: {len(fx)} | "
        f"played matched: {len(fx) - len(rem)} | "
        f"remaining: {len(rem)}"
    )

    return rem[["matchweek", "date", "HomeTeam", "AwayTeam"]]