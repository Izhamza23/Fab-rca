"""
Day 1 (part 2) — Which sensors separate passing runs from failing runs?

Reads data/secom.db (built by load_secom.py) and writes:
  data/sensor_ranking.csv   every usable sensor ranked by how differently it behaves on failed runs
  data/fails_by_week.csv    failure counts per week (do failures cluster in time?)

Run:  python explore_secom.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data") / "secom.db"


def load_wide(con):
    """Pivot the long measurements table back into one row per run, one column per sensor."""
    meas = pd.read_sql("SELECT run_id, sensor, value FROM measurements", con)
    wide = meas.pivot(index="run_id", columns="sensor", values="value")
    runs = pd.read_sql("SELECT run_id, timestamp, result FROM runs", con).set_index("run_id")
    return wide, runs


def rank_sensors(wide, runs, sensors):
    usable = sensors.loc[(sensors.is_constant == 0), "sensor"]
    failed = runs["result"] == "fail"

    rows = []
    for s in usable:
        col = wide[s]
        p, f = col[~failed], col[failed]
        pooled_std = col.std()
        effect = (f.mean() - p.mean()) / pooled_std if pooled_std and pooled_std > 0 else 0.0
        rows.append({
            "sensor": s,
            "mean_pass": p.mean(),
            "mean_fail": f.mean(),
            "effect_size": effect,                 # difference in std units; |d| > 0.3 is worth a look
            "missing_pct_pass": p.isna().mean() * 100,
            "missing_pct_fail": f.isna().mean() * 100,
        })

    out = pd.DataFrame(rows)
    out["missing_gap"] = out["missing_pct_fail"] - out["missing_pct_pass"]
    out["abs_effect"] = out["effect_size"].abs()
    return out.sort_values("abs_effect", ascending=False).drop(columns="abs_effect")


def fails_by_week(runs):
    r = runs.copy()
    r["week"] = pd.to_datetime(r["timestamp"]).dt.to_period("W").astype(str)
    weekly = r.groupby("week").agg(runs=("result", "size"),
                                   fails=("result", lambda x: (x == "fail").sum()))
    weekly["fail_rate_pct"] = (weekly["fails"] / weekly["runs"] * 100).round(1)
    return weekly


if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as con:
        wide, runs = load_wide(con)
        sensors = pd.read_sql("SELECT * FROM sensors", con)

    ranking = rank_sensors(wide, runs, sensors)
    ranking.to_csv(DB_PATH.parent / "sensor_ranking.csv", index=False)

    weekly = fails_by_week(runs)
    weekly.to_csv(DB_PATH.parent / "fails_by_week.csv")

    pd.set_option("display.width", 120)
    print(f"Usable (non-constant) sensors: {len(ranking)}")
    print("\n=== Top 15 sensors by pass/fail difference (effect size, std units) ===")
    print(ranking.head(15)[["sensor", "mean_pass", "mean_fail", "effect_size"]].round(3).to_string(index=False))

    print("\n=== Sensors that go missing more often on failed runs ===")
    gap = ranking[ranking["missing_gap"] > 5].sort_values("missing_gap", ascending=False)
    print(gap.head(10)[["sensor", "missing_pct_pass", "missing_pct_fail"]].round(1).to_string(index=False)
          if len(gap) else "None with a gap over 5 points.")

    print("\n=== Failures by week ===")
    print(weekly.to_string())
