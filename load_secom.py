"""
Day 1 — Load the UCI SECOM semiconductor dataset into SQLite.

Creates data/secom.db with three tables:
  runs          one row per production run (timestamp, pass/fail)
  measurements  one row per (run, sensor) reading  -> ~925k rows, NULL = missing
  sensors       one row per sensor with data-quality stats (missing %, constant, etc.)

Run:  python load_secom.py
"""
import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "secom.db"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
FILES = ["secom.data", "secom_labels.data"]


def download():
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        path = DATA_DIR / name
        if path.exists():
            continue
        print(f"Downloading {name} ...")
        try:
            urllib.request.urlretrieve(BASE_URL + name, path)
        except Exception as e:
            raise SystemExit(
                f"Couldn't download {name} ({e}).\n"
                "Download it manually from https://archive.ics.uci.edu/dataset/179/secom,\n"
                "unzip it, and put secom.data and secom_labels.data in the data/ folder."
            )


def load_raw():
    features = pd.read_csv(DATA_DIR / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    features.columns = [f"s{i:03d}" for i in range(features.shape[1])]

    labels = pd.read_csv(
        DATA_DIR / "secom_labels.data", sep=" ", header=None,
        names=["label", "timestamp"], quotechar='"',
    )
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")

    assert len(features) == len(labels), "feature and label row counts don't match"
    return features, labels


def build_tables(features, labels):
    run_ids = range(1, len(features) + 1)

    runs = pd.DataFrame({
        "run_id": run_ids,
        "timestamp": labels["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "result": labels["label"].map({-1: "pass", 1: "fail"}),
    })

    # Long format: one row per reading. Missing readings stay as NULL on purpose —
    # which sensors drop out, and when, can itself be a clue.
    wide = features.copy()
    wide.insert(0, "run_id", run_ids)
    measurements = wide.melt(id_vars="run_id", var_name="sensor", value_name="value")

    sensors = pd.DataFrame({
        "sensor": features.columns,
        "missing_pct": (features.isna().mean() * 100).round(2).values,
        "mean": features.mean().values,
        "std": features.std().values,
        "min": features.min().values,
        "max": features.max().values,
        "n_unique": features.nunique().values,
    })
    sensors["is_constant"] = (sensors["n_unique"] <= 1).astype(int)

    return runs, measurements, sensors


def write_db(runs, measurements, sensors):
    with sqlite3.connect(DB_PATH) as con:
        runs.to_sql("runs", con, if_exists="replace", index=False)
        sensors.to_sql("sensors", con, if_exists="replace", index=False)
        measurements.to_sql("measurements", con, if_exists="replace", index=False, chunksize=50_000)
        con.execute("CREATE INDEX IF NOT EXISTS idx_meas_sensor ON measurements(sensor)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_meas_run ON measurements(run_id)")


def summarize():
    with sqlite3.connect(DB_PATH) as con:
        q = lambda sql: pd.read_sql(sql, con)
        print("\n=== Runs by result ===")
        print(q("SELECT result, COUNT(*) AS n FROM runs GROUP BY result"))
        print("\n=== Date range ===")
        print(q("SELECT MIN(timestamp) AS first_run, MAX(timestamp) AS last_run FROM runs"))
        print("\n=== Data quality ===")
        print(q("""
            SELECT COUNT(*)                                   AS total_sensors,
                   SUM(is_constant)                           AS constant_sensors,
                   SUM(missing_pct > 50)                      AS over_50pct_missing,
                   SUM(missing_pct = 0)                       AS fully_populated
            FROM sensors
        """))
        print("\n=== Total readings ===")
        print(q("SELECT COUNT(*) AS readings, SUM(value IS NULL) AS missing FROM measurements"))


if __name__ == "__main__":
    download()
    features, labels = load_raw()
    print(f"Loaded {features.shape[0]} runs x {features.shape[1]} sensors")
    write_db(*build_tables(features, labels))
    print(f"Wrote {DB_PATH}")
    summarize()
