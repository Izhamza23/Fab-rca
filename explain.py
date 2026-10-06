import sqlite3
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

DB_PATH = Path("data") / "secom.db"
MODEL_PATH = Path("models") / "model.joblib"
PRED_PATH = Path("data") / "test_predictions.csv"
CHART_DIR = Path("charts")

TOP_N = 5                   # drivers listed per run
SPIKE = ("2008-09-29", "2008-10-12 23:59:59")
SEED = 42

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
TOWARD_FAIL, TOWARD_PASS, NEUTRAL = "#e34948", "#2a78d6", "#2a78d6"

def load_sensor_table(features):
    """Every run as one row, with the same sensor columns (and order) the model was trained on."""
    with sqlite3.connect(DB_PATH) as con:
        meas = pd.read_sql("SELECT run_id, sensor, value FROM measurements", con)
        runs = pd.read_sql("SELECT run_id, timestamp, result FROM runs", con)
    X = meas[meas["sensor"].isin(features)].pivot(index="run_id", columns="sensor", values="value")
    X = X.reindex(columns=features)                     # exact column order the model expects
    runs["timestamp"] = pd.to_datetime(runs["timestamp"])
    return X, runs.set_index("run_id").loc[X.index]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_explainer():
    """Explain the train-only model: it produced test_predictions.csv and never saw the test runs.
    (bundle["model"] is the deployment model, refit on every run including the test period.)"""
    bundle = joblib.load(MODEL_PATH)
    X, runs = load_sensor_table(bundle["eval_features"])
    preds = pd.read_csv(PRED_PATH, index_col=0, parse_dates=["timestamp"])
    explainer = shap.TreeExplainer(bundle["eval_model"])
    return explainer, bundle, X, runs, preds


def explain_run(explainer, X, run_id, top_n=TOP_N):
    row = X.loc[[run_id]]
    contrib = explainer.shap_values(row)[0]
    base = float(np.ravel(explainer.expected_value)[0])
    drivers = pd.DataFrame({
        "sensor": X.columns,
        "value": row.iloc[0].to_numpy(),
        "contribution": contrib,
    })
    drivers["direction"] = np.where(drivers["contribution"] > 0, "toward fail", "toward pass")
    drivers = drivers.reindex(drivers["contribution"].abs().sort_values(ascending=False).index)
    return base, float(sigmoid(base + contrib.sum())), drivers.head(top_n).reset_index(drop=True)


#Charts

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def global_chart(shap_df):
    impact = shap_df.abs().mean().sort_values(ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    style(ax)
    ax.barh(impact.index, impact.values, color=NEUTRAL, height=0.6)
    for y, v in enumerate(impact.values):
        ax.text(v, y, f" {v:.3g}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("average size of contribution, log-odds (either direction)", color=INK, fontsize=9)
    ax.set_title("Sensors with the biggest impact on predicted risk (test runs)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "shap_global.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def contribution_chart(labels, values, title, xlabel, path):
    """Horizontal bars: red = pushes toward fail, blue = pushes toward pass."""
    labels, values = list(labels)[::-1], np.asarray(values)[::-1]
    colors = [TOWARD_FAIL if v > 0 else TOWARD_PASS for v in values]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(values) + 1.6), facecolor=SURFACE)
    style(ax)
    ax.barh(labels, values, color=colors, height=0.6)
    ax.axvline(0, color=INK_2, lw=1)
    for y, v in enumerate(values):
        ax.text(v, y, f" {v:+.3g} ", va="center",
                ha="left" if v > 0 else "right", fontsize=8, color=INK_2)
    lo, hi = min(values.min(), 0), max(values.max(), 0)
    pad = (hi - lo) * 0.25 or 0.01
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.text(1, -0.22 if len(values) > 6 else -0.3, "red = pushes toward fail   blue = pushes toward pass",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


#Main

if __name__ == "__main__":
    CHART_DIR.mkdir(exist_ok=True)
    explainer, bundle, X, runs, preds = load_explainer()
    X_test = X.loc[preds.index]
    base = float(np.ravel(explainer.expected_value)[0])

    print(f"Explaining {len(X_test)} test runs across {X.shape[1]} sensors (takes a minute or two)...")
    shap_df = pd.DataFrame(explainer.shap_values(X_test), index=X_test.index, columns=X.columns)
    shap_df.round(5).to_csv(Path("data") / "shap_values.csv")
    model_out = bundle["eval_model"].decision_function(X_test)
    gap = np.abs(base + shap_df.sum(axis=1).to_numpy() - model_out).max()
    print(f"Check: average + contributions vs model output, largest gap = {gap:.1e} (should be ~0)")
    rows = []
    for run_id in X_test.index:
        s = shap_df.loc[run_id]
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(TOP_N)
        for rank, (sensor, c) in enumerate(top.items(), start=1):
            rows.append({"run_id": run_id, "timestamp": preds.loc[run_id, "timestamp"],
                         "result": preds.loc[run_id, "result"], "risk": preds.loc[run_id, "risk"],
                         "flagged": preds.loc[run_id, "flagged"], "rank": rank, "sensor": sensor,
                         "value": X_test.loc[run_id, sensor], "contribution": round(c, 5),
                         "direction": "toward fail" if c > 0 else "toward pass"})
    explanations = pd.DataFrame(rows)
    explanations.to_csv(Path("data") / "run_explanations.csv", index=False)

    global_chart(shap_df)
    failed = preds[preds["result"] == "fail"].sort_values("risk", ascending=False)
    print(f"\n=== Highest-risk failed runs: what drove their scores?  (typical-run risk = {sigmoid(base):.3f}) ===")
    print("contribution is in log-odds: + pushes toward fail, - toward pass")
    for run_id in failed.index[:3]:
        b, risk, drivers = explain_run(explainer, X, run_id, top_n=8)
        when = preds.loc[run_id, "timestamp"]
        print(f"\nRun {run_id}  ({when:%b %d %H:%M})  risk {risk:.3f}  "
              f"{'FLAGGED' if preds.loc[run_id, 'flagged'] else 'not flagged'}")
        print(drivers.round(4).to_string(index=False))
        contribution_chart(
            drivers["sensor"] + "  (= " + drivers["value"].map(lambda v: "missing" if pd.isna(v) else f"{v:.3g}") + ")",
            drivers["contribution"],
            f"Run {run_id} ({when:%b %d}): risk {risk:.2f} vs typical {sigmoid(b):.2f} — top 8 drivers",
            "contribution to fail risk (log-odds)", CHART_DIR / f"shap_run_{run_id}.png",
        )

    in_spike = preds["timestamp"].between(*SPIKE)
    spike = pd.DataFrame({
        "mean contribution in spike": shap_df[in_spike].mean(),
        "mean contribution outside": shap_df[~in_spike].mean(),
    })
    spike["difference"] = spike["mean contribution in spike"] - spike["mean contribution outside"]
    spike = spike.sort_values("difference", ascending=False)
    spike.round(5).to_csv(Path("data") / "shap_spike.csv")
    top_spike = spike.head(10)
    contribution_chart(top_spike.index, top_spike["difference"],
                       "Sensors pushing risk up during the Sep 29–Oct 12 spike (vs rest of test period)",
                       "extra contribution to fail risk inside the spike (log-odds)", CHART_DIR / "shap_spike.png")

    print(f"\n=== October spike: sensors pushing risk up more inside the spike than outside ===")
    print(f"Runs in spike: {in_spike.sum()}  |  mean risk in spike {preds.loc[in_spike, 'risk'].mean():.3f}"
          f"  vs outside {preds.loc[~in_spike, 'risk'].mean():.3f}")
    print(top_spike.round(4).to_string())

    print("\n=== Sensors with the biggest average impact (either direction) ===")
    print(shap_df.abs().mean().sort_values(ascending=False).head(10).round(4).to_string())
    print(f"\nCharts saved to {CHART_DIR.resolve()}")
