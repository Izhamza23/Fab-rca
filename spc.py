
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DB_PATH = Path("data") / "secom.db"
CHART_DIR = Path("charts")

SENSORS = ["s059", "s103", "s510", "s158", "s431"]

# Periods identified from the weekly fail-rate table
BASELINE = ("2008-09-01", "2008-09-28 23:59:59")   # healthy line
PERIODS = {
    "excursion (Jul 19–Aug 24)": ("2008-07-19", "2008-08-24 23:59:59"),
    "baseline (Sep 1–28)":       BASELINE,
    "second spike (Sep 29–Oct 12)": ("2008-09-29", "2008-10-12 23:59:59"),
}

D2 = 1.128

SIGMA_METHOD = "std"

ACTIVE_RULES = (1, 2, 3)

# Sensors to check for moving together
PAIR = ("s059", "s103")
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

# Colours: neutral for normal runs; reserved "critical" red + X shape for failed runs
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
PASS_DOT = "#a3a29c"
FAIL_RED = "#d03b3b"
BAND = "#f0efec"

#data analysis

def load(con):
    placeholders = ",".join("?" * len(SENSORS))
    meas = pd.read_sql(
        f"SELECT run_id, sensor, value FROM measurements WHERE sensor IN ({placeholders})",
        con, params=SENSORS,
    )
    wide = meas.pivot(index="run_id", columns="sensor", values="value")
    runs = pd.read_sql("SELECT run_id, timestamp, result FROM runs", con).set_index("run_id")
    runs["timestamp"] = pd.to_datetime(runs["timestamp"])
    return runs.join(wide).sort_values("timestamp")

#SPC

def control_limits(baseline_values, method=SIGMA_METHOD):
    """Individuals chart limits: center = baseline mean, sigma by the chosen method."""
    x = baseline_values.dropna().to_numpy()
    center = x.mean()
    if method == "mr":
        sigma = np.abs(np.diff(x)).mean() / D2
    else:
        sigma = x.std(ddof=1)
    return center, sigma


def western_electric(z, rules=ACTIVE_RULES):
    n = len(z)
    flag = (np.abs(z) > 3) if 1 in rules else np.zeros(n, dtype=bool)

    def window_rule(width, need, threshold):
        for side in (1, -1):
            beyond = (side * z) > threshold
            for i in range(n - width + 1):
                w = beyond[i:i + width]
                if w.sum() >= need:
                    flag[i:i + width] |= w

    if 2 in rules:
        window_rule(3, 2, 2)
    if 3 in rules:
        window_rule(5, 4, 1)
    if 4 in rules:
        window_rule(8, 8, 0)
    return flag


def analyze(df, sensor, method=SIGMA_METHOD, rules=ACTIVE_RULES):
    series = df[["timestamp", "result", sensor]].dropna(subset=[sensor]).copy()
    in_base = series["timestamp"].between(*BASELINE)
    center, sigma = control_limits(series.loc[in_base, sensor], method)

    series["z"] = (series[sensor] - center) / sigma
    series["out_of_control"] = western_electric(series["z"].to_numpy(), rules)

    stats = {"sensor": sensor, "center": center, "sigma": sigma,
             "UCL": center + 3 * sigma, "LCL": center - 3 * sigma,
             "runs_measured": len(series)}
    for name, (start, end) in PERIODS.items():
        p = series[series["timestamp"].between(start, end)]
        stats[f"% OOC {name}"] = round(p["out_of_control"].mean() * 100, 1) if len(p) else np.nan

    ooc = series["out_of_control"]
    failed = series["result"] == "fail"
    stats["fail rate when OOC %"] = round(failed[ooc].mean() * 100, 1) if ooc.any() else np.nan
    stats["fail rate when in control %"] = round(failed[~ooc].mean() * 100, 1)
    return series, stats

#Chart representation

def shade_periods(ax, label=True):
    ymax = ax.get_ylim()[1]
    for name, (start, end) in PERIODS.items():
        if name.startswith("baseline"):
            continue
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=BAND, zorder=0, lw=0)
        if label:
            ax.text(pd.Timestamp(start), ymax, " " + name, va="top", ha="left",
                    fontsize=8, color=INK_2)


def draw_chart(ax, series, stats, sensor, compact=False):
    t = series["timestamp"]
    y = series[sensor]
    failed = series["result"] == "fail"
    ooc = series["out_of_control"]

    ax.set_facecolor(SURFACE)
    ax.plot(t, y, color=PASS_DOT, lw=0.6, alpha=0.5, zorder=1)
    ax.scatter(t[~failed], y[~failed], s=8, color=PASS_DOT, zorder=2, label="Passed run")
    ax.scatter(t[failed], y[failed], s=36, marker="X", color=FAIL_RED, lw=0, zorder=4, label="Failed run")
    ax.scatter(t[ooc], y[ooc], s=70, facecolors="none", edgecolors=INK, lw=1, zorder=3,
               label="Out of control (WE rules)")

    c, s = stats["center"], stats["sigma"]
    ax.axhline(c, color=INK_2, lw=1, zorder=1)
    for k in (3, -3):
        ax.axhline(c + k * s, color=INK_2, lw=1, ls="--", zorder=1)
    if not compact:
        right = t.max() + pd.Timedelta(days=1)
        ax.text(right, c + 3 * s, "UCL", va="bottom", fontsize=8, color=INK_2)
        ax.text(right, c, "CL", va="bottom", fontsize=8, color=INK_2)
        ax.text(right, c - 3 * s, "LCL", va="bottom", fontsize=8, color=INK_2)

    ax.set_xlim(t.min() - pd.Timedelta(days=1), t.max() + pd.Timedelta(days=4))
    shade_periods(ax, label=not compact)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel(sensor, color=INK, fontsize=9)


def single_chart(series, stats, sensor):
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    draw_chart(ax, series, stats, sensor)
    base_pct = stats["% OOC baseline (Sep 1–28)"]
    exc_pct = stats["% OOC excursion (Jul 19–Aug 24)"]
    ax.set_title(f"{sensor} individuals chart  ·  out of control: {exc_pct}% of runs in excursion "
                 f"vs {base_pct}% in baseline", loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / f"spc_{sensor}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def overview_chart(results):
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 2.1 * len(results)),
                             sharex=True, facecolor=SURFACE)
    for ax, (sensor, (series, stats)) in zip(axes, results.items()):
        draw_chart(ax, series, stats, sensor, compact=True)
    axes[0].set_title("SPC overview: top suspect sensors, limits from Sep 1–28 baseline  "
                      "(shaded = high-fail periods)", loc="left", fontsize=11, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(CHART_DIR / "spc_overview.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


#correlation of sensors graphed

def comovement(df):
    """Correlation between suspect sensors, and a daily view of the PAIR on a shared z-score scale."""
    corr = df[SENSORS].corr().round(2)

    a, b = PAIR
    daily = pd.DataFrame(index=df["timestamp"].dt.floor("D").unique()).sort_index()
    for s_ in PAIR:
        base = df.loc[df["timestamp"].between(*BASELINE), s_]
        z = (df[s_] - base.mean()) / base.std()
        daily[s_] = z.groupby(df["timestamp"].dt.floor("D")).median()
    daily["fail_rate_%"] = (df["result"].eq("fail")
                            .groupby(df["timestamp"].dt.floor("D")).mean() * 100).round(1)

    #Chart
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s_, color in zip(PAIR, (SERIES_1, SERIES_2)):
        d = daily[s_].dropna()
        ax.plot(d.index, d.values, color=color, lw=2, marker="o", ms=4, label=s_)
        ax.text(d.index[-1] + pd.Timedelta(days=1), d.values[-1], s_, color=INK_2,
                va="center", fontsize=8)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xlim(df["timestamp"].min() - pd.Timedelta(days=1), df["timestamp"].max() + pd.Timedelta(days=4))
    shade_periods(ax)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("daily median, std devs from Sep baseline", color=INK, fontsize=9)
    ax.set_title(f"Do {a} and {b} move together?  correlation r = {corr.loc[a, b]}",
                 loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "comovement.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return corr, daily

#main

if __name__ == "__main__":
    CHART_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        df = load(con)

    results, rows = {}, []
    for sensor in SENSORS:
        series, stats = analyze(df, sensor)
        results[sensor] = (series, stats)
        rows.append(stats)
        single_chart(series, stats, sensor)
    overview_chart(results)

    summary = pd.DataFrame(rows)
    summary.to_csv(DB_PATH.parent / "spc_summary.csv", index=False)

    pd.set_option("display.width", 160)
    cols = ["sensor"] + [c for c in summary.columns if c.startswith("% OOC") or c.startswith("fail rate")]
    print(f"=== Out-of-control (OOC) rate by period  [sigma={SIGMA_METHOD}, rules={ACTIVE_RULES}] ===")
    print(summary[cols].to_string(index=False))

    #Post fix calculations
    base_col = "% OOC baseline (Sep 1–28)"
    exc_col = "% OOC excursion (Jul 19–Aug 24)"
    textbook = pd.DataFrame([analyze(df, s_, method="mr", rules=(1, 2, 3, 4))[1] for s_ in SENSORS])
    compare = pd.DataFrame({
        "sensor": SENSORS,
        "baseline OOC % (textbook)": textbook[base_col].values,
        "baseline OOC % (fixed)": summary[base_col].values,
        "excursion OOC % (textbook)": textbook[exc_col].values,
        "excursion OOC % (fixed)": summary[exc_col].values,
    })
    print("\n=== Textbook limits (moving range + all 4 rules) vs fixed limits ===")
    print("A healthy baseline should only be a few % out of control.")
    print(compare.to_string(index=False))

    corr, daily = comovement(df)
    daily.to_csv(DB_PATH.parent / "daily_pair.csv")
    print("\n=== Correlation between suspect sensors (all runs) ===")
    print(corr.to_string())
    print(f"\n=== Daily median z-scores for {PAIR[0]} and {PAIR[1]}, Aug 18–Sep 3 (look for the step) ===")
    print(daily.loc["2008-08-18":"2008-09-03"].round(2).to_string())
    print(f"\nCharts saved to {CHART_DIR.resolve()}")

pd.read_sql("SELECT COUNT(*) FROM measurements", con)