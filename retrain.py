from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from model import FLAG_RATE, SEED, SPIKE, load_features, models, select_sensors

FIRST_WEEK = "2008-08-11"
SLIDING_WEEKS = 6
N_BOOT = 1000
STRATEGIES = ["train once", "weekly, expanding", f"weekly, sliding {SLIDING_WEEKS} wk"]

PARAMS = {
    "Logistic regression": {"logisticregression__C": 0.005},
    "Gradient boosting": {"l2_regularization": 10.0, "max_iter": 50, "max_leaf_nodes": 7},
}


def flag_line(model, X_fit, y_fit, flag_rate=FLAG_RATE):
    cv = StratifiedKFold(n_splits=int(min(3, y_fit.sum())), shuffle=True, random_state=SEED)
    scores = cross_val_predict(clone(model), X_fit, y_fit, cv=cv, method="predict_proba")[:, 1]
    return float(np.quantile(scores, 1 - flag_rate))


def training_window(strategy, ts, week_start):
    if strategy == "train once":
        return ts < pd.Timestamp(FIRST_WEEK)
    if strategy == "weekly, expanding":
        return ts < week_start
    return (ts < week_start) & (ts >= week_start - pd.Timedelta(weeks=SLIDING_WEEKS))


def replay(name, base, X_all, y, runs, strategy):
    ts = runs["timestamp"]
    out, fitted = [], None
    for week_start in pd.date_range(FIRST_WEEK, ts.max(), freq="W-MON"):
        wk = ((ts >= week_start) & (ts < week_start + pd.Timedelta(weeks=1))).to_numpy()
        if not wk.any():
            continue
        fit = training_window(strategy, ts, week_start).to_numpy()
        retrain = fitted is None or (strategy != "train once" and y[fit].sum() >= 2)
        if retrain:
            cols = select_sensors(X_all[fit])
            X_fit, y_fit = X_all[fit][cols], y[fit]
            fitted = (clone(base).fit(X_fit, y_fit), cols, flag_line(base, X_fit, y_fit), int(fit.sum()))
        model, cols, threshold, n_train = fitted
        s = model.predict_proba(X_all[wk][cols])[:, 1]
        out.append(pd.DataFrame({
            "model": name, "strategy": strategy, "week": week_start, "train runs": n_train,
            "timestamp": ts[wk], "fail": y[wk].to_numpy(), "score": s, "flagged": s >= threshold,
        }))
    return pd.concat(out)


def summarize(preds):
    rows = []
    for (name, strategy), p in preds.groupby(["model", "strategy"], sort=False):
        fail, flagged = p["fail"].to_numpy() == 1, p["flagged"].to_numpy()
        tp, fp = int((fail & flagged).sum()), int((~fail & flagged).sum())
        precision = tp / max(tp + fp, 1)
        weeks = [w for _, w in p.groupby("week") if w["fail"].sum() > 0]
        lifts = [average_precision_score(w["fail"], w["score"]) / w["fail"].mean() for w in weeks]
        weights = [w["fail"].sum() for w in weeks]

        spike = p[p["timestamp"].between(*SPIKE)]
        spike_hits = spike[(spike["fail"] == 1) & spike["flagged"]]
        first = spike_hits["timestamp"].min()
        rows.append({
            "model": name, "strategy": strategy,
            "runs scored": len(p), "flagged": int(flagged.sum()),
            "caught / fails": f"{tp}/{int(fail.sum())}",
            "recall %": round(100 * tp / max(fail.sum(), 1), 1),
            "precision %": round(100 * precision, 1),
            "precision lift": round(precision / fail.mean(), 2),
            "weekly PR lift": round(float(np.average(lifts, weights=weights)), 2),
            "spike caught": f"{len(spike_hits)}/{int(spike['fail'].sum())}",
            "first spike catch": "never" if pd.isna(first)
                                 else f"{first:%b %d} (+{(first - pd.Timestamp(SPIKE[0])).days} d)",
        })
    return pd.DataFrame(rows)


def weekly_table(preds, name):
    """Per week: caught/fails (flags) for each strategy."""
    p = preds[preds["model"] == name]
    cell = p.groupby(["week", "strategy"], sort=False).apply(
        lambda w: f"{int((w['flagged'] & (w['fail'] == 1)).sum())}/{int(w['fail'].sum())} ({int(w['flagged'].sum())})",
        include_groups=False)
    table = cell.unstack("strategy")[STRATEGIES]
    table.insert(0, "runs", p[p["strategy"] == STRATEGIES[0]].groupby("week").size())
    table.index = table.index.strftime("%b %d")
    return table


def point_metrics(fail, flagged, score, week, in_spike):
    tp = int((fail & flagged).sum())
    precision = tp / max(int(flagged.sum()), 1)
    lifts, weights = [], []
    for wk in np.unique(week):
        m = week == wk
        if fail[m].any():
            lifts.append(average_precision_score(fail[m], score[m]) / fail[m].mean())
            weights.append(fail[m].sum())
    return {"recall %": 100 * tp / max(int(fail.sum()), 1),
            "precision lift": precision / fail.mean(),
            "weekly PR lift": float(np.average(lifts, weights=weights)),
            "spike caught": int((fail & flagged & in_spike).sum())}


def bootstrap(preds, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = {}
    for key, p in preds.groupby(["model", "strategy"], sort=False):
        day = p["timestamp"].dt.floor("D").to_numpy()
        days = np.unique(day)
        groups[key] = {
            "pos": [np.flatnonzero(day == d) for d in days],
            "fail": p["fail"].to_numpy() == 1, "flagged": p["flagged"].to_numpy(),
            "score": p["score"].to_numpy(), "week": p["week"].to_numpy(),
            "in_spike": p["timestamp"].between(*SPIKE).to_numpy(),
        }
    n_days = len(next(iter(groups.values()))["pos"])

    rows = []
    for b in range(n):
        pick = rng.integers(0, n_days, n_days)
        for (name, strategy), g in groups.items():
            i = np.concatenate([g["pos"][k] for k in pick])
            if not g["fail"][i].any():
                continue
            rows.append({"model": name, "strategy": strategy, "b": b,
                         **point_metrics(g["fail"][i], g["flagged"][i], g["score"][i],
                                         g["week"][i], g["in_spike"][i])})
    return pd.DataFrame(rows)


METRICS = ["recall %", "precision lift", "weekly PR lift", "spike caught"]


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return lo, hi


def interval_table(boot, summary):
    rows = []
    for _, s in summary.iterrows():
        b = boot[(boot["model"] == s["model"]) & (boot["strategy"] == s["strategy"])]
        row = {"model": s["model"], "strategy": s["strategy"]}
        for m in METRICS:
            point = s[m] if m != "spike caught" else int(s[m].split("/")[0])
            lo, hi = ci(b[m])
            row[m] = f"{point:.2f}  [{lo:.2f}, {hi:.2f}]" if m != "spike caught" else f"{point}  [{lo:.0f}, {hi:.0f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(boot, baseline="train once"):
    wide = boot.pivot_table(index=["b", "model"], columns="strategy", values=METRICS)
    rows = []
    for name in boot["model"].unique():
        w = wide.xs(name, level="model")
        for strategy in STRATEGIES:
            if strategy == baseline:
                continue
            for m in METRICS:
                d = (w[(m, strategy)] - w[(m, baseline)]).dropna()
                lo, hi = ci(d)
                rows.append({"model": name, "strategy": strategy, "metric": m,
                             "difference": round(d.mean(), 2),
                             "95% interval": f"[{lo:+.2f}, {hi:+.2f}]",
                             "excludes 0": "yes" if lo > 0 or hi < 0 else "no",
                             "better in % of resamples": round(100 * (d > 0).mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    X_all, y, runs = load_features()
    start = runs["timestamp"] < pd.Timestamp(FIRST_WEEK)
    print(f"Starting history: {runs['timestamp'].iloc[0]:%b %d} – {pd.Timestamp(FIRST_WEEK) - pd.Timedelta(days=1):%b %d}  "
          f"({start.sum()} runs, {y[start].sum()} fails)")
    print(f"Scored: {FIRST_WEEK} onward ({(~start).sum()} runs, {y[~start].sum()} fails), "
          f"flag line = top {FLAG_RATE:.0%} of each training window")

    parts = []
    for name, (base, _) in models().items():
        base = base.set_params(**PARAMS[name])
        for strategy in STRATEGIES:
            print(f"Replaying {name}: {strategy}...")
            parts.append(replay(name, base, X_all, y, runs, strategy))
    preds = pd.concat(parts)
    preds.to_csv(Path("data") / "retrain_predictions.csv")
    summary = summarize(preds)
    summary.to_csv(Path("data") / "retrain_summary.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== Production replay ===")
    print(f"Random flagging would give precision lift 1.0 and catch about {FLAG_RATE:.0%} of fails.")
    print("Weekly PR lift: PR AUC / fail rate inside each week, averaged weighted by fails (1.0 = chance).")
    print(summary.to_string(index=False))

    for name in PARAMS:
        print(f"\n=== {name}: week by week, caught/fails (runs flagged) ===")
        print(weekly_table(preds, name).to_string())

    print(f"\nBootstrapping {N_BOOT} resamples of whole days...")
    boot = bootstrap(preds)
    intervals = interval_table(boot, summary)
    paired = paired_table(boot)
    intervals.to_csv(Path("data") / "retrain_intervals.csv", index=False)
    paired.to_csv(Path("data") / "retrain_paired.csv", index=False)

    print("\n=== 95% intervals: value  [low, high], resampling whole days ===")
    print(intervals.to_string(index=False))
    print("\n=== Does retraining beat train-once? Paired differences on the same resampled days ===")
    print("If the interval excludes 0, the difference is unlikely to be luck.")
    print(paired.to_string(index=False))
