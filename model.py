import sqlite3
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, TimeSeriesSplit,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DB_PATH = Path("data") / "secom.db"
CHART_DIR = Path("charts")
MODEL_DIR = Path("models")

TRAIN_FRACTION = 0.70
FLAG_RATE = 0.05
MAX_MISSING_PCT = 50
SPIKE = ("2008-09-29", "2008-10-12 23:59:59")
SUSPECTS = ["s059", "s103", "s510"]   # August excursion sensors from the SPC analysis
CORR_GROUP = 0.6                      # sensors correlated above this are scrambled together
N_BOOT = 1000
SEED = 42


SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
PASS_DOT, FAIL_RED, BAND = "#a3a29c", "#d03b3b", "#f0efec"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"


# Data

def load_features():
    with sqlite3.connect(DB_PATH) as con:
        meas = pd.read_sql("SELECT run_id, sensor, value FROM measurements", con)
        runs = pd.read_sql("SELECT run_id, timestamp, result FROM runs", con)

    X = meas.pivot(index="run_id", columns="sensor", values="value")
    runs["timestamp"] = pd.to_datetime(runs["timestamp"])
    runs = runs.set_index("run_id").loc[X.index]

    order = runs["timestamp"].sort_values(kind="stable").index   # oldest run first
    X, runs = X.loc[order], runs.loc[order]
    y = (runs["result"] == "fail").astype(int)
    return X, y, runs


def select_sensors(X_fit, max_missing_pct=MAX_MISSING_PCT):
    """Usable sensors judged on the fitting runs only, so later runs don't influence the choice."""
    missing_ok = X_fit.isna().mean() * 100 <= max_missing_pct
    varies = X_fit.nunique() > 1
    return X_fit.columns[missing_ok & varies].tolist()


# Models

def models():
    """Each model with the hyperparameter grid it is tuned over."""
    return {
        "Logistic regression": (
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),   # keep "was missing" as a signal
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=5000),
            ),
            {"logisticregression__C": [0.005, 0.01, 0.05, 0.1, 0.5]},
        ),
        # No class_weight: any sample weighting makes sklearn 1.9 bin features ~60x slower
        # (about a minute per fit), and it didn't change the ranking metrics on this data.
        "Gradient boosting": (
            HistGradientBoostingClassifier(learning_rate=0.1, min_samples_leaf=20,
                                           random_state=SEED),
            {"max_iter": [50, 100, 300], "max_leaf_nodes": [7, 15],
             "l2_regularization": [1.0, 10.0]},
        ),
    }


def tune(model, grid, X_fit, y_fit):
    """Pick hyperparameters with rolling time folds inside the training runs; the test runs are never seen."""
    search = GridSearchCV(model, grid, cv=TimeSeriesSplit(n_splits=4),
                          scoring="average_precision", n_jobs=-1)
    search.fit(X_fit, y_fit)
    return search.best_estimator_, search.best_params_


def train_threshold(model, X_fit, y_fit, flag_rate=FLAG_RATE):
    """Flag cutoff from out-of-fold scores on the fitting runs, fixed before any later run is scored."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clone(model), X_fit, y_fit, cv=cv, method="predict_proba")[:, 1]
    return float(np.quantile(oof, 1 - flag_rate))


# Scoring

def operating_point(y_true, scores, threshold):
    flagged = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return {
        "runs flagged": int(flagged.sum()),
        "failures caught (recall %)": round(100 * tp / max(tp + fn, 1), 1),
        "flags that were real fails (precision %)": round(100 * tp / max(tp + fp, 1), 1),
        "caught / total fails": f"{tp}/{tp + fn}",
        "false alarms": int(fp),
    }


def bootstrap_ci(y_true, scores, metric, n=N_BOOT, seed=SEED):
    """95% range of a metric when the test runs are resampled with replacement."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].any():
            vals.append(metric(y_true[i], scores[i]))
    return np.percentile(vals, [2.5, 97.5])


def score(name, method, y_true, scores, threshold=None, ci=False):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    if threshold is None:
        threshold = np.quantile(scores, 1 - FLAG_RATE)
    row = {
        "model": name, "test method": method,
        "test runs": len(y_true), "test fails": int(y_true.sum()),
        "ROC AUC": round(roc_auc_score(y_true, scores), 3),
        "PR AUC": round(average_precision_score(y_true, scores), 3),
        "PR AUC 95% range": "",
    }
    if ci:
        lo, hi = bootstrap_ci(y_true, scores, average_precision_score)
        row["PR AUC 95% range"] = f"{lo:.3f}–{hi:.3f}"
    row.update(operating_point(y_true, scores, threshold))
    return row


def random_baseline(y_true, n=N_BOOT, seed=SEED):
    """Random scores many times: the mean is chance level, the spread is how lucky chance can get."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    draws = [rng.random(len(y_true)) for _ in range(n)]
    roc = [roc_auc_score(y_true, d) for d in draws]
    pr = [average_precision_score(y_true, d) for d in draws]
    lo, hi = np.percentile(pr, [2.5, 97.5])
    row = {
        "model": "Random guessing", "test method": "time-ordered split",
        "test runs": len(y_true), "test fails": int(y_true.sum()),
        "ROC AUC": round(float(np.mean(roc)), 3), "PR AUC": round(float(np.mean(pr)), 3),
        "PR AUC 95% range": f"{lo:.3f}–{hi:.3f}",
    }
    # uniform scores: the top-10% cutoff is 0.9 by definition, no peeking at the test runs
    row.update(operating_point(y_true, draws[0], 1 - FLAG_RATE))
    return row, (lo, hi)


def rolling_eval(name, model, X_all, y, runs, n_splits=5):
    """Train on everything before each period, test on that period: several honest test windows."""
    rows = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X_all):
        cols = select_sensors(X_all.iloc[tr])
        m = clone(model).fit(X_all.iloc[tr][cols], y.iloc[tr])
        s = m.predict_proba(X_all.iloc[te][cols])[:, 1]
        yt, t = y.iloc[te], runs["timestamp"].iloc[te]
        ap = average_precision_score(yt, s)
        rows.append({
            "model": name, "test period": f"{t.iloc[0]:%b %d} – {t.iloc[-1]:%b %d}",
            "train runs": len(tr), "test fails": int(yt.sum()),
            "random PR AUC (fail rate)": round(yt.mean(), 3),
            "PR AUC": round(ap, 3), "lift over random": round(ap / yt.mean(), 2),
            "ROC AUC": round(roc_auc_score(yt, s), 3),
        })
    return rows


def correlated_groups(X_fit, min_corr=CORR_GROUP):
    """Cluster sensors whose readings move together (Spearman |r| above min_corr, average linkage).
    A sensor with no strong partner ends up in a group of one."""
    corr = X_fit.rank().corr().abs().fillna(0).to_numpy(copy=True)   # Pearson on ranks = Spearman
    np.fill_diagonal(corr, 1)
    dist = squareform(np.clip(1 - corr, 0, None), checks=False)
    labels = fcluster(linkage(dist, method="average"), t=1 - min_corr, criterion="distance")
    return pd.Series(X_fit.columns).groupby(labels).apply(list).tolist()


def group_label(cols, show=3):
    return " + ".join(cols[:show]) + (f" (+{len(cols) - show} more)" if len(cols) > show else "")


def group_importance(model, X_fit, y_fit, groups, n_splits=5, n_repeats=5):
    """Permutation importance per correlated group on held-out folds of the training runs.
    Scrambling a whole group at once stops the model leaning on a correlated partner,
    so signal shared between sensors isn't hidden."""
    rng = np.random.default_rng(SEED)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    drops = np.zeros((n_splits, len(groups)))
    for k, (tr, va) in enumerate(cv.split(X_fit, y_fit)):
        m = clone(model).fit(X_fit.iloc[tr], y_fit.iloc[tr])
        X_va, y_va = X_fit.iloc[va], y_fit.iloc[va]
        base = average_precision_score(y_va, m.predict_proba(X_va)[:, 1])
        for g, cols in enumerate(groups):
            scores = []
            for _ in range(n_repeats):
                X_perm = X_va.copy()
                X_perm[cols] = X_va[cols].to_numpy()[rng.permutation(len(X_va))]
                scores.append(average_precision_score(y_va, m.predict_proba(X_perm)[:, 1]))
            drops[k, g] = base - np.mean(scores)
    imp = pd.DataFrame({"group": [group_label(c) for c in groups],
                        "n sensors": [len(c) for c in groups],
                        "importance": drops.mean(axis=0),
                        "std across folds": drops.std(axis=0),
                        "members": [" ".join(c) for c in groups]})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


#Charts

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def pr_chart(y_test, test_scores, base_rate, luck_band):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    style(ax)
    ax.axhspan(*luck_band, color=BAND, lw=0, zorder=0,
               label=f"Random guessing, 95% luck range ({luck_band[0]:.2f}–{luck_band[1]:.2f})")
    for (name, s), color in zip(test_scores.items(), (SERIES_1, SERIES_2)):
        p, r, _ = precision_recall_curve(y_test, s)
        ap = average_precision_score(y_test, s)
        ax.plot(r, p, color=color, lw=2, label=f"{name}  (PR AUC {ap:.2f})")
    ax.axhline(base_rate, color=INK_2, lw=1, ls="--",
               label=f"Random guessing, average  ({base_rate:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall: share of failures caught", color=INK, fontsize=9)
    ax.set_ylabel("Precision: share of flags that were real failures", color=INK, fontsize=9)
    ax.set_title("Precision vs recall on later runs (time-ordered test)", loc="left",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "pr_curve.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def risk_timeline(test_runs, risk, threshold):
    t = test_runs["timestamp"]
    failed = (test_runs["result"] == "fail").to_numpy()
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    style(ax)
    ax.axvspan(pd.Timestamp(SPIKE[0]), pd.Timestamp(SPIKE[1]), color=BAND, lw=0, zorder=0)
    ax.scatter(t[~failed], risk[~failed], s=8, color=PASS_DOT, zorder=2, label="Passed run")
    ax.scatter(t[failed], risk[failed], s=40, marker="X", color=FAIL_RED, lw=0, zorder=3,
               label="Failed run")
    ax.axhline(threshold, color=INK_2, lw=1, ls="--", zorder=1)
    ax.text(t.max() + pd.Timedelta(hours=12), threshold,
            f"flag line\n(top {FLAG_RATE:.0%} of\ntraining runs)", fontsize=8, color=INK_2, va="bottom")
    ax.text(pd.Timestamp(SPIKE[0]), ax.get_ylim()[1], " second spike (Sep 29–Oct 12)",
            fontsize=8, color=INK_2, va="top")
    ax.set_xlim(t.min() - pd.Timedelta(days=1), t.max() + pd.Timedelta(days=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("risk score (ranks runs; not a probability)", color=INK, fontsize=9)
    ax.set_title("Gradient boosting risk score on test runs: do failures sit above the flag line?",
                 loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "risk_timeline.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def importance_chart(imp):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.8), facecolor=SURFACE)
    style(ax)
    colors = [FAIL_RED if set(m.split()) & set(SUSPECTS) else SERIES_1 for m in top["members"]]
    ax.barh(top["group"], top["importance"], xerr=top["std across folds"], color=colors,
            height=0.6, error_kw={"ecolor": INK_2, "lw": 0.8})
    for y_pos, v in enumerate(top["importance"]):
        ax.text(v, y_pos, f" {v:.3f}", va="bottom", fontsize=8, color=INK_2)
    ax.set_xlabel("drop in PR AUC when the group is scrambled together  (bars ± std across folds)",
                  color=INK, fontsize=9)
    fig.suptitle("What the model learned to rely on (held-out folds of training runs)\n"
                 f"sensors with Spearman |r| > {CORR_GROUP} grouped; "
                 "red = contains an August excursion sensor",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "feature_importance.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


#Main

if __name__ == "__main__":
    CHART_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    X_all, y, runs = load_features()
    cut = int(len(X_all) * TRAIN_FRACTION)
    sensors = select_sensors(X_all.iloc[:cut])
    X_train, X_test = X_all.iloc[:cut][sensors], X_all.iloc[cut:][sensors]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    test_runs = runs.iloc[cut:]
    print(f"Features: {len(sensors)} of {X_all.shape[1]} sensors kept (judged on training runs only)  |  "
          f"runs: {len(X_all)}  |  fails: {y.sum()} ({y.mean():.1%})")
    print(f"Train: {runs['timestamp'].iloc[0]:%b %d} – {runs['timestamp'].iloc[cut - 1]:%b %d}  "
          f"({len(X_train)} runs, {y_train.sum()} fails)")
    print(f"Test:  {runs['timestamp'].iloc[cut]:%b %d} – {runs['timestamp'].iloc[-1]:%b %d}  "
          f"({len(X_test)} runs, {y_test.sum()} fails)")

    baseline_row, luck_band = random_baseline(y_test)
    rows, rolling, test_scores, fitted, thresholds, tuned = [baseline_row], [], {}, {}, {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X_cv = X_all[select_sensors(X_all)]   # random CV is the deliberately leaky comparison

    for name, (base, grid) in models().items():
        print(f"Tuning {name} on training runs (time folds)...")
        model, tuned[name] = tune(base, grid, X_train, y_train)   # refit on all training runs

        cv_scores = cross_val_predict(clone(model), X_cv, y, cv=cv, method="predict_proba")[:, 1]
        rows.append(score(name, "random 5-fold CV", y, cv_scores))

        thresholds[name] = train_threshold(model, X_train, y_train)
        s = model.predict_proba(X_test)[:, 1]
        rows.append(score(name, "time-ordered split", y_test, s, thresholds[name], ci=True))
        test_scores[name], fitted[name] = s, model

        rolling += rolling_eval(name, model, X_all, y, runs)

    results = pd.DataFrame(rows)
    results.to_csv(Path("data") / "model_results.csv", index=False)
    rolling = pd.DataFrame(rolling)
    rolling.to_csv(Path("data") / "rolling_results.csv", index=False)

    gb = fitted["Gradient boosting"]
    risk = test_scores["Gradient boosting"]
    threshold = thresholds["Gradient boosting"]

    preds = test_runs[["timestamp", "result"]].copy()
    preds["risk"] = risk.round(4)
    preds["flagged"] = risk >= threshold
    preds.to_csv(Path("data") / "test_predictions.csv")

    # Deployable model: same settings, refit on every run, threshold from out-of-fold scores
    deploy_sensors = select_sensors(X_all)
    deploy = clone(gb).fit(X_all[deploy_sensors], y)
    gb_row = results[(results["model"] == "Gradient boosting")
                     & (results["test method"] == "time-ordered split")].iloc[0]
    joblib.dump({"model": deploy, "features": deploy_sensors,
                 "threshold": train_threshold(gb, X_all[deploy_sensors], y),
                 "params": tuned["Gradient boosting"],
                 "trained_through": str(runs["timestamp"].iloc[-1]),
                 "time_split_pr_auc": float(gb_row["PR AUC"]),
                 # train-only model that produced test_predictions.csv (explain test runs with this)
                 "eval_model": gb, "eval_features": sensors, "eval_threshold": threshold},
                MODEL_DIR / "model.joblib")

    groups = correlated_groups(X_train)
    print(f"\nComputing importance for {len(groups)} correlated sensor groups "
          f"({sum(len(g) > 1 for g in groups)} with 2+ sensors) on training folds...")
    imp = group_importance(gb, X_train, y_train, groups)
    imp.to_csv(Path("data") / "feature_importance.csv", index=False)

    pr_chart(y_test, test_scores, y_test.mean(), luck_band)
    risk_timeline(test_runs, risk, threshold)
    importance_chart(imp)

    #Report
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Tuned hyperparameters (chosen on training runs only) ===")
    for name, params in tuned.items():
        print(f"{name}: {params}")

    print("\n=== Ranking quality (1.0 = perfect; ROC AUC 0.5 and PR AUC = fail rate are random) ===")
    print("PR AUC 95% range: bootstrap over test runs for models; luck range over 1,000 draws for random.")
    print(results[["model", "test method", "test runs", "test fails", "ROC AUC", "PR AUC",
                   "PR AUC 95% range"]].to_string(index=False))

    print(f"\n=== If engineers review runs above a flag line set at the top {FLAG_RATE:.0%} of training runs ===")
    op_cols = ["model", "runs flagged", "caught / total fails", "failures caught (recall %)",
               "flags that were real fails (precision %)", "false alarms"]
    print(results[results["test method"] == "time-ordered split"][op_cols].to_string(index=False))

    print("\n=== Rolling time evaluation (train on all earlier runs, test on the next period) ===")
    print("Lift over random: 1.0 = no better than chance.")
    print(rolling.to_string(index=False))

    in_spike = test_runs["timestamp"].between(*SPIKE).to_numpy()
    spike_fail = y_test.to_numpy().astype(bool) & in_spike
    caught = (risk >= threshold) & spike_fail
    print(f"\n=== Second spike (Sep 29–Oct 12), gradient boosting ===")
    print(f"Failures in spike: {spike_fail.sum()}  |  caught by flag: {caught.sum()}")
    print(f"Mean risk score   in spike: {risk[in_spike].mean():.3f}   outside: {risk[~in_spike].mean():.3f}")

    print("\n=== Top 15 sensor groups the model learned to rely on (held-out training folds) ===")
    print(f"Sensors with Spearman |r| > {CORR_GROUP} are scrambled together, so partners can't mask each other.")
    print(imp.head(15).drop(columns="members").round(4).to_string(index=False))
    print(f"\nAugust excursion sensors (group rank of {len(imp)}):")
    for s_ in SUSPECTS:
        hit = imp[imp["members"].str.split().apply(lambda m: s_ in m)]
        if len(hit):
            print(f"  {s_}: rank {hit.index[0] + 1}, group = {hit['members'].iloc[0]}")
    print(f"\nCharts saved to {CHART_DIR.resolve()}")
