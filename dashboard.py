from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from model import SPIKE, load_features

MODEL_PATH = Path("models") / "model.joblib"
PRED_PATH = Path("data") / "test_predictions.csv"
BASELINE = ("2008-09-01", "2008-09-28 23:59:59") 
TOP_DRIVERS = 10

FAIL_RED, TOWARD_PASS, PASS_GRAY, INK_2 = "#d03b3b", "#2a78d6", "#a3a29c", "#52514e"
MARK = "#6b6a66"


#Data
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle, shap.TreeExplainer(bundle["eval_model"])


@st.cache_data
def load_data():
    X_all, _, runs = load_features()
    preds = pd.read_csv(PRED_PATH, index_col=0, parse_dates=["timestamp"])
    preds["rank"] = preds["risk"].rank(ascending=False, method="min").astype(int)
    base = X_all[runs["timestamp"].between(*BASELINE)]
    limits = pd.DataFrame({"center": base.mean(), "sigma": base.std()})
    return X_all, runs, preds, limits


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def explain(explainer, X_all, features, run_id):
    row = X_all.loc[[run_id], features]
    contrib = explainer.shap_values(row)[0]
    base = float(np.ravel(explainer.expected_value)[0])
    drivers = pd.DataFrame({"sensor": features, "value": row.iloc[0].to_numpy(), "contribution": contrib})
    drivers = drivers.reindex(drivers["contribution"].abs().sort_values(ascending=False).index)
    return base, drivers.head(TOP_DRIVERS).reset_index(drop=True)


def run_label(preds, run_id):
    r = preds.loc[run_id]
    flag = "  ⚑" if r["flagged"] else ""
    return f"run {run_id} · {r['timestamp']:%b %d %H:%M} · {r['result'].upper()} · risk {r['risk']:.3f}{flag}"


#Charts
def risk_timeline(preds, run_id, threshold):
    data = preds.reset_index().assign(
        outcome=lambda d: d["result"].map({"fail": "Failed run", "pass": "Passed run"}),
        plot_risk=lambda d: d["risk"].clip(lower=1e-4))
    sel = data[data["run_id"] == run_id]
    tooltip = [alt.Tooltip("run_id:Q", title="run"), alt.Tooltip("timestamp:T", format="%b %d %H:%M"),
               alt.Tooltip("risk:Q", format=".4f"), alt.Tooltip("result:N")]
    y = alt.Y("plot_risk:Q", title="risk score (log scale; ranks runs, not a probability)",
              scale=alt.Scale(type="log", nice=False, padding=12))

    spike = alt.Chart(pd.DataFrame({"start": [pd.Timestamp(SPIKE[0])], "end": [pd.Timestamp(SPIKE[1])]})) \
        .mark_rect(color=MARK, opacity=0.12).encode(x="start:T", x2="end:T")
    points = alt.Chart(data).mark_point(filled=True, opacity=0.9).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%b %d")), y=y,
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["Passed run", "Failed run"],
                                                     range=[PASS_GRAY, FAIL_RED]),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("outcome:N", scale=alt.Scale(domain=["Passed run", "Failed run"],
                                                     range=["circle", "cross"]), legend=None),
        size=alt.condition(alt.datum.result == "fail", alt.value(70), alt.value(22)),
        tooltip=tooltip)
    flag = alt.Chart(pd.DataFrame({"t": [max(threshold, 1e-4)]})).mark_rule(
        color=INK_2, strokeDash=[4, 3]).encode(y="t:Q")
    ring = alt.Chart(sel).mark_point(size=320, strokeWidth=2.5, color=MARK, filled=False) \
        .encode(x="timestamp:T", y=y, tooltip=tooltip)
    return (spike + points + flag + ring).properties(height=260).interactive(bind_y=False)


def driver_chart(drivers):
    d = drivers.assign(
        label=lambda x: x["sensor"] + "  = " + x["value"].map(lambda v: "missing" if pd.isna(v) else f"{v:.4g}"),
        direction=lambda x: np.where(x["contribution"] > 0, "toward fail", "toward pass"))
    order = d["label"].tolist()
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=16).encode(
        x=alt.X("contribution:Q", title="contribution to risk (log-odds)"),
        y=alt.Y("label:N", sort=order, title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["toward fail", "toward pass"],
                                                       range=[FAIL_RED, TOWARD_PASS]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["sensor", alt.Tooltip("value:Q", format=".4g"),
                 alt.Tooltip("contribution:Q", format="+.4f"), "direction"])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK_2).encode(x="x:Q")
    return (bars + zero).properties(height=34 * len(d) + 30)


def control_chart(X_all, runs, limits, sensor, run_id):
    c, s = limits.loc[sensor, "center"], limits.loc[sensor, "sigma"]
    data = pd.DataFrame({"run_id": X_all.index, "timestamp": runs["timestamp"].to_numpy(),
                         "value": X_all[sensor].to_numpy(),
                         "outcome": runs["result"].map({"fail": "Failed run", "pass": "Passed run"}).to_numpy()})
    data = data.dropna(subset=["value"])
    data["z"] = (data["value"] - c) / s
    tooltip = [alt.Tooltip("run_id:Q", title="run"), alt.Tooltip("timestamp:T", format="%b %d %H:%M"),
               alt.Tooltip("value:Q", format=".4g"), alt.Tooltip("z:Q", format="+.2f", title="σ from center"),
               "outcome"]
    x = alt.X("timestamp:T", title=None, axis=alt.Axis(format="%b %d"))
    y = alt.Y("value:Q", title=sensor, scale=alt.Scale(zero=False))

    band = alt.Chart(pd.DataFrame({"start": [pd.Timestamp(BASELINE[0])], "end": [pd.Timestamp(BASELINE[1])]})) \
        .mark_rect(color=MARK, opacity=0.12).encode(x="start:T", x2="end:T")
    line = alt.Chart(data).mark_line(color=PASS_GRAY, strokeWidth=0.6, opacity=0.5).encode(x=x, y=y)
    points = alt.Chart(data).mark_point(filled=True).encode(
        x=x, y=y,
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["Passed run", "Failed run"],
                                                     range=[PASS_GRAY, FAIL_RED]),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("outcome:N", scale=alt.Scale(domain=["Passed run", "Failed run"],
                                                     range=["circle", "cross"]), legend=None),
        size=alt.condition(alt.datum.outcome == "Failed run", alt.value(60), alt.value(14)),
        tooltip=tooltip)
    lims = pd.DataFrame({"y": [c + 3 * s, c, c - 3 * s], "name": ["UCL", "CL", "LCL"],
                         "dash": [[4, 3], [1, 0], [4, 3]]})
    rules = alt.Chart(lims).mark_rule(color=INK_2).encode(y="y:Q", strokeDash=alt.StrokeDash("dash:N", legend=None,
                                                          scale=None))
    layers = [band, line, points, rules]
    sel = data[data["run_id"] == run_id]
    if len(sel):
        layers.append(alt.Chart(sel).mark_rule(color=MARK, strokeWidth=1, opacity=0.5).encode(x=x))
        layers.append(alt.Chart(sel).mark_point(size=320, strokeWidth=2.5, color=MARK,
                                                filled=False).encode(x=x, y=y, tooltip=tooltip))
    return alt.layer(*layers).properties(height=230).interactive()


#Page
st.set_page_config(page_title="Fab RCA run explorer", layout="wide")
bundle, explainer = load_model()
X_all, runs, preds, limits = load_data()
features, threshold = bundle["eval_features"], bundle["eval_threshold"]

st.title("Run explorer")
st.caption(f"Test runs {preds['timestamp'].min():%b %d} – {preds['timestamp'].max():%b %d}, scored by gradient "
           "boosting trained only on earlier runs. On this period the model ranks at about chance level, so "
           "treat the score as a triage hint, not a verdict.")

with st.sidebar:
    st.header("Pick a run")
    view = st.radio("Show", ["All test runs", "Failed runs", "Flagged runs", "October spike"])
    pool = preds
    if view == "Failed runs":
        pool = preds[preds["result"] == "fail"]
    elif view == "Flagged runs":
        pool = preds[preds["flagged"]]
    elif view == "October spike":
        pool = preds[preds["timestamp"].between(*SPIKE)]
    pool = pool.sort_values("risk", ascending=False)
    if pool.empty:
        st.warning("No runs in this view.")
        st.stop()
    run_id = st.selectbox(f"{len(pool)} runs, riskiest first", pool.index,
                          format_func=lambda r: run_label(preds, r))
    st.caption("⚑ = above the flag line")

r = preds.loc[run_id]
base, drivers = explain(explainer, X_all, features, run_id)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Risk score", f"{r['risk']:.3f}", help="Ranks runs against each other; not a calibrated probability.")
k2.metric("Rank", f"#{r['rank']} of {len(preds)}", f"top {100 * r['rank'] / len(preds):.0f}%",
          delta_color="off")
k3.metric("Flagged for review", "⚑ Yes" if r["flagged"] else "No", f"flag line {threshold:.3f}", delta_color="off")
k4.metric("Actual result", "✕ FAIL" if r["result"] == "fail" else "✓ PASS")

st.subheader("Where this run sits among test runs")
st.altair_chart(risk_timeline(preds, run_id, threshold), width="stretch")
st.caption("Ringed point = selected run. Dashed line = flag line. Shaded = October spike (Sep 29 – Oct 12). "
           "Scroll to zoom in time.")

st.subheader(f"Top {TOP_DRIVERS} drivers of this run's score")
st.caption(f"SHAP contributions in log-odds. Typical-run risk is {sigmoid(base):.3f}; red bars pushed this run's "
           "risk up, blue bars pushed it down.")
left, right = st.columns([3, 2])
left.altair_chart(driver_chart(drivers), width="stretch")
table = drivers.assign(
    baseline=lambda d: d["sensor"].map(limits["center"]),
    sigma_from_center=lambda d: (d["value"] - d["baseline"]) / d["sensor"].map(limits["sigma"]),
    direction=lambda d: np.where(d["contribution"] > 0, "↑ toward fail", "↓ toward pass"))
right.dataframe(
    table[["sensor", "value", "baseline", "sigma_from_center", "contribution", "direction"]],
    hide_index=True, width="stretch",
    column_config={"value": st.column_config.NumberColumn(format="%.4g"),
                   "baseline": st.column_config.NumberColumn("Sep baseline", format="%.4g"),
                   "sigma_from_center": st.column_config.NumberColumn("σ from baseline", format="%+.2f"),
                   "contribution": st.column_config.NumberColumn(format="%+.4f")})

st.subheader("Control charts")
pushing_up = drivers.loc[drivers["contribution"] > 0, "sensor"].tolist()
default = (pushing_up or drivers["sensor"].tolist())[:3]
chosen = st.multiselect("Sensors (defaults to the top drivers pushing risk up)", features, default=default)
st.caption("All runs Jul – Oct. Solid line = Sep 1–28 baseline mean (shaded period), dashed = ±3σ control limits. "
           "Vertical line and ring = selected run. Scroll to zoom, drag to pan, double-click to reset.")
for sensor in chosen:
    v = X_all.loc[run_id, sensor]
    c, s = limits.loc[sensor, "center"], limits.loc[sensor, "sigma"]
    if pd.isna(v):
        status = "missing on this run"
    else:
        z = (v - c) / s
        status = f"{v:.4g} on this run, {z:+.1f}σ from center — " + ("**outside** limits" if abs(z) > 3 else "inside limits")
    st.markdown(f"**{sensor}** · {status}")
    st.altair_chart(control_chart(X_all, runs, limits, sensor, run_id), width="stretch")
