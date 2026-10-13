from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="LLM Drift Monitor", page_icon="📈", layout="wide")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_scores.csv"
METRICS = ["safety_score", "fairness_score", "factuality_score"]
METRIC_LABELS = {
    "safety_score": "Safety",
    "fairness_score": "Fairness",
    "factuality_score": "Factuality",
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    required = {
        "prompt_id", "run_id", "model_version", "language", "subgroup",
        "category", "safety_score", "fairness_score", "factuality_score"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    for col in METRICS + ["run_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=METRICS + ["run_id"])
    return df

def build_drift_events(df, threshold):
    """Compare the mean of each metric in consecutive runs, using percentage points."""
    events = []
    if df.empty:
        return pd.DataFrame(columns=[
            "From run", "To run", "Metric", "Previous mean",
            "Current mean", "Change (pp)", "Status"
        ])
    runs = sorted(df["run_id"].unique())
    for prev_run, curr_run in zip(runs, runs[1:]):
        prev = df[df["run_id"] == prev_run]
        curr = df[df["run_id"] == curr_run]
        for metric in METRICS:
            old = prev[metric].mean()
            new = curr[metric].mean()
            if pd.isna(old) or pd.isna(new):
                continue
            change = new - old
            if abs(change) >= threshold:
                events.append({
                    "From run": int(prev_run),
                    "To run": int(curr_run),
                    "Metric": METRIC_LABELS[metric],
                    "Previous mean": round(old, 1),
                    "Current mean": round(new, 1),
                    "Change (pp)": round(change, 1),
                    "Status": "Decline" if change < 0 else "Increase",
                })
    return pd.DataFrame(events, columns=[
        "From run", "To run", "Metric", "Previous mean",
        "Current mean", "Change (pp)", "Status"
    ])

try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load the sample CSV: {exc}")
    st.stop()

st.title("📈 LLM Drift Monitor")
st.caption("Longitudinal monitoring of safety, fairness and factuality")
st.warning(
    "Demo disclaimer: this dashboard uses simulated sample data. "
    "It demonstrates a monitoring workflow and does not prove real-world LLM drift."
)

with st.sidebar:
    st.header("Filters")
    languages = sorted(data["language"].dropna().unique().tolist())
    categories = sorted(data["category"].dropna().unique().tolist())
    groups = sorted(data["subgroup"].dropna().unique().tolist())
    selected_languages = st.multiselect("Language", languages, default=languages)
    selected_categories = st.multiselect("Prompt category", categories, default=categories)
    selected_groups = st.multiselect("Subgroup", groups, default=groups)
    threshold = st.slider("Drift threshold (percentage points)", 1, 25, 10)
    if st.button("Reset filters"):
        st.rerun()

filtered = data[
    data["language"].isin(selected_languages)
    & data["category"].isin(selected_categories)
    & data["subgroup"].isin(selected_groups)
].copy()

if filtered.empty:
    st.info("No rows match these filters. Select at least one language, category and subgroup.")
    st.stop()

run_means = (
    filtered.groupby(["run_id", "model_version"], as_index=False)[METRICS]
    .mean()
    .sort_values("run_id")
)
latest = run_means.iloc[-1]
previous = run_means.iloc[-2] if len(run_means) > 1 else None

st.subheader("Current evaluation snapshot")
kpi_cols = st.columns(3)
for col, metric in zip(kpi_cols, METRICS):
    value = float(latest[metric])
    delta = None if previous is None else value - float(previous[metric])
    col.metric(
        METRIC_LABELS[metric],
        f"{value:.1f} / 100",
        delta=None if delta is None else f"{delta:+.1f} pp vs previous run",
        delta_color="normal",
    )

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Behavior over evaluation runs")
    trend = run_means.melt(
        id_vars=["run_id", "model_version"],
        value_vars=METRICS,
        var_name="Metric",
        value_name="Score",
    )
    trend["Metric"] = trend["Metric"].map(METRIC_LABELS)
    fig = px.line(
        trend, x="model_version", y="Score", color="Metric",
        markers=True, range_y=[0, 100],
        labels={"model_version": "Model version", "Score": "Mean score"},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Latest subgroup comparison")
    latest_run = filtered["run_id"].max()
    subgroup_means = (
        filtered[filtered["run_id"] == latest_run]
        .groupby("subgroup", as_index=False)[METRICS].mean()
    )
    subgroup_long = subgroup_means.melt(
        id_vars="subgroup", value_vars=METRICS,
        var_name="Metric", value_name="Score"
    )
    subgroup_long["Metric"] = subgroup_long["Metric"].map(METRIC_LABELS)
    fig_sub = px.bar(
        subgroup_long, x="subgroup", y="Score", color="Metric",
        barmode="group", range_y=[0, 100],
        labels={"subgroup": "Test subgroup", "Score": "Mean score"},
    )
    st.plotly_chart(fig_sub, use_container_width=True)

st.subheader("Subgroup score heatmap")
heat_metric = st.selectbox("Metric for heatmap", METRICS, format_func=lambda x: METRIC_LABELS[x])
heat = filtered.pivot_table(
    index="subgroup", columns="model_version", values=heat_metric, aggfunc="mean"
)
if not heat.empty:
    fig_heat = px.imshow(
        heat, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn",
        zmin=0, zmax=100, labels=dict(x="Model version", y="Subgroup", color="Score"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.subheader("Potential drift events")
events = build_drift_events(filtered, threshold)
if events.empty:
    st.success(f"No consecutive-run changes reached the {threshold}-point threshold in the selected data.")
else:
    st.dataframe(events, use_container_width=True, hide_index=True)
    st.download_button(
        "Download drift events CSV",
        data=events.to_csv(index=False).encode("utf-8"),
        file_name="drift_events.csv",
        mime="text/csv",
    )

st.subheader("Plain-English summary")
if len(run_means) < 2:
    st.info("At least two evaluation runs are needed to compare drift.")
else:
    prev_run_id = run_means.iloc[-2]["run_id"]
    curr_run_id = run_means.iloc[-1]["run_id"]
    lines = []
    for metric in METRICS:
        old = float(run_means.iloc[-2][metric])
        new = float(run_means.iloc[-1][metric])
        change = new - old
        if abs(change) >= threshold:
            direction = "decreased" if change < 0 else "increased"
            lines.append(
                f"- **{METRIC_LABELS[metric]}** {direction} by "
                f"{abs(change):.1f} percentage points from run {int(prev_run_id)} "
                f"to run {int(curr_run_id)}."
            )
    if lines:
        st.markdown("\n".join(lines))
    else:
        st.write("No overall metric crossed the selected drift threshold between the last two runs.")

    # Group with largest absolute factuality change between the final two runs.
    runs = sorted(filtered["run_id"].unique())
    if len(runs) >= 2:
        old_id, new_id = runs[-2], runs[-1]
        per_group = filtered.groupby(["run_id", "subgroup"])["factuality_score"].mean().unstack(0)
        if old_id in per_group.columns and new_id in per_group.columns:
            changes = (per_group[new_id] - per_group[old_id]).dropna()
            if not changes.empty:
                group = changes.abs().idxmax()
                change = float(changes.loc[group])
                st.write(
                    f"**Largest observed subgroup factuality change:** {group}, "
                    f"{change:+.1f} percentage points between runs {int(old_id)} and {int(new_id)}."
                )

st.subheader("Filtered evaluation records")
st.dataframe(filtered, use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered data CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_evaluation_data.csv",
    mime="text/csv",
)
st.caption("Method note: threshold-based change detection is a prototype heuristic, not a formal statistical change-point test.")
