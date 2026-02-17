import sys
import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Lovable Growth Command Center (POC)", layout="wide")
st.title("Lovable Growth Command Center (POC)")

OUT_DIR = Path("out")
FEATURES_PATH = OUT_DIR / "exports" / "features.csv"
COHORTS_PATH = OUT_DIR / "exports" / "cohorts_by_week.csv"
FUNNEL_PATH = OUT_DIR / "exports" / "funnel.csv"


@st.cache_resource
def build_outputs() -> str:
    """
    Streamlit Cloud starts from a clean container.
    If outputs are missing, run analysis once to generate out/exports.
    Uses sys.executable to guarantee we use the same Python env Streamlit is running.
    Returns stdout/stderr so failures show on the page.
    """
    try:
        res = subprocess.run(
            [sys.executable, "-m", "src.analysis", "--data_dir", "data/raw", "--out_dir", "out"],
            check=True,
            capture_output=True,
            text=True,
        )
        return f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
    except subprocess.CalledProcessError as e:
        return f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"


# Build step (runs once per container if outputs missing)
if not FEATURES_PATH.exists():
    with st.spinner("Preparing data (running analysis)…"):
        build_log = build_outputs()

    if not FEATURES_PATH.exists():
        st.error(f"Build failed while running: {sys.executable} -m src.analysis --data_dir data/raw --out_dir out")
        st.code(build_log)
        st.stop()

# Load features
feats = pd.read_csv(FEATURES_PATH)

with st.sidebar:
    st.header("Filters")
    acq = st.multiselect(
        "acquisition_channel",
        sorted(feats["acquisition_channel"].dropna().unique().tolist()),
    )
    intent = st.multiselect(
        "signup_intent",
        sorted(feats["signup_intent"].dropna().unique().tolist()),
    )
    platform = st.multiselect(
        "platform",
        sorted(feats["platform"].dropna().unique().tolist()),
    )
    country = st.multiselect(
        "country",
        sorted(feats["country"].dropna().unique().tolist()),
    )

f = feats.copy()
if acq:
    f = f[f["acquisition_channel"].isin(acq)]
if intent:
    f = f[f["signup_intent"].isin(intent)]
if platform:
    f = f[f["platform"].isin(platform)]
if country:
    f = f[f["country"].isin(country)]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Early activation (ship in 48h)", f"{f['activated_48h'].mean():.3f}")
c2.metric("7-day retention", f"{f['retained_7d'].mean():.3f}")
c3.metric("Template used (first 24h)", f"{f['template_used_flag'].mean():.3f}")

ship_rate = None
if FUNNEL_PATH.exists():
    funnel_tmp = pd.read_csv(FUNNEL_PATH)
    try:
        signup_n = float(funnel_tmp.loc[funnel_tmp["step"] == "signup", "users"].iloc[0])
        shipped_n = float(funnel_tmp.loc[funnel_tmp["step"] == "project_shipped", "users"].iloc[0])
        ship_rate = (shipped_n / signup_n) if signup_n > 0 else None
    except Exception:
        ship_rate = None

if ship_rate is None:
    ship_rate = (f["ship_ts"].notna().mean()) if "ship_ts" in f.columns else None

c4.metric("Ship rate (overall)", f"{ship_rate:.3f}" if ship_rate is not None else "n/a")

st.caption("Early activation = users who ship a project within 48 hours of signup (strict ‘time-to-value’ proxy).")

# Funnel
st.subheader("Funnel")
if FUNNEL_PATH.exists():
    funnel = pd.read_csv(FUNNEL_PATH)
    st.plotly_chart(px.funnel(funnel, x="users", y="step"), use_container_width=True)
else:
    st.info("Funnel output missing. Re-run analysis locally: python -m src.analysis --data_dir data/raw --out_dir out")

# Slice analysis
st.subheader("Activation by key slices")
slice_col = st.selectbox(
    "Slice by",
    ["template_used_flag", "error_bucket", "latency_bucket", "signup_intent", "acquisition_channel", "platform", "country"],
)
slice_df = (
    f.groupby(slice_col, dropna=False)["activated_48h"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "activation_rate", "count": "n"})
)
st.plotly_chart(px.bar(slice_df, x=slice_col, y="activation_rate", hover_data=["n"]), use_container_width=True)

# Cohorts
st.subheader("Cohorts (signup week)")
if COHORTS_PATH.exists():
    cohorts = pd.read_csv(COHORTS_PATH)
    st.dataframe(cohorts, use_container_width=True)
else:
    st.info("Cohorts output missing. Re-run analysis locally: python -m src.analysis --data_dir data/raw --out_dir out")

# Recommendations
st.subheader("What I would ship next")
tmpl = f.groupby("template_used_flag")["activated_48h"].mean()
tmpl_uplift = float(tmpl.get(1, 0)) - float(tmpl.get(0, 0))

err = f.groupby("error_bucket")["activated_48h"].mean()
err_impact = float(err.get("2+", 0)) - float(err.get("0", 0))

lat = f.groupby("latency_bucket")["activated_48h"].mean()
lat_impact = float(lat.get(">2000", 0)) - float(lat.get("<500", 0))

st.markdown(
    f"""- Template uplift (1 vs 0): **{tmpl_uplift:+.3f}**
- Errors impact (2+ vs 0): **{err_impact:+.3f}**
- Latency impact (>2000 vs <500): **{lat_impact:+.3f}**
"""
)

st.markdown(
    """**Recommended bets**
1) Intent-based template suggestions at first prompt  
2) Debug helper when users hit 2+ errors in first session  
3) Latency guardrails on first output path (p95 reduction)  
"""
)