"""Streamlit explorer for the two demand -> price models (electricity & hydrogen).

Pick a commodity and zone: cross-validated accuracy and a demand -> price
curve you can read off.

Run:  ../projects-venv/Scripts/streamlit.exe run app.py
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from price_model.config import COMMODITIES
from price_model.multivariate import predict
from price_model.neighbors import add_candidate_neighbor_prices, add_neighbor_features, load_adjacency
from price_model import api

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"   # pre-built sample parquets
OUT = ROOT / "outputs"     # trained models

# zone-code prefix -> country name, for a friendlier "Country (Code)" zone selector
COUNTRY_NAMES = {
    "AL": "Albania", "AT": "Austria", "BA": "Bosnia and Herzegovina", "BE": "Belgium",
    "BG": "Bulgaria", "CH": "Switzerland", "CY": "Cyprus", "CZ": "Czechia",
    "DE": "Germany", "DK": "Denmark", "DZ": "Algeria", "EE": "Estonia", "EG": "Egypt",
    "ES": "Spain", "FI": "Finland", "FR": "France", "GE": "Georgia", "GR": "Greece",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IL": "Israel", "IT": "Italy",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "LY": "Libya", "MA": "Morocco",
    "MD": "Moldova", "ME": "Montenegro", "MK": "North Macedonia", "MT": "Malta",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PS": "Palestine",
    "PT": "Portugal", "RO": "Romania", "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
    "SK": "Slovakia", "TN": "Tunisia", "TR": "Turkey", "UA": "Ukraine",
    "UK": "United Kingdom",
}


def zone_label(z: str) -> str:
    return f"{COUNTRY_NAMES.get(z[:2], z)} ({z})"

st.set_page_config(page_title="Demand → price models", layout="wide")

st.html("""
<style>
.st-key-section_model, .st-key-section_emulator, .st-key-section_history {
    background: rgba(127, 127, 127, 0.08);
    border: 1px solid rgba(127, 127, 127, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1.5rem 1.5rem;
    margin-bottom: 1.2rem;
}
</style>
""")

loading_placeholder = st.empty()
loading_placeholder.markdown(
    """
    <div style="position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
                z-index:9999; background:rgba(0,0,0,0.75); color:white;
                padding:24px 40px; border-radius:12px; font-size:20px;
                text-align:center;">
        ⏳ Processing...
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_samples(name):
    """Raw samples enriched with each zone's own neighbor_demand_*/
    demand_system_total columns, plus its candidate price_<zone> columns --
    same enrichment used at training time."""
    df = pd.read_parquet(INPUTS / COMMODITIES[name]["samples"])
    adjacency = load_adjacency(INPUTS / COMMODITIES[name]["adjacency"])
    enriched, _ = add_neighbor_features(df, COMMODITIES[name]["demand"], adjacency,
                                        COMMODITIES[name].get("net_demand_col"))
    enriched, _ = add_candidate_neighbor_prices(enriched, COMMODITIES[name]["target"], adjacency)
    return enriched


@st.cache_resource
def load_bundle(name):
    return joblib.load(OUT / COMMODITIES[name]["model"])



st.title("Price Emulator (Electricity & Hydrogen)")
st.caption("ENTSO-E TYNDP NT2030 / PLEXOS / CY2009")

st.sidebar.header("Model Selection")
name = st.sidebar.radio("Commodity", list(COMMODITIES), format_func=str.title)
cfg = COMMODITIES[name]

if not (OUT / cfg["model"]).exists():
    st.error(f"{cfg['model']} not found — run `build_dataset.py` then `train_model.py`.")
    st.stop()

df = load_samples(name)
bundle = load_bundle(name)
demand, target, unit = bundle["demand"], bundle["target"], bundle["unit"]
api._bundle.cache_clear()  # ensure the API reads the freshly trained bundle
zones = sorted(bundle["zones"])

default_zone = "NL00" if name == "electricity" else "NL"
choice = st.sidebar.selectbox("Selected Study Zone", zones, format_func=zone_label,
                              index=zones.index(default_zone) if default_zone in zones else 0)
e = bundle["zones"][choice]
features = e.get("features", bundle["features"])  # this zone's own feature list

d = df[df["zone"] == choice].dropna(subset=features + [target])
d = d[d[demand] > 0]
max_price = cfg.get("max_price")
if max_price is not None:
    d = d[d[target] <= max_price]

with st.container(key="section_model"):
    st.subheader(f"{name.title()} — {zone_label(choice)}")
    st.markdown("#### Trained Model Performance")
    st.caption("How well the trained model explains historical prices in this zone (5-fold cross-validation).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples (demand > 0)", f"{e['n']:,}")
    c2.metric("Cross-validated R²", f"{e['cv_r2']:.3f}")
    c3.metric("Cross-validated RMSE", f"{e['cv_rmse']:.2f} {unit}")

with st.container(key="section_emulator"):
    st.subheader("Forward-Looking Price Outlook")
    st.caption("Pick a demand level for this zone and the model emulates the resulting price, holding "
              "every other driver (weather, calendar, neighbor-zone conditions) at this zone's "
              "typical (median) historical value.")
    lo, hi = float(d[demand].min()), float(d[demand].max())
    hi_bound = hi if hi > lo else lo + 1.0
    default_q = float(d[demand].median())

    # reset the synced slider/number-input pair whenever the zone or commodity changes,
    # so a stale value from another zone can't fall outside the new min/max bounds
    zone_key = (name, choice)
    if st.session_state.get("_demand_zone_key") != zone_key:
        st.session_state.demand_slider = default_q
        st.session_state.demand_number = default_q
        st.session_state._demand_zone_key = zone_key

    def _sync_from_slider():
        st.session_state.demand_number = st.session_state.demand_slider

    def _sync_from_number():
        st.session_state.demand_slider = st.session_state.demand_number

    col_slider, col_num = st.columns([3, 1])
    col_slider.slider(f"Demand  (min {lo:,.0f} — max {hi_bound:,.0f})", lo, hi_bound,
                      key="demand_slider", on_change=_sync_from_slider)
    col_num.number_input("Type demand", min_value=lo, max_value=hi_bound,
                         key="demand_number", on_change=_sync_from_number)

    q = st.session_state.demand_slider
    price = api.electricity_price(choice, q) if name == "electricity" else api.hydrogen_price(choice, q)
    st.metric("Emulated price", f"{price:.2f} {unit}")


with st.container(key="section_history"):
    # ---- period comparison: actual vs. model emulation over a chosen date range ----
    st.subheader("Historical Validation — Emulated vs. Actual Price")
    st.caption("For the date range you pick, the emulator recomputes price hour-by-hour from that "
              "hour's real historical conditions, so you can see how closely it tracks the actual "
              "recorded price over time.")

    period_src = df[df["zone"] == choice].copy()
    period_src["datetime"] = pd.to_datetime(period_src["datetime"])
    # data is CY2009 weather mapped onto the NT2030 scenario -- relabel to 2030 for display
    period_src["datetime"] += pd.DateOffset(years=2030 - period_src["datetime"].dt.year.min())
    data_min = period_src["datetime"].min().date()
    data_max = period_src["datetime"].max().date()

    default_end = min(data_min + pd.Timedelta(days=7), data_max)
    date_range = st.date_input("Date range", value=(data_min, default_end),
                               min_value=data_min, max_value=data_max)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, tuple) and len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date = end_date = date_range

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()

    period = period_src[(period_src["datetime"].dt.date >= start_date) &
                        (period_src["datetime"].dt.date <= end_date)].sort_values("datetime")

    period["Actual"] = period[target]
    period["Emulated"] = predict(bundle, choice, period)

    series_order = ["Actual", "Emulated"]
    long = period.melt(id_vars=["datetime"], value_vars=series_order,
                       var_name="series", value_name="price")

    color_scale = alt.Scale(domain=series_order, range=["#2a78d6", "#eb6834"])
    autoscale = st.checkbox("Autoscale Y axis", value=False)

    if autoscale:
        pmin, pmax = float(long["price"].min()), float(long["price"].max())
        pad = (pmax - pmin) * 0.05 or 1.0
        y_scale = alt.Scale(domain=[pmin - pad, pmax + pad], nice=False, zero=False)
    else:
        y_scale = alt.Scale(zero=True)

    chart = alt.Chart(long).mark_line(strokeWidth=1.5).encode(
        x=alt.X("datetime:T", title="Date"),
        y=alt.Y("price:Q", title=f"Price [{unit}]", scale=y_scale),
        color=alt.Color("series:N", sort=series_order, scale=color_scale,
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("datetime:T", title="Date/hour"),
                "series", alt.Tooltip("price:Q", format=".1f")],
    ).properties(height=350)
    st.altair_chart(chart, width="stretch")

loading_placeholder.empty()
