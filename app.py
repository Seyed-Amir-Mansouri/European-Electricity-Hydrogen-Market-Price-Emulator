"""Streamlit explorer for the two demand -> price models (electricity & hydrogen).

Pick a commodity and zone: cross-validated accuracy, feature importances, a
predicted-vs-actual scatter, and a demand -> price curve you can read off.

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
from price_model import api

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"   # pre-built sample parquets
OUT = ROOT / "outputs"     # trained models

st.set_page_config(page_title="Demand → price models", layout="wide")


@st.cache_data
def load_samples(name):
    return pd.read_parquet(INPUTS / COMMODITIES[name]["samples"])


@st.cache_resource
def load_bundle(name):
    return joblib.load(OUT / COMMODITIES[name]["model"])


st.title("Demand → price models — electricity & hydrogen")
st.caption("ENTSO-E TYNDP NT2030 / PLEXOS / CY2009 · per-zone gradient-boosted trees.")

name = st.sidebar.radio("Commodity", list(COMMODITIES), format_func=str.title)
cfg = COMMODITIES[name]

if not (OUT / cfg["model"]).exists():
    st.error(f"{cfg['model']} not found — run `build_dataset.py` then `train_model.py`.")
    st.stop()

df = load_samples(name)
bundle = load_bundle(name)
features, demand, target, unit = (bundle["features"], bundle["demand"],
                                  bundle["target"], bundle["unit"])
api._bundle.cache_clear()  # ensure the API reads the freshly trained bundle
zones = sorted(bundle["zones"])

choice = st.sidebar.selectbox("Core point (zone)", zones,
                              index=zones.index("DE00") if "DE00" in zones else 0)
e = bundle["zones"][choice]

d = df[df["zone"] == choice].dropna(subset=features + [target])
d = d[d[demand] > 0]

st.subheader(f"{name.title()} — {choice}")
st.caption(f"price = F(" + ", ".join(features) + f")   ·   units: {unit}")
c1, c2, c3 = st.columns(3)
c1.metric("Samples (demand > 0)", f"{e['n']:,}")
c2.metric("Cross-validated R²", f"{e['cv_r2']:.3f}")
c3.metric("Cross-validated RMSE", f"{e['cv_rmse']:.2f} {unit}")

left, right = st.columns(2)

# ---- feature importances ----
with left:
    st.markdown("**Feature importance**")
    imp = (pd.DataFrame({"feature": list(e["importances"]),
                         "importance": list(e["importances"].values())})
           .sort_values("importance", ascending=False))
    st.altair_chart(
        alt.Chart(imp).mark_bar().encode(
            x=alt.X("importance", title="permutation importance (Δ MSE)"),
            y=alt.Y("feature", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance", format=".2f")],
        ),
        width="stretch",
    )

# ---- predicted vs actual ----
with right:
    st.markdown("**Predicted vs actual**")
    sample = d.sample(min(len(d), 4000), random_state=0)
    pv = pd.DataFrame({"actual": sample[target].to_numpy(),
                       "predicted": predict(bundle, choice, sample)})
    lim = [float(min(pv.min())), float(max(pv.max()))]
    pts = alt.Chart(pv).mark_circle(size=14, opacity=0.25).encode(
        x=alt.X("actual", title=f"Actual price [{unit}]", scale=alt.Scale(domain=lim)),
        y=alt.Y("predicted", title=f"Predicted [{unit}]", scale=alt.Scale(domain=lim)),
    )
    diag = alt.Chart(pd.DataFrame({"actual": lim, "predicted": lim})).mark_line(
        color="#e4572e", strokeDash=[4, 4]).encode(x="actual", y="predicted")
    st.altair_chart(pts + diag, width="stretch")

# ---- demand -> price curve (other drivers held at their zone medians) ----
st.subheader("Demand → price")
st.caption("The headline relationship: sweep demand, other drivers held at the zone median.")
lo, hi = float(d[demand].min()), float(d[demand].max())
qs = np.linspace(lo, hi, 200)
if name == "electricity":
    preds = api.electricity_price(choice, qs)
else:
    preds = api.hydrogen_price(choice, qs)
curve = pd.DataFrame({demand: qs, "price": preds})
scatter = alt.Chart(d.sample(min(len(d), 4000), random_state=1)).mark_circle(
    size=10, opacity=0.15).encode(x=alt.X(demand, title=f"{demand} [{unit.split('/')[-1]}]"),
                                  y=alt.Y(target, title=f"price [{unit}]"))
line = alt.Chart(curve).mark_line(color="#e4572e", size=3).encode(x=demand, y="price")
st.altair_chart(scatter + line, width="stretch")

q = st.slider(f"{demand}", lo, hi if hi > lo else lo + 1.0, float(d[demand].median()))
price = api.electricity_price(choice, q) if name == "electricity" else api.hydrogen_price(choice, q)
st.metric("Predicted price", f"{price:.2f} {unit}")

with st.expander(f"All {name} zones — cross-validated accuracy"):
    rows = [{"zone": z, "cv_r2": round(v["cv_r2"], 3),
             "cv_rmse": round(v["cv_rmse"], 2), "n": v["n"]}
            for z, v in bundle["zones"].items()]
    st.dataframe(pd.DataFrame(rows).sort_values("cv_r2", ascending=False), width="stretch")
