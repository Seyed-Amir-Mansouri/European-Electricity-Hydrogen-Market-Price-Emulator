"""Data loading, caching, and prediction glue between the Django views and
``price_model``. Mirrors what the old Streamlit app did with ``st.cache_data`` /
``st.cache_resource``, using ``lru_cache`` instead -- each commodity's sample table and
trained bundle is loaded from disk once per server process and reused after that.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from price_model.config import COMMODITIES
from price_model.multivariate import predict
from price_model.neighbors import add_candidate_neighbor_prices, add_neighbor_features, load_adjacency

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS = REPO_ROOT / "inputs"
OUT = REPO_ROOT / "outputs"

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

DEFAULT_ZONE = {"electricity": "NL00", "hydrogen": "NL"}

DEMAND_UNIT = {"electricity": "MW", "hydrogen": "MWH2"}

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def zone_label(zone: str) -> str:
    return f"{COUNTRY_NAMES.get(zone[:2], zone)} ({zone})"


def model_available(commodity: str) -> bool:
    return (OUT / COMMODITIES[commodity]["model"]).exists()


@lru_cache(maxsize=None)
def load_samples(commodity: str) -> pd.DataFrame:
    """Raw samples enriched with each zone's own neighbor_demand_*/
    demand_system_total columns, plus its candidate price_<zone> columns -- same
    enrichment used at training time."""
    cfg = COMMODITIES[commodity]
    df = pd.read_parquet(INPUTS / cfg["samples"])
    adjacency = load_adjacency(INPUTS / cfg["adjacency"])
    enriched, _ = add_neighbor_features(df, cfg["demand"], adjacency, cfg.get("net_demand_col"))
    enriched, _ = add_candidate_neighbor_prices(enriched, cfg["target"], adjacency)
    return enriched


@lru_cache(maxsize=None)
def load_bundle(commodity: str) -> dict:
    return joblib.load(OUT / COMMODITIES[commodity]["model"])


def zone_choices(commodity: str) -> list[str]:
    return sorted(load_bundle(commodity)["zones"])


def default_zone(commodity: str) -> str:
    zones = zone_choices(commodity)
    want = DEFAULT_ZONE.get(commodity)
    return want if want in zones else zones[0]


def zone_metrics(commodity: str, zone: str) -> dict:
    bundle = load_bundle(commodity)
    e = bundle["zones"][zone]
    return {
        "n": e["n"],
        "cv_r2": e["cv_r2"],
        "cv_rmse": e["cv_rmse"],
        "unit": bundle["unit"],
    }


def _relabelled_datetime(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df["datetime"])
    return dt + pd.DateOffset(years=2030 - dt.dt.year.min())


def date_bounds(commodity: str) -> dict:
    dt = _relabelled_datetime(load_samples(commodity))
    data_min, data_max = dt.min().date(), dt.max().date()
    default_end = min(data_min + pd.Timedelta(days=7), data_max)
    return {
        "min": data_min.isoformat(),
        "max": data_max.isoformat(),
        "default_end": default_end.isoformat(),
    }


def history_series(commodity: str, zone: str, start: str | None, end: str | None) -> dict:
    bundle = load_bundle(commodity)
    target, unit = bundle["target"], bundle["unit"]

    period = load_samples(commodity)
    period = period[period["zone"] == zone].copy()
    period["datetime"] = _relabelled_datetime(period)

    bounds = date_bounds(commodity)
    start_date = pd.to_datetime(start).date() if start else pd.to_datetime(bounds["min"]).date()
    end_date = pd.to_datetime(end).date() if end else pd.to_datetime(bounds["default_end"]).date()
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")

    period = period[(period["datetime"].dt.date >= start_date) &
                     (period["datetime"].dt.date <= end_date)].sort_values("datetime")

    actual = period[target]
    emulated = predict(bundle, zone, period)

    return {
        "datetime": [d.isoformat() for d in period["datetime"]],
        "actual": [None if pd.isna(v) else float(v) for v in actual],
        "emulated": [None if pd.isna(v) else float(v) for v in emulated],
        "unit": unit,
    }


def _hourly_bucket_means(commodity: str, zone: str, month: int) -> pd.DataFrame:
    """24-row table (index 0..23, one row per hour-of-day), each column the mean of
    that zone-trained model feature over every historical hour in ``month`` that falls
    on that hour-of-day (active hours only, matching how the model itself was
    trained). This is the "typical day" for that month -- both the demand curve
    (``bundle["demand"]`` column) and every supporting driver come from here, so each
    hour gets its own realistic seasonal/diurnal context instead of one flat annual
    median. Missing hours (sparse zones) fall back to the zone's overall median."""
    bundle = load_bundle(commodity)
    entry = bundle["zones"][zone]
    features = entry["features"]
    demand_col = bundle["demand"]

    df = load_samples(commodity)
    d = df.loc[(df["zone"] == zone) & (df["month"] == month) & (df[demand_col] > 0)].copy()
    if d.empty:
        raise ValueError(f"No historical data for zone {zone!r} in month {month}.")
    d["hour_of_day"] = pd.to_datetime(d["datetime"]).dt.hour

    cols = [c for c in features if c in d.columns]
    grouped = d.groupby("hour_of_day")[cols].mean().reindex(range(24))
    medians = entry["medians"]
    for c in cols:
        grouped[c] = grouped[c].fillna(medians.get(c, float(d[c].mean())))
    return grouped


def _month_samples(commodity: str, zone: str, month: int) -> pd.DataFrame:
    """Every active historical hour (demand > 0) for zone+month -- the data behind
    both the hourly baselines above and the demand-response slopes below."""
    demand_col = COMMODITIES[commodity]["demand"]
    df = load_samples(commodity)
    d = df.loc[(df["zone"] == zone) & (df["month"] == month) & (df[demand_col] > 0)]
    if d.empty:
        raise ValueError(f"No historical data for zone {zone!r} in month {month}.")
    return d


def _demand_response_slopes(commodity: str, zone: str, month: int, cols: list[str]) -> dict[str, float]:
    """OLS slope of each column in ``cols`` against demand, fit on every active hour
    in zone+month (typically 700+ points, all hours of the month pooled together) --
    "historically, when demand moved by 1 MW here this month, this column moved by
    beta on average." This is what lets editing the demand curve realistically shift
    every other driver (price curve) instead of freezing them at their hourly
    average: e.g. thermal/imports typically show a
    strong positive slope (dispatched to meet demand) while wind/solar's slope is
    usually near zero (weather-driven, not demand-driven) -- exactly the kind of real
    behaviour a flat assumption can't capture."""
    demand_col = COMMODITIES[commodity]["demand"]
    d = _month_samples(commodity, zone, month)
    x = d[demand_col].to_numpy(dtype=float)
    x_centered = x - x.mean()
    x_var = float((x_centered ** 2).sum())

    slopes = {}
    for c in cols:
        if c not in d.columns or x_var == 0:
            slopes[c] = 0.0
            continue
        y = d[c].to_numpy(dtype=float)
        cov = float((x_centered * (y - y.mean())).sum())
        slopes[c] = cov / x_var
    return slopes


def monthly_demand_curve(commodity: str, zone: str, month: int) -> dict:
    """The initial editable 24-point demand curve for a zone + month."""
    bundle = load_bundle(commodity)
    demand_col = bundle["demand"]
    grouped = _hourly_bucket_means(commodity, zone, month)
    return {
        "hours": list(range(24)),
        "demand": [float(v) for v in grouped[demand_col]],
        "unit": DEMAND_UNIT[commodity],
    }


def predict_hourly_curve(commodity: str, zone: str, month: int, demand_values: list[float]) -> dict:
    """Hourly price for a (possibly user-edited) 24-point demand curve. Every other
    driver starts at its zone+month+hour-of-day historical average (see
    ``_hourly_bucket_means``) and then shifts with demand using that zone+month's
    fitted demand-response slope (``_demand_response_slopes``) -- e.g. if a neighbour
    zone's demand historically rose alongside this one, raising the curve raises that
    too, instead of freezing every driver except demand itself. Calendar context
    (month/season/hour) stays fixed -- editing demand doesn't change what time of
    year/day it is. Each shifted feature is clipped to this month's own observed
    range to avoid unrealistic extrapolation. ``residual_load`` (electricity) is
    recomputed from the shifted demand/wind/solar rather than shifted directly,
    since it's derived, not an independent driver."""
    if len(demand_values) != 24:
        raise ValueError("Expected 24 hourly demand values.")

    bundle = load_bundle(commodity)
    entry = bundle["zones"][zone]
    features = entry["features"]
    demand_col = bundle["demand"]

    grouped = _hourly_bucket_means(commodity, zone, month)
    baseline_demand = grouped[demand_col].to_numpy(dtype=float)
    delta = np.asarray(demand_values, dtype=float) - baseline_demand

    calendar = {"month", "season", "hour"}
    responsive = [f for f in features if f not in calendar and f not in (demand_col, "residual_load")]
    slopes = _demand_response_slopes(commodity, zone, month, responsive)
    month_df = _month_samples(commodity, zone, month)

    X = grouped[features].copy()
    for f in responsive:
        lo, hi = float(month_df[f].min()), float(month_df[f].max())
        X[f] = (X[f].to_numpy(dtype=float) + slopes[f] * delta).clip(lo, hi)
    X[demand_col] = demand_values
    if "residual_load" in features:
        X["residual_load"] = X[demand_col] - X.get("wind", 0.0) - X.get("solar", 0.0)

    prices = predict(bundle, zone, X)
    return {
        "hours": list(range(24)),
        "price": [float(p) for p in prices],
        "unit": bundle["unit"],
    }
