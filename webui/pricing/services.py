"""Data loading, caching, and prediction glue between the Django views and
``price_model``. Mirrors what the old Streamlit app did with ``st.cache_data`` /
``st.cache_resource``, using ``lru_cache`` instead -- each commodity's sample table and
trained bundle is loaded from disk once per server process and reused after that.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from price_model.config import COMMODITIES
from price_model.multivariate import predict
from price_model.neighbors import add_candidate_neighbor_prices, add_neighbor_features, load_adjacency

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS = REPO_ROOT / "inputs"
OUT = REPO_ROOT / "outputs"

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

DEFAULT_ZONE = {"electricity": "NL00", "hydrogen": "NL"}


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
    # data is CY2009 weather mapped onto the NT2030 scenario -- relabel to 2030 for display
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
