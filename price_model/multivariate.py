"""Multivariate per-zone price model:  price = F(demand, supporting drivers).

For each core point we train a gradient-boosted regression tree
(``HistGradientBoostingRegressor``) — the best-performing family in cross-validation by
a wide margin. Accuracy is reported as 5-fold (shuffled) cross-validated R^2 and RMSE,
so the numbers are honest out-of-sample.

The model is commodity-agnostic: the caller passes the ``target`` price column and the
shared list of base ``features`` (see ``price_model/config.py``). ``demand`` is always
the primary feature; the rest are supporting context. If an ``adjacency`` map is given,
each zone also gets its own ``neighbor_demand_<N>`` / ``neighbor_demand_total`` /
``demand_system_total`` columns (``price_model/neighbors.py``) appended to the base list
-- so the *effective* feature list is per zone, stored as each zone's own ``features``
entry rather than one shared list. Per-zone feature **medians** are stored in the bundle
so a caller can predict from demand alone (missing features fall back to median).

A trained bundle is a joblib-serialisable dict::

    {commodity, target, features, demand, unit,
     zones: {zone: {model, cv_r2, cv_rmse, n, importances, medians, features}}}

``features`` at the top level is the shared *base* list; each zone's own ``features``
entry is that base list plus its neighbour/system-total columns, and is what
``predict()`` actually uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict

from .neighbors import add_candidate_neighbor_prices, add_neighbor_features

MIN_SAMPLES = 200


def _new_estimator() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=20,
        l2_regularization=1.0, random_state=0,
    )


def _cv_scores(X: np.ndarray, y: np.ndarray, k: int = 5):
    """Cross-validated R^2 and RMSE via out-of-fold predictions (shuffled folds)."""
    kf = KFold(n_splits=min(k, len(y)), shuffle=True, random_state=0)
    pred = cross_val_predict(_new_estimator(), X, y, cv=kf)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y)))
    return r2, rmse


def train_zone(df_zone: pd.DataFrame, target: str, features: list[str], demand: str):
    """Train + CV-score one zone on its active hours (demand > 0)."""
    d = df_zone.dropna(subset=features + [target])
    d = d[d[demand] > 0]
    if len(d) < MIN_SAMPLES:
        return None
    X = d[features].to_numpy()
    y = d[target].to_numpy()

    cv_r2, cv_rmse = _cv_scores(X, y)
    model = _new_estimator().fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=3, random_state=0)
    return {
        "model": model,
        "cv_r2": cv_r2, "cv_rmse": cv_rmse, "n": int(len(d)),
        "importances": dict(zip(features, (float(v) for v in perm.importances_mean))),
        "medians": {f: float(d[f].median()) for f in features},
        "features": features,
    }


def train_all(df: pd.DataFrame, commodity: str, target: str, features: list[str],
              demand: str, unit: str, adjacency: dict[str, list[str]] | None = None,
              net_demand_col: str | None = None, max_price: float | None = None) -> dict:
    """Train a model per zone. Returns a joblib-serialisable bundle.

    If ``adjacency`` is given, each zone's feature list is ``features`` plus its own
    neighbour/system-total *net* demand columns (see ``price_model/neighbors.py``).
    ``net_demand_col`` picks which column that's based on (e.g. electricity's
    ``residual_load`` = demand - wind - solar); defaults to raw ``demand`` if unset.

    Each zone also unconditionally gets the top-5 system-wide most price-correlated
    zones' own price added as features, regardless of interconnection -- many zones
    move together with a zone they aren't directly wired to (shared weather, fuel
    costs, or transitive interconnection through a hub). This is now always applied
    (no baseline-vs-candidate comparison) -- a handful of zones with no well-correlated
    neighbour (e.g. GE00, PL00, MT00) may score marginally lower than a "keep whichever
    is better" policy would have given them, in exchange for one consistent feature set
    across every zone.

    If ``max_price`` is given, hours where *that zone's own* price exceeds it are
    dropped entirely -- from both training and CV scoring -- before anything else.
    """
    zone_extra: dict[str, list[str]] = {}
    if adjacency:
        df, zone_extra = add_neighbor_features(df, demand, adjacency, net_demand_col)

    neighbor_price_candidates: dict[str, dict[str, list[str]]] = {}
    if adjacency:
        df, neighbor_price_candidates = add_candidate_neighbor_prices(df, target, adjacency)

    zones = {}
    for zone, g in df.groupby("zone", sort=True):
        if max_price is not None:
            g = g[g[target] <= max_price]
        zone_features = features + zone_extra.get(zone, [])
        cols = neighbor_price_candidates.get(zone, {}).get("top_n", [])
        if cols:
            zone_features = zone_features + cols
        best = train_zone(g, target, zone_features, demand)
        if best is None:
            continue
        if cols:
            best["neighbor_price_method"] = "top_n"
        zones[zone] = best
    return {"commodity": commodity, "target": target, "features": features,
            "demand": demand, "unit": unit, "zones": zones}


def predict(bundle: dict, zone: str, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predict price for ``zone`` from a frame/array of that zone's own features."""
    entry = bundle["zones"][zone]
    feats = entry.get("features", bundle["features"])
    if isinstance(X, pd.DataFrame):
        X = X[feats].to_numpy()
    return entry["model"].predict(np.asarray(X, float))
