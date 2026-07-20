"""Multivariate per-zone price model:  price = F(demand, supporting drivers).

For each core point we train a gradient-boosted regression tree
(``HistGradientBoostingRegressor``) — the best-performing family in cross-validation by
a wide margin. Accuracy is reported as 5-fold (shuffled) cross-validated R^2 and RMSE,
so the numbers are honest out-of-sample.

The model is commodity-agnostic: the caller passes the ``target`` price column and the
list of ``features`` (see ``price_model/config.py``). ``demand`` is always the primary
feature; the rest are supporting context. Per-zone feature **medians** are stored in the
bundle so a caller can predict from demand alone (missing features fall back to median).

A trained bundle is a joblib-serialisable dict::

    {commodity, target, features, demand, unit,
     zones: {zone: {model, cv_r2, cv_rmse, n, importances, medians}}}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict

MIN_SAMPLES = 200  # a zone needs this many active hours for a stable fit


def _new_estimator() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
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
    model = _new_estimator().fit(X, y)  # final model on all of the zone's samples
    perm = permutation_importance(model, X, y, n_repeats=3, random_state=0)
    return {
        "model": model,
        "cv_r2": cv_r2, "cv_rmse": cv_rmse, "n": int(len(d)),
        "importances": dict(zip(features, (float(v) for v in perm.importances_mean))),
        "medians": {f: float(d[f].median()) for f in features},
    }


def train_all(df: pd.DataFrame, commodity: str, target: str, features: list[str],
              demand: str, unit: str) -> dict:
    """Train a model per zone. Returns a joblib-serialisable bundle."""
    zones = {}
    for zone, g in df.groupby("zone", sort=True):
        res = train_zone(g, target, features, demand)
        if res is not None:
            zones[zone] = res
    return {"commodity": commodity, "target": target, "features": features,
            "demand": demand, "unit": unit, "zones": zones}


def predict(bundle: dict, zone: str, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predict price for ``zone`` from a frame/array of the bundle's features."""
    entry = bundle["zones"][zone]
    if isinstance(X, pd.DataFrame):
        X = X[bundle["features"]].to_numpy()
    return entry["model"].predict(np.asarray(X, float))
