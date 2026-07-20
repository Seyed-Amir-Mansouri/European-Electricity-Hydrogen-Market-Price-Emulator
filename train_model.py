"""Train the per-zone demand -> price models for both commodities and save them.

For each commodity in ``price_model/config.py`` reads its feature table and writes:

* ``outputs/<commodity>_model.joblib`` -- the trained per-zone model bundle
* ``outputs/<commodity>_metrics.csv``  -- per-zone CV R^2, RMSE, n, top features

Usage:
    ../projects-venv/Scripts/python.exe train_model.py                 # both commodities
    ../projects-venv/Scripts/python.exe train_model.py --only hydrogen
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from price_model.config import COMMODITIES
from price_model.multivariate import train_all

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"   # pre-built sample parquets ship here (skip build_dataset)
OUT = ROOT / "outputs"     # trained models + metrics land here


def _metrics_frame(bundle: dict) -> pd.DataFrame:
    rows = []
    for zone, e in bundle["zones"].items():
        top = sorted(e["importances"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        rows.append({
            "zone": zone, "cv_r2": round(e["cv_r2"], 4),
            "cv_rmse": round(e["cv_rmse"], 2), "n": e["n"],
            "top_features": ", ".join(f"{k} ({v:.1f})" for k, v in top),
        })
    return pd.DataFrame(rows).sort_values("cv_r2", ascending=False)


def _summary(bundle: dict) -> str:
    r2 = np.array([e["cv_r2"] for e in bundle["zones"].values()])
    n = np.array([e["n"] for e in bundle["zones"].values()], float)
    return (f"{len(r2)} zones | mean CV R2 {r2.mean():.3f} | "
            f"sample-weighted R2 {np.average(r2, weights=n):.3f} | "
            f"median {np.median(r2):.3f}")


def train_commodity(name: str) -> None:
    cfg = COMMODITIES[name]
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUTS / cfg["samples"])
    bundle = train_all(df, name, cfg["target"], cfg["features"],
                       cfg["demand"], cfg["unit"])
    joblib.dump(bundle, OUT / cfg["model"])
    metrics = _metrics_frame(bundle)
    metrics.to_csv(OUT / cfg["metrics"], index=False)

    print(f"\n=== {name.upper()}  ({cfg['demand']} -> {cfg['target']}, {cfg['unit']}) ===")
    print("features:", ", ".join(cfg["features"]))
    print(_summary(bundle))
    print(f"wrote {cfg['model']}, {cfg['metrics']}")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(metrics.to_string(index=False))


def main(only: str | None = None) -> None:
    for name in ([only] if only else COMMODITIES):
        train_commodity(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(COMMODITIES), help="train just one commodity")
    args = ap.parse_args()
    main(args.only)
