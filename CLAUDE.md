# CLAUDE.md — Project 2

Two per-zone **demand → price** models learned from the ENTSO-E TYNDP **NT2030 / PLEXOS /
CY2009** `MMStandardOutputFile`:

* **electricity** — `Hourly Market Data`, `Demand [MW]` → `Marginal Cost [EUR/MWh]`
* **hydrogen** — `Hourly H2 Data`, `Demand [MWH2]` → `Marginal Cost [EUR/MWhH2]`

Public API (`price_model/api.py`): `electricity_price(zone, demand, **ctx)` and
`hydrogen_price(zone, h2_demand, **ctx)`. Demand is the only required arg; other features
default to the zone's median (stored in the bundle). See `README.md`.

## Environment
- Deps in the shared venv `../projects-venv` (use `../projects-venv/Scripts/python.exe`).
  scikit-learn + joblib added for the model.
- CPU lacks AVX2 → use `polars-lts-cpu` if polars is ever needed (this project uses pandas).
- Windows / PowerShell; Bash tool also available.
- **Commits: no Claude co-author trailer. Don't print git history after committing.**
- Pushes to its **own repo** (separate from Project 1) — URL TBD.

## Design
- `config.py::COMMODITIES` is the single source of truth: per commodity `target`, `demand`,
  `features`, and output filenames. Add/adjust features here.
- `extract.py` — one streaming pass per sheet (`_stream`), summing the ~55/9 raw Categories
  into feature groups. `extract_electricity()` and `extract_hydrogen()` (H2 zone codes are
  stripped of the `_H2` suffix). openpyxl read-only; elec pass ≈ 2.5 min, H2 ≈ 25 s.
- `multivariate.py` — commodity-agnostic: `train_all(df, commodity, target, features,
  demand, unit)`; per zone `HistGradientBoostingRegressor`, 5-fold shuffled CV (`cv_r2`,
  `cv_rmse`), permutation importances, and per-zone feature `medians`. Trains on active
  hours (`demand > 0`), needs ≥ `MIN_SAMPLES` (200). `predict(bundle, zone, X)`.
- `api.py` — builds a feature row from zone medians, overrides `demand` (+ any kwargs),
  recomputes derived `residual_load` if not supplied, predicts. Vectorised over demand.
- **Electrolyser load was removed** as a feature/input (the old design); demand replaced it.

## Key findings (why multivariate, not f(demand) alone)
- Gross demand alone predicts price poorly: electricity R² ≈ 0.04–0.30, hydrogen ≈ 0.
- Electricity price is driven by `residual_load` = demand − wind − solar (merit order).
  Multivariate electricity CV R² ≈ 0.6–0.98 (best UK00/ES00/NL00/DE00).
- Hydrogen price is driven by the supply mix (electrolyser_gen, smr, storage); multivariate
  CV R² ≈ 0.3–0.73 (best DK/AT/DE). Thinner market, some near-constant-price zones.
- Supporting features include dispatch *outcomes* (thermal, storage, balance) → holding them
  at median gives a clean demand-sweep curve, but a strict forecast should set them as inputs.

## Pipeline
`build_dataset.py` → `outputs/{elec_samples,h2_samples}.parquet`;
`train_model.py [--only <commodity>]` → `outputs/<commodity>_model.joblib` + `_metrics.csv`;
`app.py` (Streamlit) — commodity + zone selector, importances, predicted-vs-actual, and the
demand→price curve.
