# Price Formation

Two per-zone functions learned from the ENTSO-E TYNDP **NT2030 / PLEXOS / climate year
2009** market-model output (`inputs/MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx`)
that map **demand → price** for each European bidding zone:

| Commodity | Source sheet | Demand input | Price output |
|-----------|--------------|--------------|--------------|
| **Electricity** | `Hourly Market Data` | `Demand [MW]` | `Marginal Cost` → `EUR/MWh` |
| **Hydrogen** | `Hourly H2 Data` | `Demand [MWH2]` | `Marginal Cost` → `EUR/MWhH2` |

Each hour of the simulated year is one training sample. A separate gradient-boosted model
(`HistGradientBoostingRegressor`) is fitted **per zone**, so `electricity_price("DE00", …)`
and `electricity_price("FR00", …)` use different curves reflecting each market.

## Quick start

```python
from price_model import electricity_price, hydrogen_price, available_zones

electricity_price("DE00", 55000)              # 72.0  EUR/MWh   at 55 GW demand (median weather)
electricity_price("DE00", 55000, wind=20000)  # 76.4  EUR/MWh   overriding a driver
electricity_price("DE00", [30000, 50000, 70000])   # vectorised -> array of prices
hydrogen_price("DE", 1200)                    # 65.4  EUR/MWhH2  at 1200 MWH2 demand

available_zones("electricity")   # 63 zones ;  available_zones("hydrogen") -> 25 zones
```

**Demand is the only required argument.** Every other feature is optional and, if omitted,
defaults to that zone's median value over the simulated year — so a bare
`electricity_price(zone, demand)` returns the price at that demand under *typical* system
conditions. Pass any feature as a keyword (`wind=…`, `month=…`, `hour=…`) to set a specific
scenario.

## Available features

The model input for each commodity is its **demand** plus the supporting drivers below.
The *base* list (shared by every zone) is defined in
[`price_model/config.py`](price_model/config.py) — edit it there to add or drop features.
On top of the base list, every zone also gets its own **neighbour/system-total demand**
features (see below), so the *effective* feature list is per zone, not identical across
zones — a zone's own list lives at `bundle["zones"][zone]["features"]`.

Both base lists are deliberately narrow: demand, net demand, weather, calendar/time
context, and (for hydrogen) the electrolysis feedstock price — no dispatch-outcome
features (e.g. thermal generation, storage, balance), since those are simulation
*outputs*, not things a caller would actually know or forecast ahead of time.

### Electricity — `price_eur_mwh = F(...)` (base list)

| Feature | Meaning |
|---------|---------|
| `demand` | **Primary input.** Total electricity demand, losses included [MW] |
| `residual_load` | The zone's own **net demand**: `demand − wind − solar` [MW] |
| `wind` | Wind generation: onshore + offshore [MW] |
| `solar` | Solar generation: photovoltaic + thermal [MW] |
| `month` | Calendar month, 1–12 |
| `season` | 0 = winter (DJF) … 3 = autumn (SON) |
| `hour` | Absolute hour-of-year, 0–8735 |

### Hydrogen — `h2_price = F(...)` (base list)

| Feature | Meaning |
|---------|---------|
| `h2_demand` | **Primary input.** Hydrogen demand, losses included [MWH2] |
| `elec_price` | Electrolysis feedstock cost: demand-weighted electricity price in the matching country [EUR/MWh] |
| `month` | Calendar month, 1–12 |
| `season` | 0 = winter (DJF) … 3 = autumn (SON) |
| `hour` | Absolute hour-of-year, 0–8735 |

Hydrogen has no wind/solar equivalent tied to H2 zones, so there's no separate "net
demand" for it — `h2_demand` is the only demand quantity available.

### Neighbour / system-total demand (per zone, both commodities)

Added on top of the base list, from the workbook's crossborder-exchange sheets
(`price_model/extract.py::extract_adjacency`, `price_model/neighbors.py`):

| Feature | Meaning |
|---------|---------|
| `neighbor_demand_<N>` | Raw demand of each zone directly interconnected to this one, one column per real neighbour, named by the neighbour's own code (e.g. `neighbor_demand_DE00` for FR00) |
| `neighbor_net_demand_<N>` | Same, but net demand (`residual_load`) — **electricity only** |
| `neighbor_demand_total` / `neighbor_net_demand_total` | Sum across that zone's own neighbours |
| `demand_system_total` / `net_demand_system_total` | Sum across *every* zone for that commodity — same value for every zone at a given hour |

Neighbour count varies a lot per zone — median ~3, up to 17 for a hub like DE00 — so each
zone ends up with a different number of `neighbor_demand_*`/`neighbor_net_demand_*`
columns; there is no fixed, shared list across zones for these.

The extracted parquet tables also carry additional columns (`thermal`, `hydro`, `battery`,
`balance`, `dsr`, `vre`, `ens`, `dumped` for electricity; `electrolyser_gen`, `smr`,
`storage`, `hns`, `h2_net_trade` for hydrogen) — these are available for analysis but are
**not** model inputs.

## Why not demand alone?

Gross demand by itself is a weak price predictor. Price is set by weather and time context,
not raw demand:

* **Electricity** — wind & solar (near-zero marginal cost) shift the merit order; `month`,
  `season`, and `hour` capture seasonal and time-of-day structure.
* **Hydrogen** — `elec_price` (the electrolysis feedstock cost) is the single dominant driver;
  `month`/`season`/`hour` add further calendar context on top.

Keeping these supporting drivers lifts accuracy to the levels below, while `demand` stays the
headline knob you turn.

## Accuracy (5-fold cross-validated R², per zone)

| Commodity | Zones modelled | mean R² | median R² | RMSE (typical) | Strongest zones |
|-----------|---------------:|--------:|----------:|----------------|-----------------|
| Electricity | 63 | **0.75** | **0.80** | 5–15 EUR/MWh | UK00 0.98, IE00 0.94, NL00 0.93, FI00 0.93, ES00 0.92 |
| Hydrogen | 25 | **0.42** | **0.43** | 8–15 EUR/MWhH2 | GR 0.81, DK 0.72, FR 0.69, NL 0.69, UK 0.62 |

Hydrogen is a thinner, more administratively-priced market (some zones sit at a near-constant
price or a −1000 spill floor), so it is intrinsically harder to fit than electricity. Full
per-zone tables: `outputs/electricity_metrics.csv`, `outputs/hydrogen_metrics.csv`.

A zone is modelled only if it has ≥ 200 *active* hours (demand > 0); that is why 63 of 77
electricity zones and 25 of 32 hydrogen zones get a model.

## Model hyperparameters — generalization vs. memorization

Every zone fits a `sklearn.ensemble.HistGradientBoostingRegressor`
(`price_model/multivariate.py::_new_estimator`). Its hyperparameters aren't just tuning
knobs for accuracy — they set where the model sits on the bias/variance spectrum, from
"smooth, generalizes to unseen demand" to "memorizes the training year almost exactly."
The **live** config is deliberately regularized; pushed the other way, cross-validated
(out-of-sample) R² collapses even as training R² approaches 1.

| Hyperparameter | What it controls | Live value | To push toward R² ≈ 1 (memorization) |
|---|---|---|---|
| `max_leaf_nodes` | Max leaves per tree — the single biggest capacity knob. Low = each tree can only carve a few coarse regions; high = it can isolate small groups of near-identical rows. | `31` | `None` (no cap) |
| `min_samples_leaf` | Min. training rows a leaf must contain. Not set explicitly today, so it's sklearn's own default (`20`) — already a real regularizer even though it's invisible in the code. | *(default)* `20` | `1` (a single row can be its own leaf) |
| `l2_regularization` | Shrinks each leaf's predicted value toward its parent's — dampens the fit to noisy/idiosyncratic rows. | `1.0` | `0.0` |
| `learning_rate` | Shrinkage applied to each tree's correction. Low = conservative, needs many trees to reduce error; high = each tree fully applies its fix, converging faster to zero training error. | `0.06` | `1.0` |
| `max_iter` | Number of boosting trees. More trees = more chances to reduce remaining error, given enough leaf/depth capacity to use them. | `300` | higher (e.g. `1000+`), though less critical than the leaf/depth caps above |
| `max_depth` | Max tree depth. Not set today (sklearn default `None` = unlimited) — depth is currently bounded indirectly by `max_leaf_nodes`. | *(default)* `None` | `None` (already unconstrained — the leaf cap is what needs loosening) |

Measured effect of flipping just the first four rows (same feature set, same zones,
5-fold CV) — `no_regularization_test.py` in this repo:

| Commodity | | Live (regularized) | Unregularized |
|---|---|---:|---:|
| Electricity | train R² | 0.87 | **0.98** (median **1.00**) |
| Electricity | **CV R²** | **0.74** | 0.54 ↓ |
| Hydrogen | train R² | 0.81 | **0.96** |
| Hydrogen | **CV R²** | **0.70** | 0.55 ↓ |

The unregularized config memorizes the training rows — helped by `hour` being a unique
0–8735 index per row, so an uncapped tree can effectively key a leaf per hour — but it is
*worse*, not better, at pricing demand/weather conditions it wasn't shown. This is why the
live config trades some training-set fit for real predictive accuracy: an emulator that
only reproduces history it already has isn't useful for asking "what if" questions about
demand it hasn't seen. There is no hyperparameter setting that gets both R² = 1 *and*
genuine generalization at once — the two are in direct tension, because R² = 1 on this
data (real optimizer output, not noise) is only exactly recoverable by reconstructing the
underlying PLEXOS dispatch optimization itself (the generator-level merit-order LP), not
by fitting a statistical model harder.

## Pipeline

The sample tables `inputs/elec_samples.parquet` and `inputs/h2_samples.parquet` are
**committed to the repo**, so you skip extraction and go straight to training:

```bash
# 1. Train both demand->price models from the committed samples
#    -> outputs/{electricity,hydrogen}_model.joblib  (+ *_metrics.csv)
../projects-venv/Scripts/python.exe train_model.py
../projects-venv/Scripts/python.exe train_model.py --only hydrogen   # retrain just one

# 2. Explore in the browser: commodity + zone selector, importances,
#    predicted-vs-actual, and the demand->price curve
../projects-venv/Scripts/streamlit.exe run app.py
```

Training is the slow step (per-zone cross-validation + permutation importances):
electricity (63 zones) is the long pole at several minutes; hydrogen is quicker.

Regenerating the sample parquets from the raw ~100 MB workbook is a maintainer step
(`build_dataset.py`, not tracked in the repo); it streams each sheet once
(electricity ≈ 2.5 min, hydrogen ≈ 25 s).

## Repository layout

| Path | What |
|------|------|
| `inputs/*.parquet` | **committed** sample tables (`elec_samples`, `h2_samples`) — train straight from these |
| `inputs/…xlsx` | the raw source workbook (git-ignored, ~100 MB) |
| `price_model/config.py` | the two commodities: target, demand, feature list, output filenames — the single source of truth |
| `price_model/extract.py` | stream a sheet → tidy per-(zone, hour) feature table (`extract_electricity`, `extract_hydrogen`) |
| `price_model/multivariate.py` | per-zone gradient-boosted model, 5-fold CV scoring, permutation importances, `predict()` |
| `price_model/api.py` | **`electricity_price()`, `hydrogen_price()`, `available_zones()`** — the deliverable functions |
| `train_model.py` | sample parquets → the two model bundles + metric CSVs |
| `app.py` | Streamlit explorer |
| `outputs/` | generated `*_model.joblib`, `*_metrics.csv` (git-ignored) |
| `build_dataset.py` | workbook → sample parquets (maintainer-only regeneration, **not tracked**) |

## How the workbook is parsed

Each hourly sheet is a wide PLEXOS export: row 10 holds the technology/metric *Category*,
row 11 the zone *Country* code, and rows 13+ the 8,736 hourly values (one column per
zone × category). `extract.py` sums the raw categories into the feature groups above with a
single streaming pass (openpyxl read-only, so the 100 MB file never loads fully into memory).
Hydrogen zone codes carry an `_H2` suffix in the sheet, which is stripped (e.g. `AT_H2` → `AT`).
