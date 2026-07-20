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
conditions. Pass any feature as a keyword (`wind=…`, `thermal=…`, `smr=…`) to set a specific
scenario. For electricity, `residual_load` is recomputed automatically as
`demand − wind − solar` unless you pass it explicitly.

## Available features

The model input for each commodity is its **demand** plus the supporting drivers below (all
per zone, per hour, in MW / MWH2). The list is defined in
[`price_model/config.py`](price_model/config.py) — edit it there to add or drop features.

### Electricity — `price_eur_mwh = F(...)`

| Feature | Meaning |
|---------|---------|
| `demand` | **Primary input.** Total electricity demand, losses included [MW] |
| `wind` | Wind generation: onshore + offshore [MW] |
| `solar` | Solar generation: photovoltaic + thermal [MW] |
| `residual_load` | `demand − wind − solar` [MW] — *derived*; the dominant price driver (merit order) |
| `thermal` | Thermal generation: nuclear + lignite + hard coal + gas + oil + oil-shale [MW] |
| `hydro` | Hydro net: run-of-river + reservoir + pondage + pumped-storage (turbine − pump) [MW] |
| `battery` | Battery net: discharge − charge [MW] |
| `balance` | Net position / cross-border exchange balance [MW] |
| `dsr` | Demand-side response: explicit + implicit [MW] |

### Hydrogen — `h2_price = F(...)`

| Feature | Meaning |
|---------|---------|
| `h2_demand` | **Primary input.** Hydrogen demand, losses included [MWH2] |
| `electrolyser_gen` | Hydrogen produced by electrolysers [MWH2] |
| `smr` | Steam-methane-reformer output [MWH2] |
| `storage` | H2 storage net: discharge − charge [MWH2] |
| `balance` | H2 net position / cross-border exchange [MWH2] |
| `dumped` | Dumped (spilled) hydrogen [MWH2] |
| `hns` | Hydrogen not served [MWH2] |

The extracted parquet tables also carry `vre` (wind + solar), `ens` (energy not served) and
`dumped` for electricity; these are available for analysis but are not model inputs. The old
electrolyser-electricity-load input from the earlier design has been **removed** — total
electricity demand replaces it.

## Why not demand alone?

Gross demand by itself is a weak price predictor (electricity CV R² ≈ 0.04–0.30, hydrogen
≈ 0). Price is set by the *residual* system state, not raw demand:

* **Electricity** — the merit order is driven by `residual_load` = demand net of near-zero-cost
  wind & solar. Adding the weather drivers is what makes the model accurate.
* **Hydrogen** — price tracks the supply mix (electrolysers vs. steam-methane-reforming vs.
  storage), so those enter as features.

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
