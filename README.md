# Price Formation

Per-zone **demand → price** models for European electricity and hydrogen markets, trained
on the ENTSO-E TYNDP NT2030 / PLEXOS / CY2009 market-model output. Each bidding zone gets
its own gradient-boosted model, so the demand → price relationship (merit order for
electricity, supply mix for hydrogen) is learned separately per zone rather than forced
into one continent-wide curve.

| Commodity | Demand input | Price output | Zones modelled | Mean CV R² |
|-----------|--------------|---------------|----------------|------------|
| Electricity | `Demand [MW]` | `Marginal Cost [EUR/MWh]` | 63 | 0.94 |
| Hydrogen | `Demand [MWH2]` | `Marginal Cost [EUR/MWhH2]` | 25 | 0.96 |

Most zones land north of 0.95 R²; a handful of thin, poorly-interconnected markets
(MT00, LY00, EG00 for electricity; HR for hydrogen) sit lower because there's less signal
to learn from in the first place, not because the model is undertrained.

## Features

- Python API — `electricity_price(zone, demand, **ctx)` / `hydrogen_price(zone, h2_demand, **ctx)`
- Demand is the only required argument; every other driver (wind, solar, neighbouring-zone
  demand, correlated zones' prices, ...) defaults to that zone's historical median
- Interactive Django web app — commodity/zone picker, cross-validated accuracy, an
  hour-by-hour actual-vs-emulated historical price chart, and a "Future Analysis" panel
  where you drag a zone's typical 24-hour demand curve and see the price curve recompute
  live; `app.bat` launches it and opens your browser automatically
- Dockerfile for a self-contained, runnable version of the app

## Why per-zone, and why more than just demand

Demand alone is a weak predictor — electricity CV R² sits around 0.04–0.30 and hydrogen
close to 0 if that's the only feature. What actually drives price:

- **Electricity** is set by the merit order, so what matters is *residual load*
  (demand − wind − solar), not raw demand. A calm, sunny hour and a windy, cloudy hour at
  the same demand can have very different prices.
- **Hydrogen** is set by the supply mix — electrolyser output, SMR, storage — plus the
  electricity price feeding the electrolysers.
- Neighbouring zones matter too: each zone also gets its own neighbours' demand (raw and,
  for electricity, net of renewables), the system-wide total, and the prices of the 5
  zones it historically moves most closely with — whether or not they're directly wired
  together. Interconnection counts vary a lot (median ~3 neighbours, up to 17 for a hub
  like DE00), so the exact feature list is per zone, not one shared set.

Some of these features are dispatch *outcomes* rather than independent inputs — holding
them at their historical median (or letting them respond to demand automatically, in the
web app's Future Analysis panel) gives a realistic demand-sweep curve, but a strict
forward forecast would need to supply them directly.

## Project structure

```
price_model/
  config.py        # commodities: target, demand, feature list, output filenames
  extract.py        # workbook sheet -> tidy per-(zone, hour) feature table
  neighbors.py        # neighbour/system-total demand + correlated-zone price features
  multivariate.py       # per-zone model training, CV scoring, predict()
  api.py                # electricity_price(), hydrogen_price(), available_zones()

build_dataset.py    # raw workbook -> inputs/{elec,h2}_samples.parquet + adjacency JSONs
                     # (maintainer-only, not tracked in git -- the raw .xlsx isn't either)
train_model.py       # sample parquets -> outputs/*_model.joblib + *_metrics.csv

webui/               # Django web app
  manage.py
  priceui/           # project config (settings, urls)
  pricing/           # app
    services.py       # data loading/caching + prediction glue (mirrors old app.py logic)
    views.py            # index page + JSON API (zones / metrics / history / demand curve)
    templates/pricing/index.html
    static/pricing/     # css + vanilla JS (no build step) + vendored Chart.js

inputs/              # sample parquets + adjacency JSONs (committed), raw workbook (git-ignored)
outputs/             # trained models + metrics (git-ignored)
```

## Installation

### Windows (recommended)

Double-click `app.bat`, or run it from a shell:

```bash
app.bat
```

It creates a local `.venv`, installs `requirements.txt`, trains the models if
`outputs/*.joblib` aren't present yet (from the committed sample parquets — no need to run
`build_dataset.py`), and launches the Django app at `http://127.0.0.1:8000/`, also
reachable on your LAN/Tailscale IP. Safe to re-run: it skips any step that's already done.
Training both models from scratch takes ~45–50 minutes (electricity ~40 min, hydrogen ~7 min).

### Local (Python, manual)

Dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Docker

```bash
docker build -t price-model .
docker run -p 8000:8000 price-model
```

Then open `http://localhost:8000`.

## Usage

### Python API

```python
from price_model import electricity_price, hydrogen_price, available_zones

electricity_price("DE00", 55000)              # price at 55 GW demand, everything else at median
electricity_price("DE00", 55000, wind=20000)  # override a specific feature
electricity_price("DE00", [30000, 50000, 70000])  # vectorised over demand

hydrogen_price("DE", 1200)                    # price at 1200 MWH2 demand

available_zones("electricity")
available_zones("hydrogen")
```

### Django app

`app.bat` (see Installation above) handles this end-to-end — it also opens
`http://127.0.0.1:8000/` in your browser once the server is up. To run it manually
instead:

```bash
python webui/manage.py runserver
```

Pick a commodity (electricity/hydrogen) and a zone to see:

- **Trained Model Performance** — sample count, cross-validated R², cross-validated RMSE
- **Validation — Emulated vs. PLEXOS Price** — an hour-by-hour chart for a date range you
  pick, with an autoscale toggle for the Y axis
- **Future Analysis** — that zone's typical 24-hour demand shape for a chosen month. Drag
  any hour's point to a different demand value and the price chart below recomputes on
  release, with every other driver shifting along its real historical relationship to
  demand for that zone and month (rather than staying frozen at a flat average)

The frontend is plain HTML/CSS/JS (no npm/build step) calling a small JSON API —
`/api/zones/<commodity>/`, `/api/metrics/<commodity>/<zone>/`,
`/api/history/<commodity>/<zone>/`, `/api/monthly-demand-curve/<commodity>/<zone>/`,
`/api/monthly-price-curve/<commodity>/<zone>/` — and charts with a locally vendored
Chart.js, so the app works fully offline (no CDN dependency).

### Training pipeline

```bash
# Train both models from the committed sample parquets
python train_model.py

# Retrain a single commodity
python train_model.py --only hydrogen
```

Regenerating the sample parquets and adjacency JSONs from the raw workbook
(`build_dataset.py`) is a maintainer-only step, since the raw `.xlsx` isn't tracked in
the repo.
