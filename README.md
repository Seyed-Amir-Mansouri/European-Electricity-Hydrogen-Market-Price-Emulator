# Price Formation

Per-zone **demand → price** models for European electricity and hydrogen markets, trained
on the ENTSO-E TYNDP NT2030 / PLEXOS / CY2009 market-model output. A separate
gradient-boosted model is fit per bidding zone, so each zone gets its own demand → price
curve.

| Commodity | Demand input | Price output | Zones modelled |
|-----------|--------------|---------------|----------------|
| Electricity | `Demand [MW]` | `Marginal Cost [EUR/MWh]` | 63 |
| Hydrogen | `Demand [MWH2]` | `Marginal Cost [EUR/MWhH2]` | 25 |

## Features

- Python API — `electricity_price(zone, demand, **ctx)` / `hydrogen_price(zone, h2_demand, **ctx)`
- Demand is the only required argument; every other feature defaults to that zone's median
- Interactive Streamlit app to explore accuracy, feature importances, and the demand → price curve per zone
- Dockerfile for a self-contained, runnable version of the app

## Project structure

```
price_model/
  config.py        # commodities: target, demand, feature list, output filenames
  extract.py        # workbook sheet -> tidy per-(zone, hour) feature table
  multivariate.py    # per-zone model training, CV scoring, predict()
  api.py             # electricity_price(), hydrogen_price(), available_zones()
  neighbors.py        # neighbour/system-total demand features

build_dataset.py    # raw workbook -> outputs/{elec,h2}_samples.parquet (maintainer-only)
train_model.py       # sample parquets -> outputs/*_model.joblib + *_metrics.csv
app.py               # Streamlit app

inputs/              # sample parquets + adjacency JSONs (committed), raw workbook (git-ignored)
outputs/             # trained models + metrics (git-ignored)
```

## Installation

### Local (Python)

Dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Docker

```bash
docker build -t price-model .
docker run -p 8501:8501 price-model
```

Then open `http://localhost:8501`.

## Usage

### Python API

```python
from price_model import electricity_price, hydrogen_price, available_zones

electricity_price("DE00", 55000)              # price at 55 GW demand, median weather
electricity_price("DE00", 55000, wind=20000)  # override a specific feature
electricity_price("DE00", [30000, 50000, 70000])  # vectorised over demand

hydrogen_price("DE", 1200)                    # price at 1200 MWH2 demand

available_zones("electricity")
available_zones("hydrogen")
```

### Streamlit app

```bash
streamlit run app.py
```

Pick a commodity and zone to see cross-validated accuracy, feature importances,
predicted-vs-actual, and the demand → price curve.

### Training pipeline

```bash
# Train both models from the committed sample parquets
python train_model.py

# Retrain a single commodity
python train_model.py --only hydrogen
```

Regenerating the sample parquets from the raw workbook (`build_dataset.py`) is a
maintainer-only step, since the raw `.xlsx` is not tracked in the repo.
