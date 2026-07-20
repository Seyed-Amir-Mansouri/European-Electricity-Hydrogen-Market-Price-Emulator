"""Configuration for the two commodity price models.

Each commodity maps a **demand** input to a **price** output, learned per bidding zone
from the ENTSO-E TYNDP NT2030 / PLEXOS / CY2009 workbook:

* electricity -- ``Hourly Market Data`` sheet, price = Marginal Cost [EUR/MWh]
* hydrogen    -- ``Hourly H2 Data``     sheet, price = Marginal Cost [EUR/MWhH2]

``demand`` is the primary input; the remaining ``features`` are supporting context (they
default to each zone's median when a caller supplies only demand). ``target`` is the
price column the model predicts.
"""
from __future__ import annotations

COMMODITIES = {
    "electricity": {
        "unit": "EUR/MWh",
        "demand": "demand",
        "target": "price_eur_mwh",
        # demand (primary) + exogenous weather drivers + dispatch outcomes.
        # NB: electrolyser load is intentionally NOT a feature here.
        "features": ["demand", "wind", "solar", "residual_load",
                     "thermal", "hydro", "battery", "balance", "dsr"],
        "samples": "elec_samples.parquet",
        "model": "electricity_model.joblib",
        "metrics": "electricity_metrics.csv",
    },
    "hydrogen": {
        "unit": "EUR/MWhH2",
        "demand": "h2_demand",
        "target": "h2_price",
        # H2 demand (primary) + the hydrogen supply mix.
        "features": ["h2_demand", "electrolyser_gen", "smr",
                     "storage", "balance", "dumped", "hns"],
        "samples": "h2_samples.parquet",
        "model": "hydrogen_model.joblib",
        "metrics": "hydrogen_metrics.csv",
    },
}
