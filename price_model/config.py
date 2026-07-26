"""Configuration for the two commodity price models.

Each commodity maps a **demand** input to a **price** output, learned per bidding zone
from the ENTSO-E TYNDP NT2030 / PLEXOS / CY2009 workbook:

* electricity -- ``Hourly Market Data`` sheet, price = Marginal Cost [EUR/MWh]
* hydrogen    -- ``Hourly H2 Data``     sheet, price = Marginal Cost [EUR/MWhH2]

``demand`` is the primary input; the remaining ``features`` are supporting context (they
default to each zone's median when a caller supplies only demand). ``target`` is the
price column the model predicts. ``adjacency`` names the JSON file (built by
``build_dataset.py`` from the workbook's crossborder-exchange sheet) mapping each zone to
its directly interconnected neighbours -- used to add per-zone ``neighbor_net_demand_<N>``
/ ``neighbor_net_demand_total`` / ``net_demand_system_total`` features (see
``price_model/neighbors.py``). Because neighbour counts vary a lot per zone (median ~3,
up to 17 for a hub like DE00), each zone ends up with its own feature list, stored per
zone in the trained bundle rather than as one shared list.

``net_demand_col``, if set, both (a) is added to the zone's own base ``features`` as its
own net demand, and (b) is the column used for the neighbour/system-total *net* features
-- built *alongside* (not instead of) the raw-demand versions of each, since both are
wanted. "Net demand" means demand net of renewables (electricity's existing
``residual_load = demand - wind - solar``), not raw demand. Hydrogen has no wind/solar
equivalent tied to H2 zones, so it's left unset there -- only raw-demand neighbour/
system-total features exist for hydrogen, and there's no separate "own net demand" for
it either (h2_demand is the only demand quantity available).
"""
from __future__ import annotations

COMMODITIES = {
    "electricity": {
        "unit": "EUR/MWh",
        "demand": "demand",
        "target": "price_eur_mwh",
        # demand + net demand (residual_load = demand - wind - solar) for the zone
        # itself, + weather drivers + calendar/time context, + ens/dumped (energy not
        # served / curtailed) -- these two are dispatch outcomes like thermal/hydro/
        # balance below, but they're also the *cause* of scarcity/oversupply pricing
        # in thin markets (e.g. GE00, MT00 CV R^2 rose from ~0.3-0.4 to >0.9 with
        # them added), so they're kept in despite the general exclusion. Other
        # dispatch-outcome features (thermal, hydro, battery, dsr) still excluded.
        # Neighbour/system-total demand features (both raw and net) are added per
        # zone on top of this (see `adjacency`/`net_demand_col`).
        "features": ["demand", "residual_load", "wind", "solar", "month", "season", "hour",
                     "ens", "dumped"],
        "samples": "elec_samples.parquet",
        "adjacency": "elec_adjacency.json",
        "net_demand_col": "residual_load",  # demand - wind - solar, already in the parquet
        "max_price": 500,  # hours where this zone's own price exceeds this are dropped
                            # entirely, from both training and CV scoring
        "model": "electricity_model.joblib",
        "metrics": "electricity_metrics.csv",
    },
    "hydrogen": {
        "unit": "EUR/MWhH2",
        "demand": "h2_demand",
        "target": "h2_price",
        # H2 demand (primary) + electrolysis feedstock price + calendar/time context,
        # + dumped/hns (curtailed H2 / hydrogen not served) -- the H2 analogue of
        # electricity's ens/dumped above, kept in for the same reason (scarcity/
        # oversupply pricing signal in thin markets). The rest of the hydrogen
        # supply-mix / trade features (electrolyser_gen, smr, storage, balance,
        # h2_net_trade) still intentionally excluded. Neighbour/system-total demand
        # features are added per zone on top of this (see `adjacency` above).
        "features": ["h2_demand", "elec_price", "month", "season", "hour", "dumped", "hns"],
        "samples": "h2_samples.parquet",
        "adjacency": "h2_adjacency.json",
        "net_demand_col": None,  # no renewables column for H2 zones -- falls back to h2_demand
        "model": "hydrogen_model.joblib",
        "metrics": "hydrogen_metrics.csv",
    },
}
