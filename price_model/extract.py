"""Extract per-zone, per-hour feature tables from the MMStandardOutputFile.

The ENTSO-E TYNDP workbook (PLEXOS, NT2030, CY2009) stores each hourly sheet in a wide
PLEXOS layout:

* row 10 (0-indexed) -> ``Category``  : metric / technology, e.g. ``"Nuclear [MW]"``,
  ``"Demand [MW]"``, ``"Marginal Cost [EUR]"``
* row 11             -> ``Country``   : the bidding-zone / core-point code (AL00, AT_H2, ...)
* row 12             -> unique code   : ``<zone>_<n>``  (ignored)
* rows 13..8748      -> hourly values : col 0 = Hour, col 1 = date code

We aggregate the many technology columns of each core point into a compact set of
features, one row per (zone, hour). Two sheets, two tables:

**electricity** (``Hourly Market Data``)::

    price_eur_mwh   Marginal Cost                        <- target
    demand          Demand [MW]                          <- primary input
    wind/solar      wind & solar generation              } exogenous weather drivers
    vre             wind + solar                         (derived)
    residual_load   demand - vre                         (derived; key price driver)
    thermal/hydro/battery/balance/dsr/ens/dumped         dispatch outcomes (context)

**hydrogen** (``Hourly H2 Data``)::

    h2_price          Marginal Cost [EUR/MWhH2]          <- target
    h2_demand         Demand [MWH2]                      <- primary input
    electrolyser_gen  Electrolyser (gen.)                } hydrogen supply mix
    smr               Steam methane reformer             }
    storage           H2 storage discharge - charge      }
    balance/dumped/hns                                   } net position / spill / unserved

A single streaming pass per sheet with openpyxl (read-only) keeps memory bounded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import openpyxl
except ImportError as e:  # pragma: no cover
    raise SystemExit("openpyxl is required: pip install openpyxl") from e

CAT_ROW, CTRY_ROW, FIRST_DATA_ROW = 10, 11, 13
MAXROWS = 9000  # upper bound on hourly rows (actual ~8736)

DEFAULT_XLSX = (
    Path(__file__).resolve().parent.parent
    / "inputs"
    / "MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx"
)

THERMAL_PREFIX = ("Nuclear", "Lignite", "Hard coal", "Hard Coal", "Gas ",
                  "Light oil", "Heavy oil", "Oil shale")


def _classify_elec(cat: str):
    c = cat.strip()
    if c.startswith("Marginal Cost"):             return ("price_eur_mwh", 1)
    if c.startswith("Demand [MW]"):               return ("demand", 1)
    if c.startswith("Demand Side Response"):      return ("dsr", 1)
    if c.startswith("Wind"):                      return ("wind", 1)
    if c.startswith("Solar"):                     return ("solar", 1)
    if c.startswith(("Run-of-River", "Reservoir", "Pondage")): return ("hydro", 1)
    if c.startswith("Pump Storage") and "turbine" in c: return ("hydro", 1)
    if c.startswith("Pump Storage") and "pump" in c:    return ("hydro", -1)
    if c.startswith("Battery Storage discharge"): return ("battery", 1)
    if c.startswith("Battery Storage charge"):    return ("battery", -1)
    if c.startswith("Balance"):                   return ("balance", 1)
    if c.startswith("Energy Not Served"):         return ("ens", 1)
    if c.startswith("Dumped Energy"):             return ("dumped", 1)
    if any(c.startswith(p) for p in THERMAL_PREFIX): return ("thermal", 1)
    return (None, 0)


def _classify_h2(cat: str):
    c = cat.strip()
    if c.startswith("Marginal Cost"):        return ("h2_price", 1)
    if c.startswith("Demand"):               return ("h2_demand", 1)
    if c.startswith("Electrolyser"):         return ("electrolyser_gen", 1)
    if c.startswith("Steam methane"):        return ("smr", 1)
    if c.startswith("H2 storage discharge"): return ("storage", 1)
    if c.startswith("H2 storage charge"):    return ("storage", -1)
    if c.startswith("Balance"):              return ("balance", 1)
    if c.startswith("Dumped"):               return ("dumped", 1)
    if c.startswith("Hydrogen Not Served"):  return ("hns", 1)
    return (None, 0)


ELEC_GROUPS = ["price_eur_mwh", "demand", "dsr", "wind", "solar", "hydro",
               "battery", "balance", "ens", "dumped", "thermal"]
H2_GROUPS = ["h2_price", "h2_demand", "electrolyser_gen", "smr", "storage",
             "balance", "dumped", "hns"]


def _stream(xlsx_path, sheet, classify, groups, strip_suffix=""):
    """Stream one sheet, summing each zone's columns into the named ``groups``."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet]
    cat = ctry = specs = None
    zones: list[str] = []
    accum: dict[str, dict[str, np.ndarray]] = {}
    nrows = 0
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == CAT_ROW:
            cat = list(row)
        elif i == CTRY_ROW:
            ctry = list(row)
        elif i == FIRST_DATA_ROW - 1:
            specs = []
            zset = set()
            for j in range(2, len(cat)):
                if cat[j] is None or ctry[j] is None:
                    continue
                g, sign = classify(str(cat[j]))
                if g is None:
                    continue
                z = str(ctry[j]).strip().replace(strip_suffix, "") if strip_suffix else str(ctry[j]).strip()
                specs.append((j, z, g, sign))
                zset.add(z)
            zones = sorted(zset)
            accum = {z: {g: np.zeros(MAXROWS) for g in groups} for z in zones}
        elif i >= FIRST_DATA_ROW:
            if row[0] is None:
                break
            r = nrows
            for j, z, g, sign in specs:
                v = row[j]
                if v is not None:
                    accum[z][g][r] += sign * v
            nrows += 1
    wb.close()
    return zones, nrows, accum


def _assemble(zones, nrows, accum, groups, year):
    dt = pd.date_range(f"{year}-01-01", periods=nrows, freq="h")
    frames = []
    for z in zones:
        d = {g: accum[z][g][:nrows] for g in groups}
        df = pd.DataFrame(d)
        df.insert(0, "zone", z)
        df.insert(1, "hour", np.arange(nrows, dtype="int32"))
        df.insert(2, "datetime", dt)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extract_electricity(xlsx_path: str | Path = DEFAULT_XLSX, year: int = 2009) -> pd.DataFrame:
    """Return the per-(zone, hour) electricity feature table."""
    zones, n, accum = _stream(xlsx_path, "Hourly Market Data", _classify_elec, ELEC_GROUPS)
    feat = _assemble(zones, n, accum, ELEC_GROUPS, year)
    feat["vre"] = feat["wind"] + feat["solar"]
    feat["residual_load"] = feat["demand"] - feat["vre"]
    return feat


def extract_hydrogen(xlsx_path: str | Path = DEFAULT_XLSX, year: int = 2009) -> pd.DataFrame:
    """Return the per-(zone, hour) hydrogen feature table (zone codes stripped of _H2)."""
    zones, n, accum = _stream(xlsx_path, "Hourly H2 Data", _classify_h2, H2_GROUPS,
                              strip_suffix="_H2")
    return _assemble(zones, n, accum, H2_GROUPS, year)


if __name__ == "__main__":  # smoke test
    e = extract_electricity()
    h = extract_hydrogen()
    print("electricity:", e.shape, e["zone"].nunique(), "zones")
    print("hydrogen:   ", h.shape, h["zone"].nunique(), "zones")
