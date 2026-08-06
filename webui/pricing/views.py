from __future__ import annotations

from django.http import JsonResponse
from django.shortcuts import render

from price_model.config import COMMODITIES

from . import services


def index(request):
    return render(request, "pricing/index.html", {
        "commodities": list(COMMODITIES),
    })


def api_zones(request, commodity):
    if commodity not in COMMODITIES:
        return JsonResponse({"error": f"Unknown commodity {commodity!r}."}, status=404)
    if not services.model_available(commodity):
        return JsonResponse(
            {"error": f"{COMMODITIES[commodity]['model']} not found -- "
                      f"run build_dataset.py then train_model.py."},
            status=404,
        )
    zones = services.zone_choices(commodity)
    return JsonResponse({
        "zones": [{"code": z, "label": services.zone_label(z)} for z in zones],
        "default": services.default_zone(commodity),
    })


def api_metrics(request, commodity, zone):
    if commodity not in COMMODITIES:
        return JsonResponse({"error": f"Unknown commodity {commodity!r}."}, status=404)
    bundle = services.load_bundle(commodity)
    if zone not in bundle["zones"]:
        return JsonResponse({"error": f"Unknown zone {zone!r} for {commodity}."}, status=404)
    return JsonResponse({
        "zone_label": services.zone_label(zone),
        "commodity_label": commodity.title(),
        "date_bounds": services.date_bounds(commodity),
        **services.zone_metrics(commodity, zone),
    })


def api_history(request, commodity, zone):
    if commodity not in COMMODITIES:
        return JsonResponse({"error": f"Unknown commodity {commodity!r}."}, status=404)
    bundle = services.load_bundle(commodity)
    if zone not in bundle["zones"]:
        return JsonResponse({"error": f"Unknown zone {zone!r} for {commodity}."}, status=404)

    start = request.GET.get("start") or None
    end = request.GET.get("end") or None
    try:
        data = services.history_series(commodity, zone, start, end)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse(data)
