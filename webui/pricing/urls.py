from django.urls import path

from . import views

app_name = "pricing"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/zones/<str:commodity>/", views.api_zones, name="api_zones"),
    path("api/metrics/<str:commodity>/<str:zone>/", views.api_metrics, name="api_metrics"),
    path("api/history/<str:commodity>/<str:zone>/", views.api_history, name="api_history"),
    path("api/monthly-demand-curve/<str:commodity>/<str:zone>/", views.api_monthly_demand_curve, name="api_monthly_demand_curve"),
    path("api/monthly-price-curve/<str:commodity>/<str:zone>/", views.api_monthly_price_curve, name="api_monthly_price_curve"),
]
