from django.urls import path

from . import views

app_name = "pricing"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/zones/<str:commodity>/", views.api_zones, name="api_zones"),
    path("api/metrics/<str:commodity>/<str:zone>/", views.api_metrics, name="api_metrics"),
    path("api/history/<str:commodity>/<str:zone>/", views.api_history, name="api_history"),
]
