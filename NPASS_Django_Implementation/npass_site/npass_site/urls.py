# URLs for the Django application

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("dash/", include("django_plotly_dash.urls")),  # mounts Dash routes
    path("", include("npass.urls")),                    # app routes (home + API)
]
