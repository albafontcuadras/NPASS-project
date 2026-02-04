# URL configuration for Django

from django.urls import path
from .views import home, predict_view

urlpatterns = [
    path("", home, name="home"),
    path("api/predict", predict_view, name="predict"),
]
