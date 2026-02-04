# Configurating the Django application to import Dash so that it can use it later

from django.apps import AppConfig

class NpassConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "npass"

    def ready(self):
        from . import dash_app 
