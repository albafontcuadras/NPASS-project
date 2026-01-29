from django.apps import AppConfig

class NpassConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "npass"

    def ready(self):
        # Ensure the Dash app is imported so it's registered
        from . import dash_app  # noqa: F401
