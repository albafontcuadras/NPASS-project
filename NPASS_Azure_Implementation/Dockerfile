# Defines a Docker image that packages: FastAPI backend, Dash UI and ML model artifacts into a
# single container for Azure deployment

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_django.txt ./requirements_azure.txt
RUN pip install --no-cache-dir -r requirements_azure.txt

COPY npass_site ./npass_site

WORKDIR /app/npass_site
ENV DJANGO_SETTINGS_MODULE=npass_site.settings

RUN python manage.py migrate

EXPOSE 8000
CMD ["gunicorn", "-w", "3", "-b", "0.0.0.0:8000", "npass_site.wsgi:application"]
