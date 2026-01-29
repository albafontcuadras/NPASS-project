FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_phase1.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY npass_site ./npass_site

WORKDIR /app/npass_site
ENV DJANGO_SETTINGS_MODULE=npass_site.settings

RUN python manage.py migrate

EXPOSE 8000
CMD ["gunicorn", "-w", "3", "-b", "0.0.0.0:8000", "npass_site.wsgi:application"]
