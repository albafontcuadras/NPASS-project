#!/usr/bin/env bash
set -o errexit

pip install -r requirements_phase1.txt

cd npass_site
python manage.py collectstatic --noinput
python manage.py migrate
