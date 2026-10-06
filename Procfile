release: python manage.py migrate
web: gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 3 --timeout 120 cyber_physical_banking.wsgi:application
