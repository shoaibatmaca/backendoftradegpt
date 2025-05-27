# web: gunicorn core.wsgi --log-file - 
# #or works good with external database
# web: python manage.py migrate && gunicorn core.wsgi
# web: python manage.py migrate && gunicorn core.wsgi:application --log-file -
web: python manage.py migrate && gunicorn core.wsgi:application --timeout 60 --log-file -


