# Start celery
celery -A mlplatform worker -s /tmp/tmp.db --pool=solo -l info

# Start flower
celery -A mlplatform flower --port=5555

# Start server. Add --noreload if got error Script manage.py doesn't exist
python manage.py runserver --noreload

# Start docker
# docker start redis
# docker start my-postgre