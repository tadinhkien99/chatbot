#!/bin/bash

if [ "$SERVICE" = "app" ]; then
    exec uvicorn app:app --host 0.0.0.0 --port 8009 --workers 1
elif [ "$SERVICE" = "worker" ]; then
    exec celery -A tasks worker --loglevel=info
else
    exec "$@"
fi
