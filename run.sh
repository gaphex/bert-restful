#!/bin/bash
gunicorn --workers 2 --timeout 60 --bind 0.0.0.0:5000 wsgi:app
