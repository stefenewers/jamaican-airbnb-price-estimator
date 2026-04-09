"""
Application entry point.

Development:
    python run.py

Production (via Docker):
    gunicorn "run:app" --bind 0.0.0.0:5000 --workers 2
"""
from api.app import create_app
from config import API_HOST, API_PORT, DEBUG

app = create_app()

if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
