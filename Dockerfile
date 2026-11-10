# Iris classifier - production image (Flask app served by gunicorn)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train_iris_model.py gunicorn.conf.py ./
COPY templates ./templates
COPY data ./data
# Retrain inside the image so the pickled model always matches the
# installed scikit-learn version (training takes ~1s).
RUN python train_iris_model.py

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"5000\")}/health')" || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
