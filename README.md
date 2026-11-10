# Cloud-Resources: Iris Classifier Capstone (ML Engineering Bootcamp)

A small, complete ML deployment: a RandomForest Iris species classifier served by a
Flask web UI and JSON REST API, run with gunicorn and packaged with Docker. It is built
to the bootcamp's capstone rubric (`docs/full_capstone_rubric.txt`,
`docs/step11_deployment_rubric.txt`) and cloud-resource guidelines
(`docs/guidelines_analysis.txt`): prototype locally first, keep the deployment simple,
and shut instances down when you're done.

## Repository layout

```
app.py                  Flask app: UI (/), API (/predict), health (/health)
train_iris_model.py     Data pipeline + training -> model/ and data/
gunicorn.conf.py        Production server settings (env-configurable)
Dockerfile              Container image (retrains at build so the pickle matches sklearn)
requirements.txt        Pinned runtime deps;  requirements-dev.txt: test deps
model/                  iris_classifier_rf.joblib + metrics.json
data/                   iris_features.csv, iris_target.csv (written by training)
templates/index.html    UI
tests/test_app.py       pytest suite (API, UI, validation, model quality)
.github/workflows/ci.yml  CI: tests + docker build
docs/                   Rubrics, cloud guidelines, original checklist
archive/original-uploads/  Original uploaded deliverables, kept unchanged
Projects for Accenture Machine Learning Engineer Role/  Separate portfolio project
```

## Quick start (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python train_iris_model.py          # optional: model/ is committed
pytest -q                           # 20 tests
gunicorn -c gunicorn.conf.py app:app   # or: python app.py
```

Then open http://localhost:5000.

## Docker

```bash
docker build -t iris-app .
docker run --rm -p 5000:5000 iris-app
```

The image runs as a non-root user and includes a `HEALTHCHECK` against `/health`.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | HTML form UI |
| POST | `/predict` | JSON or form body with `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (cm) |
| GET | `/health` | `200 {"status":"healthy","model_loaded":true}`, or `503` if no model is loaded |

```bash
curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"sepal_length":6.0,"sepal_width":2.7,"petal_length":4.5,"petal_width":1.4}'
# {"prediction":"versicolor","prediction_index":1,"probabilities":{"setosa":0.0,"versicolor":1.0,"virginica":0.0}}
```

Errors return `400` with `{"error": "..."}` for missing fields, non-numeric values,
non-finite values, values outside (0, 50] cm, or a body that isn't a JSON object. A
request without a loaded model returns `503`.

## Model

The pipeline is `load_iris` → CSVs in `data/` → a stratified 70/30 split (seed 42) →
`RandomForestClassifier(n_estimators=100)` → `model/`. From the committed
`model/metrics.json` (scikit-learn 1.9.1):

- **5-fold CV accuracy (train split):** 0.952 ± 0.030
- **Hold-out test accuracy (45 samples):** 0.889. Setosa is perfect; the errors are
  versicolor/virginica confusions.

The tests gate on CV ≥ 0.90 and hold-out ≥ 0.85. One error on the 45-sample hold-out
moves accuracy by 2.2 points, so the CV score is the more stable signal.

## Logging & monitoring

- App logs go to stderr. Set `IRIS_LOG_FILE=/path/app.log` to also write a file.
  Each prediction logs its class, input features and latency in ms. Bad requests log
  at WARNING.
- gunicorn access logs go to stdout.
- Use `/health` for load-balancer and container health checks.

Configuration env vars: `IRIS_MODEL_PATH`, `IRIS_LOG_FILE`, `LOG_LEVEL`, `PORT`,
`WEB_CONCURRENCY`, `GUNICORN_TIMEOUT`.

## Deploying to the cloud (cost-conscious)

Any container host works, for example Cloud Run, AWS App Runner, or a small
EC2/Lightsail instance running `docker run`. The model is tiny (CPU only), so the
smallest instance or a free tier is enough. Per the guidelines: **stop the service
when you're not demoing it.**

## What changed from the original upload

The original `app.py` (kept as `archive/original-uploads/app_original_broken.py`)
didn't parse: it had stray control characters in the logging config. It also
hard-coded `/home/ubuntu/...` paths, which the Docker image doesn't have. This
version fixes both and adds:

- input validation
- prediction probabilities
- an app factory
- tests
- a gunicorn config
- a non-root Docker image with a health check
- CI
