# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

This is a machine learning **portfolio/coursework repository** (Machine Learning Engineering & AI Bootcamp capstone work plus job-application portfolio projects), not a single runnable application. Files were added via GitHub web uploads, so the repo is a collection of flattened deliverables, docs, and archives rather than a conventional project layout. There is no repo-level `requirements.txt`, no test suite checked in, and no active CI (the `ci-cd-pipeline.yml` file is documentation — it lives inside a project folder, not under `.github/workflows/`).

It contains three independent projects:

### 1. Iris Flower Classification (repository root)

A Flask + Docker capstone deliverable that serves a RandomForest Iris classifier.

- `app.py` — Flask app with `POST /predict` (JSON or form data), `GET /health`, and a home page. **Known issues in the root copy**: line 10 has a corrupted logging `filename=` string (syntax error — the file will not run as-is), and `MODEL_PATH` is hardcoded to `/home/ubuntu/iris_project/model`.
- `Dockerfile` — builds from `python:3.11-slim`, runs via Gunicorn on port 5000. It COPYs `requirements.txt`, `model/`, `data/`, and `templates/`, **none of which exist at the repo root** — they are inside `iris_project_deliverable.zip`.
- `iris_project_deliverable.zip` — the **complete, canonical version** of this project (`home/ubuntu/iris_project/` with a clean `app.py`, trained model `iris_classifier_rf.joblib`, data CSVs, `templates/index.html`, `requirements.txt`). Extract this zip if you need the full working project rather than patching the root copies.
- `README (1).md` — the Iris project's full README (setup, API reference, Docker usage). The root `README.md` is just a title.
- `guidelines_text.txt`, `guidelines_analysis.txt`, `full_capstone_rubric.txt` — bootcamp cloud-resource guidelines and capstone rubric this project was built against (local prototyping first, small instances, shut down unused cloud instances, `debug=False` in production).

Run it (after extracting the zip so all pieces exist together):

```bash
pip install -r requirements.txt
python app.py                    # Flask dev server on 0.0.0.0:5000
# or containerized:
docker build -t iris-app .
docker run -p 5000:5000 iris-app # Gunicorn
```

### 2. Federal ML Portfolio (`Projects for Accenture Machine Learning Engineer Role/`)

Portfolio projects targeting federal/defense ML work. Main project: **Army Vehicle Predictive Maintenance** — XGBoost failure prediction 14 days ahead from synthetic vehicle sensor telemetry, with SHAP explainability and a FastAPI serving layer.

- `data_generator.py` — `VehicleSensorDataGenerator`, synthetic sensor telemetry (CLI: `--num-vehicles`, `--days`, seeded for reproducibility)
- `xgboost_classifier.py` — `VehicleFailurePredictor`, training/evaluation CLI (`--train`, `--predict --input ... --output ...`)
- `explainer.py` — `ModelExplainer`, SHAP explanations for predictions
- `app.py` — FastAPI app (Pydantic schemas with field validation, HTTPBearer auth stub, Swagger at `/docs`)
- `lightgbm_classifier.py` — a **separate** project: hospital 30-day readmission prediction with LightGBM + SMOTE
- `Dockerfile` — multi-stage build, non-root `appuser`, port 8000
- `README.md`, `federal-ml-guide.md`, `federal-compliance.md`, `deployment-guide.md`, `ci-cd-pipeline.yml`, `CONTRIBUTING.md` — extensive documentation, including NIST SP 800-171 compliance framing
- `federal-ml-portfolio.tar.gz` — archived full portfolio

**Layout mismatch to be aware of**: the README, CONTRIBUTING, and CI YAML describe an `src/data/`, `src/models/`, `src/api/`, `tests/` package layout under `projects/army-vehicle-predictive-maintenance/`, but the actual checked-in files sit flat in this one folder. The described `tests/`, `notebooks/`, and `k8s/` directories do not exist in the repo. Don't assume documented paths exist — verify first.

Commands (per the project's own docs; deps in its `requirements.txt`):

```bash
cd "Projects for Accenture Machine Learning Engineer Role"
pip install -r requirements.txt
python data_generator.py --num-vehicles 1000 --days 365
python xgboost_classifier.py --train
uvicorn app:app --host 0.0.0.0 --port 8000     # serve the API
```

Code-quality tooling the portfolio standardizes on (mirrors `ci-cd-pipeline.yml`):

```bash
black .                                  # formatting
flake8 . --max-line-length=100 --ignore=E203,W503
mypy . --ignore-missing-imports
bandit -r . -ll                          # security scan
pytest tests/ -v --cov                   # tests (no tests are currently checked in)
```

### 3. GTA 6 Hype Survey Analysis (`gta6-hype-survey/`)

A small, self-contained tabular ML project on the Kaggle "GTA 6 Fan Expectations & Hype Survey 2026" dataset (synthetic survey data). Follows the same module pattern as the federal portfolio.

- `gta6_hype_analysis.py` — `HypeSurveyModel`: preorder-intent classification (RandomForest in a leak-free `ColumnTransformer` pipeline, benchmarked against a 5-fold CV LogisticRegression AUC) plus fan-persona `KMeans` segmentation. CLI: `--train`, `--data <csv>`, `--output <dir>`, `--personas <k>`.
- **Runs without the dataset**: if `--data` is omitted (or the file is absent), it generates a clearly-labeled *synthetic stand-in* matching the Kaggle schema so the pipeline executes end to end. This is a demo scaffold, not a substitute for the real CSV in a submission. The Kaggle dataset is itself synthetic.
- `artifacts/` is gitignored (trained `.joblib` models are written there at runtime); `.gitkeep` preserves the directory.

```bash
cd gta6-hype-survey
pip install -r requirements.txt
python gta6_hype_analysis.py --train --data gta6_hype_survey.csv --output artifacts/
python gta6_hype_analysis.py --train     # synthetic fallback, no CSV needed
```

## Conventions

- Python 3.11, `python:3.11-slim` base images, Gunicorn (Flask project) / Uvicorn (FastAPI project) for serving.
- Module pattern in the portfolio code: docstring header, `logging` configured at module level, a single main class, `argparse` CLI entry point, models persisted with `joblib`.
- The zip/tar.gz archives are uploaded deliverables and the most complete versions of their projects — treat them as reference sources of truth, and don't delete or regenerate them casually.
- Filenames contain spaces (`Projects for Accenture Machine Learning Engineer Role/`, `README (1).md`) — quote paths in shell commands.
- Documentation here doubles as portfolio writing (metrics, architecture diagrams, compliance sections describe the *intended* production system, not what's checked in). When editing code, verify claims against the actual files rather than the docs.

## Media & Video References

YouTube references cited during development (e.g. graphics/visual-fidelity direction for
the interactive demos). The embed code for each is recorded here so it is available even
though this environment's network policy blocks outbound requests to `www.youtube.com`.

### aV3sPhzuQbQ — graphics / visual-fidelity reference

- Watch: https://youtu.be/aV3sPhzuQbQ
- Used as a "boost the graphics" reference for the top-down open-world demo (cinematic
  post-processing pass: time-of-day color grading, sun/moon glint, vignette, film grain,
  depth-graded water with sun sparkle).

Standard responsive iframe embed:

```html
<iframe
  width="560" height="315"
  src="https://www.youtube.com/embed/aV3sPhzuQbQ"
  title="YouTube video player"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  referrerpolicy="strict-origin-when-cross-origin"
  allowfullscreen>
</iframe>
```

Privacy-enhanced (`youtube-nocookie`) variant:

```html
<iframe
  width="560" height="315"
  src="https://www.youtube-nocookie.com/embed/aV3sPhzuQbQ"
  title="YouTube video player"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  allowfullscreen>
</iframe>
```

Markdown thumbnail link (renders on GitHub, which strips raw iframes):

```markdown
[![Watch on YouTube](https://img.youtube.com/vi/aV3sPhzuQbQ/hqdefault.jpg)](https://youtu.be/aV3sPhzuQbQ)
```

To add another reference, append a new `### <video-id>` block in the same format.
