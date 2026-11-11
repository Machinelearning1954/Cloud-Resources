"""Flask web app + REST API serving the Iris species classifier.

Endpoints:
    GET  /         HTML form UI
    POST /predict  JSON or form input -> predicted species
    GET  /health   liveness/readiness check (model loaded?)

Configuration (environment variables):
    IRIS_MODEL_PATH  path to the joblib model (default: ./model/iris_classifier_rf.joblib)
    IRIS_LOG_FILE    optional log file path (logs always go to stderr too)
    LOG_LEVEL        logging level (default: INFO)
    PORT             port for `python app.py` (default: 5000)
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get("IRIS_MODEL_PATH", BASE_DIR / "model" / "iris_classifier_rf.joblib")
)
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]
# Generous sanity bounds (cm); real Iris measurements are 0.1-7.9.
MIN_VALUE, MAX_VALUE = 0.0, 50.0
# Min/max of each feature in the training data (data/iris_features.csv). Inputs
# outside these are still answered, but flagged: the model has never seen them.
TRAINING_RANGE = {
    "sepal_length": (4.3, 7.9),
    "sepal_width": (2.0, 4.4),
    "petal_length": (1.0, 6.9),
    "petal_width": (0.1, 2.5),
}
MAX_BODY_BYTES = 16 * 1024  # a valid request is under 200 bytes


def _configure_logging() -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.environ.get("IRIS_LOG_FILE")
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("iris_app")


logger = _configure_logging()


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        logger.error("Model file not found at %s", path)
        return None
    try:
        loaded = joblib.load(path)
        logger.info("Model loaded successfully from %s", path)
        return loaded
    except Exception:  # noqa: BLE001 - log and keep serving /health
        logger.exception("Error loading model from %s", path)
        return None


class ValidationError(ValueError):
    """Raised for client input problems (-> HTTP 400)."""


def parse_features(payload) -> np.ndarray:
    """Validate a mapping of feature name -> value and return a (1, 4) array."""
    if not hasattr(payload, "get"):
        raise ValidationError("Request body must be a JSON object.")
    missing = [f for f in FEATURES if payload.get(f) in (None, "")]
    if missing:
        raise ValidationError(f"Missing required field(s): {', '.join(missing)}")
    values = []
    for name in FEATURES:
        raw = payload.get(name)
        if isinstance(raw, bool):
            raise ValidationError(f"Field '{name}' must be a number.")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            raise ValidationError(f"Field '{name}' must be a number.") from None
        if not math.isfinite(value) or not (MIN_VALUE < value <= MAX_VALUE):
            raise ValidationError(
                f"Field '{name}' must be > {MIN_VALUE} and <= {MAX_VALUE} (cm)."
            )
        values.append(value)
    return np.array(values, dtype=float).reshape(1, -1)


_UNSET = object()


def create_app(model=_UNSET) -> Flask:
    """Build the Flask app. Pass `model` to inject one (tests); None = no model."""
    app = Flask(__name__)
    app.config["MODEL"] = load_model() if model is _UNSET else model
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        return response

    @app.errorhandler(413)
    def too_large(_error):
        logger.warning("Rejected oversized request body")
        return jsonify({"error": f"Request body too large (limit {MAX_BODY_BYTES} bytes)."}), 413

    @app.errorhandler(500)
    def server_error(_error):
        # Never leak a stack trace or framework page to the client.
        return jsonify({"error": "Internal server error."}), 500

    def _predict(features: np.ndarray) -> dict:
        clf = app.config["MODEL"]
        idx = int(clf.predict(features)[0])
        result = {"prediction": TARGET_NAMES[idx], "prediction_index": idx}
        outside = [
            name for name, value in zip(FEATURES, features[0])
            if not TRAINING_RANGE[name][0] <= value <= TRAINING_RANGE[name][1]
        ]
        result["out_of_training_range"] = outside
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(features)[0]
            result["probabilities"] = {
                TARGET_NAMES[int(c)]: round(float(p), 4)
                for c, p in zip(clf.classes_, probs)
            }
        return result

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        start = time.perf_counter()
        is_json = request.is_json
        if app.config["MODEL"] is None:
            logger.error("Prediction failed: model not loaded")
            if is_json:
                return jsonify({"error": "Model not loaded."}), 503
            return render_template("index.html", prediction_text="Error: model not loaded."), 503

        payload = request.get_json(silent=True) if is_json else request.form
        try:
            features = parse_features(payload)
        except ValidationError as exc:
            logger.warning("Bad prediction request: %s", exc)
            if is_json:
                return jsonify({"error": str(exc)}), 400
            return render_template("index.html", prediction_text=f"Error: {exc}", **_form_echo()), 400

        result = _predict(features)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "prediction=%s features=%s out_of_range=%s latency_ms=%.2f",
            result["prediction"], features.tolist()[0], result["out_of_training_range"], latency_ms,
        )
        if is_json:
            return jsonify(result)
        return render_template(
            "index.html",
            prediction_text=f"Predicted Iris Species: {result['prediction']}" + (
                " (warning: outside the range of flowers the model was trained on: "
                + ", ".join(result["out_of_training_range"]) + ")"
                if result["out_of_training_range"] else ""
            ),
            **_form_echo(),
        )

    @app.get("/health")
    def health():
        loaded = app.config["MODEL"] is not None
        status = 200 if loaded else 503
        return jsonify({"status": "healthy" if loaded else "unhealthy", "model_loaded": loaded}), status

    return app


def _form_echo() -> dict:
    return {
        "sl": request.form.get("sepal_length"),
        "sw": request.form.get("sepal_width"),
        "pl": request.form.get("petal_length"),
        "pw": request.form.get("petal_width"),
    }


# Module-level app for gunicorn: `gunicorn app:app`
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
