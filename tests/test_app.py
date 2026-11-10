"""API, UI and model-quality tests. Run with: pytest -q"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as app_module  # noqa: E402
from train_iris_model import train  # noqa: E402

SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
VIRGINICA = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train a fresh model into a temp dir so tests never depend on a stale artifact."""
    out = tmp_path_factory.mktemp("artifacts")
    metrics = train(model_dir=out / "model", data_dir=out / "data")
    model = app_module.load_model(out / "model" / "iris_classifier_rf.joblib")
    return model, metrics


@pytest.fixture()
def client(trained):
    model, _ = trained
    return app_module.create_app(model=model).test_client()


def test_model_accuracy_floor(trained):
    _, metrics = trained
    # 45-sample holdout is noisy (one error = 2.2pp); CV is the stabler gate.
    assert metrics["test_accuracy"] >= 0.85
    assert metrics["cv_accuracy_mean"] >= 0.9


def test_committed_model_loads_and_predicts():
    model = app_module.load_model(app_module.MODEL_PATH)
    assert model is not None, "model/iris_classifier_rf.joblib missing - run train_iris_model.py"
    assert int(model.predict([[5.1, 3.5, 1.4, 0.2]])[0]) == 0


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "healthy", "model_loaded": True}


def test_health_without_model():
    c = app_module.create_app(model=None).test_client()
    r = c.get("/health")
    assert r.status_code == 503
    assert r.get_json()["model_loaded"] is False


def test_predict_without_model_returns_503():
    c = app_module.create_app(model=None).test_client()
    r = c.post("/predict", json=SETOSA)
    assert r.status_code == 503


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Iris Flower Species Prediction" in r.data


@pytest.mark.parametrize("payload,expected", [(SETOSA, "setosa"), (VIRGINICA, "virginica")])
def test_predict_json(client, payload, expected):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.get_json()
    assert body["prediction"] == expected
    assert set(body["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_predict_json_accepts_numeric_strings(client):
    r = client.post("/predict", json={k: str(v) for k, v in SETOSA.items()})
    assert r.status_code == 200
    assert r.get_json()["prediction"] == "setosa"


def test_predict_form(client):
    r = client.post("/predict", data={k: str(v) for k, v in SETOSA.items()})
    assert r.status_code == 200
    assert b"Predicted Iris Species: setosa" in r.data


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ({"sepal_length": 5.1}, "Missing required field"),
        ({**SETOSA, "petal_width": "abc"}, "must be a number"),
        ({**SETOSA, "petal_width": True}, "must be a number"),
        ({**SETOSA, "petal_width": -1}, "must be >"),
        ({**SETOSA, "petal_width": 1000}, "must be >"),
        ([1, 2, 3, 4], "JSON object"),
    ],
)
def test_predict_json_bad_input(client, payload, fragment):
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert fragment in r.get_json()["error"]


def test_predict_malformed_json(client):
    r = client.post("/predict", data="{not json", content_type="application/json")
    assert r.status_code == 400


def test_predict_nan_rejected(client):
    r = client.post("/predict", data=json.dumps({**SETOSA, "sepal_length": "NaN"}),
                    content_type="application/json")
    assert r.status_code == 400


def test_predict_form_bad_input(client):
    r = client.post("/predict", data={**{k: str(v) for k, v in SETOSA.items()}, "sepal_width": "x"})
    assert r.status_code == 400
    assert b"must be a number" in r.data


def test_predict_get_not_allowed(client):
    assert client.get("/predict").status_code == 405
