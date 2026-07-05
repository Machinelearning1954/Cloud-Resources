# Iris Flower Classification — Production Application & Game

This project implements a machine learning model to classify Iris flower species and deploys it as a production-ready web application using Flask and Docker. It was built as the deployment capstone (Step 11: Deployment Implementation) for the Machine Learning Engineering & AI Bootcamp, and adheres to the provided cloud resource guidelines.

It also ships with **🌸 Iris Hunter**, a browser game at `/game`: each round a mystery flower is drawn to scale from a real sample in the Iris dataset, you guess the species, and the RandomForest model makes its own prediction. Ten rounds, streak bonuses — highest score wins. Can you beat the AI?

## Project Structure

```
.
├── app.py                        # Flask application (API + UI)
├── train_iris_model.py           # Script to train the Iris model
├── Dockerfile                    # Docker configuration
├── requirements.txt              # Python dependencies
├── model/
│   └── iris_classifier_rf.joblib # Trained RandomForest model
├── data/
│   ├── iris_features.csv         # Iris dataset features
│   └── iris_target.csv           # Iris dataset target
├── templates/
│   └── index.html                # HTML template for the UI
├── full_capstone_rubric.txt      # Capstone rubric (course material)
├── guidelines_analysis.txt       # Analysis of the cloud resource guidelines
├── guidelines_text.txt           # Extracted guidelines text
└── Projects for Accenture Machine Learning Engineer Role/   # Separate portfolio project
```

The application writes its runtime log to `build_log.txt` (created on first run; not tracked in git).

## Features

*   **Machine Learning Model:** RandomForestClassifier trained on the Iris dataset (100% accuracy on the held-out test set).
*   **Web API:**
    *   `POST /predict` — accepts feature data (JSON or form) and returns the predicted Iris species.
    *   `GET /health` — health check endpoint.
*   **User Interface:** A simple web page to input flower measurements and get predictions.
*   **Iris Hunter game (`/game`):** An arcade-style human-vs-model game served from the same app:
    *   `GET /game` — the game UI (SVG flower rendered to scale, keyboard controls, sound effects).
    *   `GET /game/round` — returns a random unlabeled sample from the real dataset.
    *   `POST /game/guess` — body `{"sample_id": <int>, "guess": <0|1|2>}`; returns the true species, whether the player was right, and what the model predicted. Guesses are validated server-side so the answer is never exposed to the browser before the guess.
*   **Logging:** Application events, predictions, and errors are logged to `build_log.txt` for monitoring and debugging.
*   **Containerization:** Dockerized with a `python:3.11-slim` base image and served by Gunicorn.
*   **Data Pipeline:** The training script loads the dataset, persists it to `data/`, and saves the trained model to `model/` where the app loads it.

## Prerequisites

*   Python 3.9+ (3.11 recommended)
*   pip
*   Docker (for containerized deployment)

## Running Locally (without Docker)

```bash
git clone https://github.com/Machinelearning1954/Cloud-Resources.git
cd Cloud-Resources

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional — the trained model is already included in model/.
# Run this only if you want to retrain from scratch:
python train_iris_model.py

python app.py
```

The application will be available at `http://localhost:5000`.

## Running with Docker (recommended)

```bash
docker build -t iris-app .
docker run -p 5000:5000 iris-app
```

The application will be available at `http://localhost:5000`. The container runs the app with Gunicorn bound to `0.0.0.0:5000`.

## API Endpoints

### Predict

*   **URL:** `/predict`
*   **Method:** `POST`
*   **JSON body:**
    ```json
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    ```
    (The same four fields are also accepted as HTML form data from the UI.)
*   **Success response:**
    ```json
    {
        "prediction": "setosa",
        "prediction_index": 0
    }
    ```
*   **Example:**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
      -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' \
      http://localhost:5000/predict
    ```

### Health Check

*   **URL:** `/health`
*   **Method:** `GET`
*   **Success response:** `{"status": "healthy", "model_loaded": true}`
*   **Example:** `curl http://localhost:5000/health`

## Logging

The application logs to `build_log.txt` (in the working directory, or `/app/build_log.txt` inside the container):

*   Application startup and model loading status
*   Incoming requests to endpoints
*   Prediction inputs and outputs
*   Errors encountered during processing

## Cloud Resource Guidelines Adherence

*   **Local prototyping first:** Both training and serving run on a local machine before any cloud resources are needed.
*   **Containerization:** Docker packaging makes the app portable to any container platform (Cloud Run, ECS, App Service, Kubernetes).
*   **Resource conservation:** Lightweight app on a slim base image; Gunicorn as a production WSGI server.
*   **Turn it off:** When deployed to a cloud provider, instances should be shut down when not in use. The app doesn't need to run 24/7 — it can be spun up on request.

## Further Development

*   **Cloud deployment:** Push the image to a container registry (Docker Hub, ECR, GCR, ACR) and deploy via a managed container service.
*   **Scalability:** Run multiple Gunicorn workers and container replicas behind a load balancer.
*   **Security:** Configure HTTPS and review Flask/cloud security best practices before public exposure.
*   **CI/CD:** Add a pipeline for automated testing, building, and deployment.
