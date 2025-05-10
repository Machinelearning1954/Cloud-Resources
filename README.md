# Iris Flower Classification - Production Application

This project implements a machine learning model to classify Iris flower species and deploys it as a production-ready web application using Flask and Docker. It adheres to the guidelines provided for cloud resource usage and capstone project deployment.

## Project Structure

```
iris_project/
├── app.py                  # Flask application
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── model/
│   └── iris_classifier_rf.joblib # Trained RandomForest model
├── data/
│   ├── iris_features.csv   # Iris dataset features
│   └── iris_target.csv     # Iris dataset target
├── templates/
│   └── index.html          # HTML template for the UI
├── train_iris_model.py     # Script to train the Iris model
└── build_log.txt           # Log file for application events (will be created on run)
```

## Features

*   **Machine Learning Model:** Uses a RandomForestClassifier trained on the Iris dataset.
*   **Web API:** Provides a RESTful API for predictions.
    *   `POST /predict`: Accepts feature data (JSON or form) and returns the predicted Iris species.
    *   `GET /health`: Health check endpoint.
*   **User Interface:** A simple web page to input flower measurements and get predictions.
*   **Logging:** Implements logging for monitoring application events, performance, and debugging.
*   **Containerization:** Dockerized for easy deployment and scalability.
*   **Data Pipeline:** Basic data loading and model persistence are handled.

## Prerequisites

*   Python 3.9+ (Python 3.11 used for development)
*   pip (Python package installer)
*   Docker (for containerized deployment)
*   Git (for cloning the repository)

## Setup and Running the Application

There are two main ways to run this application: locally using Python and Flask directly, or using Docker.

### 1. Running Locally (without Docker)

**a. Clone the Repository (if applicable)**

```bash
git clone <repository-url>
cd iris_project
```

**b. Create a Virtual Environment (Recommended)**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**c. Install Dependencies**

```bash
pip install -r requirements.txt
```

**d. Train the Model (if not already present or to retrain)**

The `model/iris_classifier_rf.joblib` and `data/` files are included. If you need to retrain:

```bash
python train_iris_model.py
```

This will save the trained model to `model/iris_classifier_rf.joblib` and the dataset to the `data/` directory.

**e. Run the Flask Application**

```bash
python app.py
```

The application will typically be available at `http://127.0.0.1:5000` or `http://0.0.0.0:5000`.

### 2. Running with Docker (Recommended for Production-like Environment)

**a. Build the Docker Image**

Navigate to the `iris_project` directory (where the `Dockerfile` is located) and run:

```bash
docker build -t iris-app .
```

This command builds a Docker image tagged as `iris-app` using the instructions in the `Dockerfile`.

**b. Run the Docker Container**

Once the image is built, run a container from it:

```bash
docker run -p 5000:5000 iris-app
```

*   `-p 5000:5000`: This maps port 5000 of the host machine to port 5000 of the Docker container (where the Flask app is running via Gunicorn).

The application will be accessible at `http://localhost:5000` in your web browser.

## API Endpoints

### Predict

*   **URL:** `/predict`
*   **Method:** `POST`
*   **Data format (Form Data):**
    *   `sepal_length`: float
    *   `sepal_width`: float
    *   `petal_length`: float
    *   `petal_width`: float
*   **Data format (JSON):**
    ```json
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    ```
*   **Success Response (JSON):**
    ```json
    {
        "prediction": "setosa",
        "prediction_index": 0
    }
    ```
*   **Example cURL (JSON):**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
    -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' \
    http://localhost:5000/predict
    ```

### Health Check

*   **URL:** `/health`
*   **Method:** `GET`
*   **Success Response (JSON):**
    ```json
    {
        "status": "healthy",
        "model_loaded": true
    }
    ```
*   **Example cURL:**
    ```bash
    curl http://localhost:5000/health
    ```

## Logging

The application logs events to `build_log.txt` (relative to where `app.py` is run, or inside the container at `/app/build_log.txt`). This includes:
*   Application startup and model loading status.
*   Incoming requests to endpoints.
*   Prediction inputs and outputs.
*   Errors encountered during processing.

## Cloud Resource Guidelines Adherence

*   **Local Prototyping:** The model training script (`train_iris_model.py`) and Flask app (`app.py`) can be run locally first.
*   **Containerization:** Docker is used for packaging, aligning with recommendations for MLaaS platforms or custom containerization.
*   **Resource Conservation:** The application is lightweight. The Docker image is based on `python:3.11-slim` to keep its size minimal. Gunicorn is used as a production-ready WSGI server.
*   **Shutting Down Instances:** When deployed to a cloud provider, instances should be managed according to the guidelines (e.g., shut down when not in use, use appropriate instance sizes).

## Further Development & Deployment Notes

*   **Cloud Deployment:** To deploy this to a cloud provider (AWS, GCP, Azure), you would typically push the Docker image to a container registry (e.g., Docker Hub, ECR, GCR, ACR) and then deploy it using a container orchestration service (e.g., Kubernetes, ECS, Cloud Run, App Service).
*   **Scalability:** Gunicorn allows running multiple worker processes. For higher scalability, consider using a load balancer and multiple container instances.
*   **Security:** For a true production environment, ensure HTTPS is configured, and review security best practices for Flask and your chosen cloud platform.
*   **CI/CD:** Implement a CI/CD pipeline for automated testing, building, and deployment.
*   **Advanced Data Pipelines:** For more complex projects, a dedicated data pipeline tool (e.g., Apache Airflow, Kubeflow Pipelines) would be used.

This README provides the necessary instructions to run and understand the application, fulfilling the documentation requirements of the capstone project.

