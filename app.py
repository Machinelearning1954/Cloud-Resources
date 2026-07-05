from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import csv
import random
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename="build_log.txt", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")

# Define paths relative to this file so the app works both locally and inside the container.
# MODEL_DIR can be overridden with an environment variable if needed.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "model"))
MODEL_PATH = os.path.join(MODEL_DIR, "iris_classifier_rf.joblib")

# Load the trained model
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        app.logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as e:
        app.logger.error("Error loading model: %s", str(e))
        model = None
else:
    app.logger.error("Model file not found at %s", MODEL_PATH)
    model = None

# Define target names for Iris dataset (for user-friendly output)
iris_target_names = ["setosa", "versicolor", "virginica"]

# Load the dataset for the game: each entry is (features, target_index).
# The game serves random real samples and checks guesses server-side.
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
game_samples = []
try:
    with open(os.path.join(DATA_DIR, "iris_features.csv")) as ff, \
         open(os.path.join(DATA_DIR, "iris_target.csv")) as tf:
        feature_rows = list(csv.reader(ff))[1:]  # skip header
        target_rows = list(csv.reader(tf))[1:]
        for feats, target in zip(feature_rows, target_rows):
            game_samples.append(([float(v) for v in feats], int(target[0])))
    app.logger.info("Game dataset loaded: %d samples", len(game_samples))
except Exception as e:
    app.logger.error("Could not load game dataset: %s", str(e))

@app.route("/")
def home():
    app.logger.info("Home page accessed.")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        app.logger.error("Prediction attempt failed: Model not loaded.")
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    try:
        if request.is_json:
            data = request.get_json()
            app.logger.info("Received JSON data for prediction: %s", data)
            features = np.array([data["sepal_length"], data["sepal_width"], 
                                 data["petal_length"], data["petal_width"]]).reshape(1, -1)
        else:
            # Form data
            app.logger.info("Received form data for prediction: %s", request.form)
            features = np.array([float(request.form["sepal_length"]),
                                 float(request.form["sepal_width"]),
                                 float(request.form["petal_length"]),
                                 float(request.form["petal_width"])]).reshape(1, -1)
        
        app.logger.info("Features for prediction: %s", features)
        prediction_index = model.predict(features)
        predicted_class_name = iris_target_names[prediction_index[0]]
        app.logger.info("Prediction successful: %s (index: %s)", predicted_class_name, prediction_index[0])
        
        if request.is_json:
            return jsonify({"prediction": predicted_class_name, "prediction_index": int(prediction_index[0])})
        else:
            return render_template("index.html", 
                                   prediction_text=f"Predicted Iris Species: {predicted_class_name}",
                                   sl=request.form["sepal_length"], 
                                   sw=request.form["sepal_width"], 
                                   pl=request.form["petal_length"], 
                                   pw=request.form["petal_width"])

    except Exception as e:
        app.logger.error("Error during prediction: %s", str(e))
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        else:
            return render_template("index.html", prediction_text=f"Error: {str(e)}"), 400

@app.route("/game")
def game():
    app.logger.info("Game page accessed.")
    return render_template("game.html")

@app.route("/game/round")
def game_round():
    """Serve a random real sample from the dataset (without its label)."""
    if not game_samples:
        app.logger.error("Game round requested but dataset is not loaded.")
        return jsonify({"error": "Game dataset not loaded."}), 500
    sample_id = random.randrange(len(game_samples))
    features = game_samples[sample_id][0]
    app.logger.info("Game round served: sample %d", sample_id)
    return jsonify({
        "sample_id": sample_id,
        "features": {
            "sepal_length": features[0],
            "sepal_width": features[1],
            "petal_length": features[2],
            "petal_width": features[3],
        },
    })

@app.route("/game/guess", methods=["POST"])
def game_guess():
    """Score the player's guess against the truth and the model's prediction."""
    if not game_samples:
        return jsonify({"error": "Game dataset not loaded."}), 500
    try:
        data = request.get_json()
        sample_id = int(data["sample_id"])
        guess = int(data["guess"])
        if not (0 <= sample_id < len(game_samples)) or not (0 <= guess <= 2):
            raise ValueError("sample_id or guess out of range")
    except Exception as e:
        app.logger.error("Invalid game guess: %s", str(e))
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400

    features, truth = game_samples[sample_id]
    model_prediction = None
    if model is not None:
        model_prediction = int(model.predict(np.array(features).reshape(1, -1))[0])
    app.logger.info("Game guess: sample %d, player %d, model %s, truth %d",
                    sample_id, guess, model_prediction, truth)
    return jsonify({
        "truth": truth,
        "truth_name": iris_target_names[truth],
        "player_correct": guess == truth,
        "model_prediction": model_prediction,
        "model_prediction_name": iris_target_names[model_prediction] if model_prediction is not None else None,
        "model_correct": model_prediction == truth if model_prediction is not None else None,
    })

@app.route("/health")
def health_check():
    # Basic health check
    if model is not None:
        app.logger.info("Health check: OK")
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    else:
        app.logger.warning("Health check: Model not loaded")
        return jsonify({"status": "unhealthy", "model_loaded": False}), 500

if __name__ == "__main__":
    # Ensure the app listens on 0.0.0.0 to be accessible externally if deployed/exposed
    app.run(host="0.0.0.0", port=5000, debug=False) # debug=False for production as per guidelines

