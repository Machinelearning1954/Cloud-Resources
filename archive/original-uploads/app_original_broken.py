from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename=uild_log.txtuild_log.txt", level=logging.INFO, 
                    format=%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")

# Define paths
MODEL_DIR = "/home/ubuntu/iris_project/model"
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

