import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import joblib
import os

# Save the model and data relative to this script so it works on any machine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Save the dataset for potential future use or reference in the data pipeline
X.to_csv(os.path.join(DATA_DIR, "iris_features.csv"), index=False)
y.to_csv(os.path.join(DATA_DIR, "iris_target.csv"), index=False, header=["target"])

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 30% test, 70% training

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the trained model
model_path = os.path.join(MODEL_DIR, "iris_classifier_rf.joblib")
joblib.dump(clf, model_path)

print(f"Model trained and saved to {model_path}")
print(f"Dataset features saved to {os.path.join(DATA_DIR, 'iris_features.csv')}")
print(f"Dataset target saved to {os.path.join(DATA_DIR, 'iris_target.csv')}")
