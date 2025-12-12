"""
XGBoost Classifier for Vehicle Failure Prediction

Trains and evaluates XGBoost model for predicting vehicle component failures
14 days in advance.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleFailurePredictor:
    """XGBoost-based predictive maintenance model."""
    
    def __init__(self, model_params=None):
        """
        Initialize predictor with model parameters.
        
        Args:
            model_params: Dictionary of XGBoost parameters
        """
        self.model_params = model_params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 3,  # Handle class imbalance
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(self, df, target_col='failure_within_14_days'):
        """
        Prepare features for training/prediction.
        
        Args:
            df: Input DataFrame
            target_col: Target variable column name
            
        Returns:
            X, y: Feature matrix and target vector
        """
        # Define feature columns
        sensor_features = [
            'engine_temp', 'oil_pressure', 'coolant_level', 'battery_voltage',
            'fuel_consumption', 'vibration_level', 'transmission_temp',
            'brake_pressure', 'tire_pressure_fl', 'tire_pressure_fr',
            'tire_pressure_rl', 'tire_pressure_rr'
        ]
        
        rolling_features = [
            f'{feat}_{window}d_{stat}'
            for feat in ['engine_temp', 'oil_pressure', 'vibration_level',
                        'fuel_consumption', 'transmission_temp']
            for window in [7]
            for stat in ['mean', 'std']
        ]
        
        lag_features = [
            f'{feat}_lag1'
            for feat in ['engine_temp', 'oil_pressure', 'vibration_level',
                        'fuel_consumption', 'transmission_temp']
        ]
        
        change_features = [
            f'{feat}_change'
            for feat in ['engine_temp', 'oil_pressure', 'vibration_level',
                        'fuel_consumption', 'transmission_temp']
        ]
        
        usage_features = [
            'mileage', 'days_since_maintenance', 'tire_pressure_asymmetry'
        ]
        
        self.feature_names = (
            sensor_features + rolling_features + lag_features + 
            change_features + usage_features
        )
        
        # Extract features and target
        X = df[self.feature_names].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X, y
    
    def train(self, df, test_size=0.2, validation_size=0.1):
        """
        Train XGBoost model with train/validation/test split.
        
        Args:
            df: Training DataFrame
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Preparing features...")
        X, y = self.prepare_features(df)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Positive class rate: {y_train.mean():.2%}")
        
        # Scale features
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info("Training XGBoost model...")
        self.model = xgb.XGBClassifier(**self.model_params)
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        logger.info(f"Best iteration: {self.model.best_iteration}")
        
        # Evaluate on all sets
        metrics = {}
        for name, X_set, y_set in [
            ('train', X_train_scaled, y_train),
            ('validation', X_val_scaled, y_val),
            ('test', X_test_scaled, y_test)
        ]:
            set_metrics = self._evaluate(X_set, y_set, name)
            metrics[name] = set_metrics
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return metrics
    
    def _evaluate(self, X, y, set_name='test'):
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            set_name: Name of dataset (for logging)
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        logger.info(f"\n{set_name.upper()} SET METRICS:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        logger.info(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn:5d}  FP: {fp:5d}")
        logger.info(f"  FN: {fn:5d}  TP: {tp:5d}")
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix or DataFrame
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].fillna(X.mean())
        
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Convert to risk scores (0-100)
        risk_scores = (probabilities * 100).astype(int)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_scores': risk_scores
        }
    
    def predict_single(self, vehicle_data):
        """
        Predict failure risk for a single vehicle.
        
        Args:
            vehicle_data: Dictionary or Series with vehicle sensor data
            
        Returns:
            Dictionary with prediction details
        """
        # Convert to DataFrame
        if isinstance(vehicle_data, dict):
            df = pd.DataFrame([vehicle_data])
        else:
            df = pd.DataFrame([vehicle_data.to_dict()])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        result = self.predict(df)
        
        risk_score = int(result['risk_scores'][0])
        probability = float(result['probabilities'][0])
        
        # Determine risk level and recommendation
        if risk_score >= 80:
            risk_level = 'CRITICAL'
            recommendation = 'Immediate inspection required - ground vehicle if possible'
        elif risk_score >= 60:
            risk_level = 'HIGH'
            recommendation = 'Schedule inspection within 3 days'
        elif risk_score >= 40:
            risk_level = 'MEDIUM'
            recommendation = 'Schedule inspection within 7 days'
        elif risk_score >= 20:
            risk_level = 'LOW'
            recommendation = 'Monitor closely, routine maintenance'
        else:
            risk_level = 'MINIMAL'
            recommendation = 'Continue normal operations'
        
        return {
            'risk_score': risk_score,
            'probability': probability,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'prediction': int(result['predictions'][0])
        }
    
    def save_model(self, model_dir='models'):
        """
        Save trained model and artifacts.
        
        Args:
            model_dir: Directory to save model files
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / 'xgboost_model.pkl'
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save scaler
        scaler_path = model_dir / 'feature_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature names
        features_path = model_dir / 'feature_names.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Saved feature names to {features_path}")
        
        # Save feature importance
        importance_path = model_dir / 'feature_importance.csv'
        self.feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance to {importance_path}")
        
        # Save model parameters
        params_path = model_dir / 'model_params.json'
        with open(params_path, 'w') as f:
            json.dump(self.model_params, f, indent=2)
        logger.info(f"Saved model parameters to {params_path}")
    
    def load_model(self, model_dir='models'):
        """
        Load trained model and artifacts.
        
        Args:
            model_dir: Directory containing model files
        """
        model_dir = Path(model_dir)
        
        # Load model
        model_path = model_dir / 'xgboost_model.pkl'
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load scaler
        scaler_path = model_dir / 'feature_scaler.pkl'
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
        
        # Load feature names
        features_path = model_dir / 'feature_names.json'
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        logger.info(f"Loaded feature names from {features_path}")
        
        # Load feature importance
        importance_path = model_dir / 'feature_importance.csv'
        self.feature_importance = pd.read_csv(importance_path)
        logger.info(f"Loaded feature importance from {importance_path}")


def main():
    """Command line interface for model training."""
    parser = argparse.ArgumentParser(
        description='Train XGBoost model for vehicle failure prediction'
    )
    parser.add_argument(
        '--data', type=str,
        default='data/synthetic/vehicle_sensor_data.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--output-dir', type=str, default='models',
        help='Directory to save trained model'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} records")
    
    # Train model
    predictor = VehicleFailurePredictor()
    metrics = predictor.train(df, test_size=args.test_size)
    
    # Save model
    predictor.save_model(args.output_dir)
    
    # Save metrics
    metrics_path = Path(args.output_dir) / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved training metrics to {metrics_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {args.output_dir}")
    print(f"Test Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test ROC AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"Test Recall: {metrics['test']['recall']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
