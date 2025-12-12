"""
LightGBM Classifier for Hospital Readmission Prediction

Trains and evaluates LightGBM model for predicting 30-day readmissions
with SMOTE for handling class imbalance.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReadmissionPredictor:
    """LightGBM-based hospital readmission prediction model."""
    
    def __init__(self, model_params=None):
        """
        Initialize predictor with model parameters.
        
        Args:
            model_params: Dictionary of LightGBM parameters
        """
        self.model_params = model_params or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        self.model = None
        self.calibrated_model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(self, df, target_col='readmitted_30d'):
        """
        Prepare features for training/prediction.
        
        Args:
            df: Input DataFrame
            target_col: Target variable column name
            
        Returns:
            X, y: Feature matrix and target vector
        """
        # Numeric features
        numeric_features = [
            'age', 'comorbidity_count', 'prior_admissions_30d',
            'prior_admissions_90d', 'prior_admissions_1y',
            'prior_ed_visits_30d', 'prior_ed_visits_90d',
            'length_of_stay', 'medication_count',
            'distance_to_facility_miles', 'social_support_score',
            'service_connected_pct'
        ]
        
        # Binary features
        binary_features = [
            'service_connected', 'high_risk_medications',
            'mental_health_diagnosis', 'substance_use_disorder'
        ]
        
        # Categorical features
        categorical_features = [
            'gender', 'race', 'primary_diagnosis',
            'housing_status', 'discharge_disposition'
        ]
        
        df_processed = df.copy()
        
        # Encode categorical variables
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                df_processed[col] = df[col].astype(str).apply(
                    lambda x: self.label_encoders[col].transform([x])[0]
                    if x in self.label_encoders[col].classes_
                    else -1
                )
        
        # Convert binary features to int
        for col in binary_features:
            df_processed[col] = df_processed[col].astype(int)
        
        # Create derived features
        df_processed['admission_intensity'] = (
            df_processed['prior_admissions_30d'] * 3 +
            df_processed['prior_admissions_90d']
        )
        
        df_processed['ed_utilization'] = (
            df_processed['prior_ed_visits_30d'] * 2 +
            df_processed['prior_ed_visits_90d']
        )
        
        df_processed['high_utilizer'] = (
            (df_processed['prior_admissions_1y'] >= 3) |
            (df_processed['prior_ed_visits_90d'] >= 4)
        ).astype(int)
        
        df_processed['polypharmacy'] = (df_processed['medication_count'] >= 10).astype(int)
        
        df_processed['complex_patient'] = (
            (df_processed['comorbidity_count'] >= 5) &
            (df_processed['medication_count'] >= 10)
        ).astype(int)
        
        df_processed['social_risk'] = (
            (df_processed['housing_status'] != 0) |  # Not stable housing
            (df_processed['social_support_score'] <= 2)
        ).astype(int)
        
        # All feature columns
        self.feature_names = (
            numeric_features + binary_features + categorical_features +
            ['admission_intensity', 'ed_utilization', 'high_utilizer',
             'polypharmacy', 'complex_patient', 'social_risk']
        )
        
        X = df_processed[self.feature_names].copy()
        y = df_processed[target_col].copy() if target_col in df_processed.columns else None
        
        return X, y
    
    def train(self, df, test_size=0.2, use_smote=True, calibrate=True):
        """
        Train LightGBM model with optional SMOTE and calibration.
        
        Args:
            df: Training DataFrame
            test_size: Proportion of data for testing
            use_smote: Whether to use SMOTE for class imbalance
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Preparing features...")
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Readmission rate (train): {y_train.mean():.2%}")
        logger.info(f"Readmission rate (test): {y_test.mean():.2%}")
        
        # Scale features
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance with SMOTE
        if use_smote:
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            logger.info(f"After SMOTE: {len(X_train_scaled)} samples")
            logger.info(f"Balanced readmission rate: {y_train.mean():.2%}")
        
        # Train model
        logger.info("Training LightGBM model...")
        self.model = lgb.LGBMClassifier(**self.model_params)
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        logger.info(f"Best iteration: {self.model.best_iteration_}")
        
        # Calibrate probabilities
        if calibrate:
            logger.info("Calibrating probabilities...")
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method='sigmoid', cv=3
            )
            # Use original test set for calibration
            X_cal, X_test_cal, y_cal, y_test_cal = train_test_split(
                X_test_scaled, y_test, test_size=0.5, random_state=42, stratify=y_test
            )
            self.calibrated_model.fit(X_cal, y_cal)
        
        # Evaluate on all sets
        metrics = {}
        
        # Training set
        train_metrics = self._evaluate(X_train_scaled, y_train, 'train')
        metrics['train'] = train_metrics
        
        # Test set
        test_metrics = self._evaluate(X_test_scaled, y_test, 'test')
        metrics['test'] = test_metrics
        
        # Cross-validation
        logger.info("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1
        )
        logger.info(f"CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        metrics['cv_auc_mean'] = float(cv_scores.mean())
        metrics['cv_auc_std'] = float(cv_scores.std())
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 15 Most Important Features:")
        for idx, row in self.feature_importance.head(15).iterrows():
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
        # Use calibrated model if available
        model = self.calibrated_model if self.calibrated_model else self.model
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'specificity': 0.0,  # Calculate below
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'pr_auc': average_precision_score(y, y_pred_proba)
        }
        
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        logger.info(f"\n{set_name.upper()} SET METRICS:")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:   {metrics['precision']:.4f}")
        logger.info(f"  Recall:      {metrics['recall']:.4f}")
        logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        logger.info(f"  F1 Score:    {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
        logger.info(f"  PR AUC:      {metrics['pr_auc']:.4f}")
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
            X, _ = self.prepare_features(X)
        
        X_scaled = self.scaler.transform(X)
        
        model = self.calibrated_model if self.calibrated_model else self.model
        
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Convert to risk scores (0-100)
        risk_scores = (probabilities * 100).astype(int)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_scores': risk_scores
        }
    
    def predict_single(self, patient_data):
        """
        Predict readmission risk for a single patient.
        
        Args:
            patient_data: Dictionary or Series with patient data
            
        Returns:
            Dictionary with prediction details
        """
        # Convert to DataFrame
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = pd.DataFrame([patient_data.to_dict()])
        
        result = self.predict(df)
        
        risk_score = int(result['risk_scores'][0])
        probability = float(result['probabilities'][0])
        
        # Determine risk level and recommendation
        if risk_score >= 75:
            risk_level = 'VERY HIGH'
            recommendation = 'Intensive care coordination with weekly follow-up'
        elif risk_score >= 50:
            risk_level = 'HIGH'
            recommendation = 'Enhanced discharge planning and home health referral'
        elif risk_score >= 30:
            risk_level = 'MODERATE'
            recommendation = 'Standard discharge planning with follow-up within 7 days'
        else:
            risk_level = 'LOW'
            recommendation = 'Routine discharge planning'
        
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
        model_path = model_dir / 'lightgbm_model.pkl'
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save calibrated model
        if self.calibrated_model:
            calibrated_path = model_dir / 'calibrated_model.pkl'
            joblib.dump(self.calibrated_model, calibrated_path)
            logger.info(f"Saved calibrated model to {calibrated_path}")
        
        # Save scaler
        scaler_path = model_dir / 'feature_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save label encoders
        encoders_path = model_dir / 'label_encoders.pkl'
        joblib.dump(self.label_encoders, encoders_path)
        logger.info(f"Saved label encoders to {encoders_path}")
        
        # Save feature names
        features_path = model_dir / 'feature_names.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Saved feature names to {features_path}")
        
        # Save feature importance
        importance_path = model_dir / 'feature_importance.csv'
        self.feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance to {importance_path}")
    
    def load_model(self, model_dir='models'):
        """
        Load trained model and artifacts.
        
        Args:
            model_dir: Directory containing model files
        """
        model_dir = Path(model_dir)
        
        # Load model
        model_path = model_dir / 'lightgbm_model.pkl'
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load calibrated model
        calibrated_path = model_dir / 'calibrated_model.pkl'
        if calibrated_path.exists():
            self.calibrated_model = joblib.load(calibrated_path)
            logger.info(f"Loaded calibrated model from {calibrated_path}")
        
        # Load scaler
        scaler_path = model_dir / 'feature_scaler.pkl'
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
        
        # Load label encoders
        encoders_path = model_dir / 'label_encoders.pkl'
        self.label_encoders = joblib.load(encoders_path)
        logger.info(f"Loaded label encoders from {encoders_path}")
        
        # Load feature names
        features_path = model_dir / 'feature_names.json'
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        logger.info(f"Loaded feature names from {features_path}")


def main():
    """Command line interface for model training."""
    parser = argparse.ArgumentParser(
        description='Train LightGBM model for hospital readmission prediction'
    )
    parser.add_argument(
        '--data', type=str,
        default='data/synthetic/va_patient_data.csv',
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
    parser.add_argument(
        '--no-smote', action='store_true',
        help='Disable SMOTE oversampling'
    )
    parser.add_argument(
        '--no-calibration', action='store_true',
        help='Disable probability calibration'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} records")
    
    # Train model
    predictor = ReadmissionPredictor()
    metrics = predictor.train(
        df,
        test_size=args.test_size,
        use_smote=not args.no_smote,
        calibrate=not args.no_calibration
    )
    
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
