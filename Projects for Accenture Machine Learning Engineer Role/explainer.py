"""
SHAP Explainability Module

Provides interpretable explanations for vehicle failure predictions using SHAP values.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """SHAP-based model explainability for vehicle failure predictions."""
    
    def __init__(self, model, feature_names, scaler=None):
        """
        Initialize explainer.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            scaler: Feature scaler (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background, max_samples=100):
        """
        Create SHAP explainer with background dataset.
        
        Args:
            X_background: Background dataset for SHAP (DataFrame or array)
            max_samples: Maximum samples to use for background
        """
        logger.info("Creating SHAP explainer...")
        
        # Sample background data if too large
        if len(X_background) > max_samples:
            background_sample = X_background.sample(n=max_samples, random_state=42)
        else:
            background_sample = X_background
        
        # Scale if scaler provided
        if self.scaler is not None:
            background_sample = self.scaler.transform(background_sample)
        
        # Create TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model, background_sample)
        logger.info("SHAP explainer created")
        
    def explain_predictions(self, X):
        """
        Generate SHAP values for predictions.
        
        Args:
            X: Feature matrix to explain
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Scale if scaler provided
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_scaled)
        
        return self.shap_values
    
    def explain_single_prediction(self, vehicle_data, top_n=10):
        """
        Explain a single vehicle prediction with top contributing factors.
        
        Args:
            vehicle_data: Dictionary or DataFrame with vehicle sensor data
            top_n: Number of top features to return
            
        Returns:
            Dictionary with explanation details
        """
        # Convert to DataFrame
        if isinstance(vehicle_data, dict):
            df = pd.DataFrame([vehicle_data])
        else:
            df = pd.DataFrame([vehicle_data])
        
        # Ensure all features present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        X = df[self.feature_names]
        
        # Get prediction
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        prediction_proba = self.model.predict_proba(X_scaled)[0, 1]
        risk_score = int(prediction_proba * 100)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Get base value (expected value)
        base_value = self.explainer.expected_value
        
        # Create explanation DataFrame
        explanation_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': X.iloc[0].values,
            'shap_value': shap_values[0]
        })
        
        # Sort by absolute SHAP value
        explanation_df['abs_shap'] = explanation_df['shap_value'].abs()
        explanation_df = explanation_df.sort_values('abs_shap', ascending=False)
        
        # Get top contributors
        top_features = explanation_df.head(top_n)
        
        # Format explanation
        explanation = {
            'risk_score': risk_score,
            'base_risk': float(base_value),
            'top_factors': []
        }
        
        for _, row in top_features.iterrows():
            factor = {
                'feature': self._format_feature_name(row['feature']),
                'value': float(row['value']),
                'contribution': float(row['shap_value']),
                'impact': 'increases' if row['shap_value'] > 0 else 'decreases'
            }
            explanation['top_factors'].append(factor)
        
        return explanation
    
    def _format_feature_name(self, feature):
        """Convert feature name to human-readable format."""
        name_mapping = {
            'engine_temp': 'Engine Temperature',
            'oil_pressure': 'Oil Pressure',
            'coolant_level': 'Coolant Level',
            'battery_voltage': 'Battery Voltage',
            'fuel_consumption': 'Fuel Consumption',
            'vibration_level': 'Vibration Level',
            'transmission_temp': 'Transmission Temperature',
            'brake_pressure': 'Brake Pressure',
            'tire_pressure_fl': 'Tire Pressure (Front Left)',
            'tire_pressure_fr': 'Tire Pressure (Front Right)',
            'tire_pressure_rl': 'Tire Pressure (Rear Left)',
            'tire_pressure_rr': 'Tire Pressure (Rear Right)',
            'mileage': 'Total Mileage',
            'days_since_maintenance': 'Days Since Maintenance',
            'tire_pressure_asymmetry': 'Tire Pressure Asymmetry'
        }
        
        # Handle rolling and lag features
        for base_name, display_name in name_mapping.items():
            if feature.startswith(base_name):
                if '_7d_mean' in feature:
                    return f'{display_name} (7-day avg)'
                elif '_7d_std' in feature:
                    return f'{display_name} (7-day variability)'
                elif '_lag1' in feature:
                    return f'{display_name} (previous day)'
                elif '_change' in feature:
                    return f'{display_name} (rate of change)'
        
        return name_mapping.get(feature, feature.replace('_', ' ').title())
    
    def get_global_feature_importance(self, X, top_n=20):
        """
        Calculate global feature importance using mean absolute SHAP values.
        
        Args:
            X: Feature matrix
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        # Calculate mean absolute SHAP values
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.shap_values).mean(axis=0)
        })
        
        importance = importance.sort_values('importance', ascending=False)
        importance['importance_pct'] = (
            importance['importance'] / importance['importance'].sum() * 100
        )
        
        return importance.head(top_n)
    
    def plot_waterfall(self, vehicle_data, output_path=None, max_display=10):
        """
        Create waterfall plot for single prediction.
        
        Args:
            vehicle_data: Vehicle sensor data
            output_path: Path to save plot (optional)
            max_display: Maximum features to display
        """
        # Prepare data
        if isinstance(vehicle_data, dict):
            df = pd.DataFrame([vehicle_data])
        else:
            df = pd.DataFrame([vehicle_data])
        
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        X = df[self.feature_names]
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=self.explainer.expected_value,
            data=X.iloc[0].values,
            feature_names=self.feature_names
        )
        
        # Create plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved waterfall plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_summary(self, X, output_path=None, max_display=20):
        """
        Create summary plot showing global feature importance.
        
        Args:
            X: Feature matrix
            output_path: Path to save plot (optional)
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, X, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_force(self, vehicle_data, output_path=None):
        """
        Create force plot for single prediction.
        
        Args:
            vehicle_data: Vehicle sensor data
            output_path: Path to save plot (optional)
        """
        # Prepare data
        if isinstance(vehicle_data, dict):
            df = pd.DataFrame([vehicle_data])
        else:
            df = pd.DataFrame([vehicle_data])
        
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        X = df[self.feature_names]
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Create force plot
        force_plot = shap.force_plot(
            self.explainer.expected_value,
            shap_values[0],
            X.iloc[0],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved force plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_text_explanation(self, vehicle_data):
        """
        Generate human-readable text explanation.
        
        Args:
            vehicle_data: Vehicle sensor data
            
        Returns:
            Formatted text explanation
        """
        explanation = self.explain_single_prediction(vehicle_data)
        
        text = f"Vehicle Failure Risk Assessment\n"
        text += f"{'='*60}\n\n"
        text += f"Risk Score: {explanation['risk_score']}/100\n"
        
        if explanation['risk_score'] >= 80:
            text += f"Risk Level: CRITICAL\n"
        elif explanation['risk_score'] >= 60:
            text += f"Risk Level: HIGH\n"
        elif explanation['risk_score'] >= 40:
            text += f"Risk Level: MEDIUM\n"
        elif explanation['risk_score'] >= 20:
            text += f"Risk Level: LOW\n"
        else:
            text += f"Risk Level: MINIMAL\n"
        
        text += f"\nTop Contributing Factors:\n"
        text += f"{'-'*60}\n"
        
        for i, factor in enumerate(explanation['top_factors'], 1):
            impact_sign = '+' if factor['contribution'] > 0 else ''
            text += f"\n{i}. {factor['feature']}\n"
            text += f"   Current Value: {factor['value']:.2f}\n"
            text += f"   Risk Contribution: {impact_sign}{factor['contribution']:.2f}\n"
            text += f"   Impact: {factor['impact']} risk\n"
        
        text += f"\n{'='*60}\n"
        
        return text


if __name__ == '__main__':
    # Example usage
    import joblib
    
    # Load model
    model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/feature_scaler.pkl')
    
    # Load feature names
    import json
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Load sample data
    df = pd.read_csv('data/synthetic/vehicle_sensor_data.csv')
    X = df[feature_names].head(100)
    
    # Create explainer
    explainer = ModelExplainer(model, feature_names, scaler)
    explainer.create_explainer(X)
    
    # Explain single prediction
    sample = df.iloc[0]
    explanation = explainer.generate_text_explanation(sample)
    print(explanation)
