import os
import json
import pickle
import pandas as pd
import numpy as np
from .features import FeatureEngineer

class FlightDelayPredictor:
    def __init__(self, model_path='models/flight_delay_model_tuned.pkl'):
        """
        Initialize the flight delay predictor
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model pickle file
        """
        self.model_path = model_path
        self.feature_engineer = FeatureEngineer()
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Load any additional model metadata if available
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
        # Cache the feature names the model expects
        self.expected_feature_names = []
        if hasattr(self.model, 'feature_names_in_'):
            self.expected_feature_names = list(self.model.feature_names_in_)
        
    def predict_delay(self, flight_info):
        """
        Predict flight delay probability
        
        Parameters:
        -----------
        flight_info : dict
            Flight information (see FeatureEngineer.prepare_features for format)
            
        Returns:
        --------
        dict
            Prediction results including:
            - probability: float (probability of delay)
            - is_delayed: bool (True if probability > 0.5)
            - risk_category: str (low/medium/high)
            - estimated_delay_min: float (if available in model)
            - feature_importance: dict (top features influencing prediction)
        """
        # Prepare features
        features = self.feature_engineer.prepare_features(flight_info)
        
        # Convert to DataFrame for prediction
        features_df = pd.DataFrame([features])
        
        # Save original features for analysis
        original_features = features_df.copy()
        
        # Ensure feature columns match model's expected input
        if self.expected_feature_names:
            # Add missing columns with zeros
            missing_cols = set(self.expected_feature_names) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0
                
            # Keep only required columns in the right order
            features_df = features_df[self.expected_feature_names]
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            # For models that output probabilities
            probability = self.model.predict_proba(features_df)[0][1]  # Probability of class 1 (delayed)
        else:
            # For models that output raw predictions
            probability = self.model.predict(features_df)[0]
        
        # Determine if delayed
        is_delayed = probability > 0.5
        
        # Categorize risk
        risk_category = self._categorize_risk(probability)
        
        # Get feature importance for this prediction if possible
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            # Create a mapping of feature names to their importance
            all_importances = dict(zip(self.expected_feature_names, self.model.feature_importances_))
            
            # Only include features that are actually set (non-zero) in the original data
            for col in original_features.columns:
                if col in all_importances and original_features[col].values[0] != 0:
                    feature_importance[col] = all_importances[col]
            
            # Sort by importance and take top 5
            feature_importance = dict(sorted(feature_importance.items(), 
                                             key=lambda x: x[1], reverse=True)[:5])
        
        # Prepare result
        result = {
            'probability': float(probability),
            'is_delayed': bool(is_delayed),
            'risk_category': risk_category,
            'feature_importance': feature_importance
        }
        
        # Add estimated delay minutes if available
        if hasattr(self, 'delay_estimator'):
            result['estimated_delay_min'] = self.delay_estimator.predict(features_df)[0]
            
        return result
    
    def _categorize_risk(self, probability):
        """
        Categorize delay risk based on probability
        
        Parameters:
        -----------
        probability : float
            Probability of delay
            
        Returns:
        --------
        str
            Risk category: 'low', 'medium', or 'high'
        """
        if probability < 0.3:
            return 'low'
        elif probability < 0.6:
            return 'medium'
        else:
            return 'high' 