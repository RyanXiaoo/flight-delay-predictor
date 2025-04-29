import os
import json
import pickle
import pandas as pd
import numpy as np
from .features import FeatureEngineer

class FlightDelayPredictor:
    def __init__(self, model_path='models/delay_prediction_model.pkl'):
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
        """
        # Prepare features
        features = self.feature_engineer.prepare_features(flight_info)
        
        # Convert to DataFrame for prediction
        features_df = pd.DataFrame([features])
        
        # Ensure feature columns match model's expected input
        if hasattr(self.model, 'feature_names_in_'):
            # For sklearn models that store feature names
            missing_cols = set(self.model.feature_names_in_) - set(features_df.columns)
            extra_cols = set(features_df.columns) - set(self.model.feature_names_in_)
            
            # Add missing columns with zeros
            for col in missing_cols:
                features_df[col] = 0
                
            # Keep only required columns in the right order
            features_df = features_df[self.model.feature_names_in_]
        
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
        
        # Prepare result
        result = {
            'probability': float(probability),
            'is_delayed': bool(is_delayed),
            'risk_category': risk_category
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