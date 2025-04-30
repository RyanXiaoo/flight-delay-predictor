import pickle
import pandas as pd
import numpy as np
import os

class DelayPredictor:
    def __init__(self, model_path='../models/flight_delay_model_tuned.pkl'):
        """
        Initialize the delay predictor with a trained model
        
        Parameters:
        -----------
        model_path : str
            Path to the pickled model file
        """
        # Load the model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded successfully from {model_path}")
    
    def predict(self, features):
        """
        Make delay predictions based on input features
        
        Parameters:
        -----------
        features : dict or pd.DataFrame
            Features for prediction
            
        Returns:
        --------
        dict
            Prediction results with probability and classification
        """
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Make prediction
        delay_prob = self.model.predict_proba(features)[:, 1]
        delay_pred = self.model.predict(features)
        
        return {
            "delay_probability": float(delay_prob[0]),
            "is_delayed": bool(delay_pred[0]),
            "delay_risk": self._categorize_risk(delay_prob[0])
        }
    
    def _categorize_risk(self, probability):
        """Categorize delay risk based on probability"""
        if probability < 0.2:
            return "Low"
        elif probability < 0.5:
            return "Medium"
        else:
            return "High" 