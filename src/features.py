import pandas as pd
import numpy as np
from datetime import datetime
import holidays

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineering components"""
        self.us_holidays = holidays.US()
        
    def prepare_features(self, flight_info):
        """
        Prepare features from raw flight info
        
        Parameters:
        -----------
        flight_info : dict
            Raw flight information with keys like:
            - airline: str (airline code)
            - origin: str (origin airport code)
            - dest: str (destination airport code)
            - distance: float (flight distance in miles)
            - scheduled_time: float (scheduled flight time in minutes)
            - departure_date: str (YYYY-MM-DD)
            - departure_time: str (HH:MM)
            
        Returns:
        --------
        dict
            Engineered features ready for model prediction
        """
        features = {}
        
        # Copy original features
        features.update(flight_info)
        
        # Parse date and time
        if 'departure_date' in flight_info and 'departure_time' in flight_info:
            dt_str = f"{flight_info['departure_date']} {flight_info['departure_time']}"
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        elif 'departure_datetime' in flight_info:
            dt = datetime.strptime(flight_info['departure_datetime'], "%Y-%m-%d %H:%M")
        else:
            dt = datetime.now()
        
        # Add temporal features
        features['day_of_week'] = dt.weekday()
        features['month'] = dt.month
        features['day'] = dt.day
        features['hour'] = dt.hour
        
        # Derive time block features
        if 5 <= dt.hour < 12:
            features['time_block'] = 'morning'
        elif 12 <= dt.hour < 17:
            features['time_block'] = 'afternoon'
        elif 17 <= dt.hour < 21:
            features['time_block'] = 'evening'
        else:
            features['time_block'] = 'night'
        
        # Holiday indicator
        features['is_holiday'] = 1 if dt.date() in self.us_holidays else 0
        
        # Handle categorical features (one-hot encoding)
        if 'airline' in flight_info:
            features[f"airline_{flight_info['airline']}"] = 1
            
        if 'origin' in flight_info:
            features[f"origin_{flight_info['origin']}"] = 1
            
        if 'dest' in flight_info:
            features[f"dest_{flight_info['dest']}"] = 1
            
        if 'time_block' in features:
            features[f"time_block_{features['time_block']}"] = 1
            
        return features 