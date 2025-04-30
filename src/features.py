import pandas as pd
import numpy as np
from datetime import datetime
import holidays

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineering components"""
        self.us_holidays = holidays.US()
        
        # Map airline codes to full names expected by the model
        self.airline_map = {
            "AA": "AIRLINE_American Airlines Inc.",
            "DL": "AIRLINE_Delta Air Lines Inc.",
            "UA": "AIRLINE_United Air Lines Inc.",
            "WN": "AIRLINE_Southwest Airlines Co.",
            "B6": "AIRLINE_JetBlue Airways",
            "AS": "AIRLINE_Alaska Airlines Inc.",
            "NK": "AIRLINE_Spirit Air Lines",
            "F9": "AIRLINE_Frontier Airlines Inc.",
            "G4": "AIRLINE_Allegiant Air",
            "HA": "AIRLINE_Hawaiian Airlines Inc.",
            "SY": "AIRLINE_Sun Country Airlines d/b/a MN Airlines",
            "MQ": "AIRLINE_Envoy Air",
            "OH": "AIRLINE_PSA Airlines Inc.",
            "YX": "AIRLINE_Republic Airways",
            "OO": "AIRLINE_SkyWest Airlines Inc.",
            "YV": "AIRLINE_Mesa Airlines Inc.",
            "9E": "AIRLINE_Endeavor Air Inc."
        }
        
        # Top airports the model was likely trained on (based on passenger volume)
        self.top_airports = [
            "ATL", "DFW", "DEN", "ORD", "LAX", "CLT", "LAS", "PHX", 
            "MCO", "SEA", "MIA", "JFK", "EWR", "SFO", "BOS", "MSP", 
            "DTW", "FLL", "PHL", "BWI", "SLC", "IAD", "SAN", "IAH", 
            "TPA", "AUS", "BNA", "PDX", "STL", "MCI"
        ]
        
    def prepare_features(self, flight_info):
        """
        Prepare features from raw flight info to match model expectations
        
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
        # Initialize features with all expected features set to 0
        features = {}
        
        # Initialize all possible airline features
        for airline_name in self.airline_map.values():
            features[airline_name] = 0
            
        # Initialize all possible airport features
        for airport in self.top_airports:
            features[f"ORIGIN_{airport}"] = 0
            features[f"DEST_{airport}"] = 0
            
        # Initialize common numeric features
        for feature in ["DAY_OF_WEEK", "MONTH", "DAY", "HOUR", "DISTANCE", 
                       "SCHEDULED_TIME", "IS_HOLIDAY"]:
            features[feature] = 0
            
        # Initialize time block features
        for block in ["MORNING", "AFTERNOON", "EVENING", "NIGHT"]:
            features[f"TIME_BLOCK_{block}"] = 0
        
        # Parse date and time
        if 'departure_date' in flight_info and 'departure_time' in flight_info:
            dt_str = f"{flight_info['departure_date']} {flight_info['departure_time']}"
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        elif 'departure_datetime' in flight_info:
            dt = datetime.strptime(flight_info['departure_datetime'], "%Y-%m-%d %H:%M")
        else:
            dt = datetime.now()
        
        # Add temporal features in the correct format expected by the model
        features['DAY_OF_WEEK'] = dt.weekday()
        features['MONTH'] = dt.month
        features['DAY'] = dt.day
        features['HOUR'] = dt.hour
        
        # Add distance in the format expected by the model
        if 'distance' in flight_info:
            features['DISTANCE'] = float(flight_info['distance'])
        
        # Add scheduled time
        if 'scheduled_time' in flight_info:
            features['SCHEDULED_TIME'] = float(flight_info['scheduled_time'])
            
        # Holiday indicator
        features['IS_HOLIDAY'] = 1 if dt.date() in self.us_holidays else 0
        
        # Handle airline - map the code to the full name expected by the model
        if 'airline' in flight_info:
            airline_code = flight_info['airline']
            if airline_code in self.airline_map:
                features[self.airline_map[airline_code]] = 1
            else:
                # Default to most common airline if not found
                features["AIRLINE_Delta Air Lines Inc."] = 1
        
        # Handle origin and destination
        if 'origin' in flight_info:
            origin = flight_info['origin']
            if origin in self.top_airports:
                features[f"ORIGIN_{origin}"] = 1
            else:
                # Default to ATL (most common origin) if airport not in list
                features["ORIGIN_ATL"] = 1
            
        if 'dest' in flight_info:
            dest = flight_info['dest']
            if dest in self.top_airports:
                features[f"DEST_{dest}"] = 1
            else:
                # Default to DFW (common destination) if airport not in list
                features["DEST_DFW"] = 1
            
        # Time block features
        time_block = ""
        if 5 <= dt.hour < 12:
            time_block = "MORNING"
        elif 12 <= dt.hour < 17:
            time_block = "AFTERNOON"
        elif 17 <= dt.hour < 21:
            time_block = "EVENING"
        else:
            time_block = "NIGHT"
            
        features[f"TIME_BLOCK_{time_block}"] = 1
        
        # Add derived features that might help
        # Time of day matters a lot for delays
        features["IS_EARLY_MORNING"] = 1 if 5 <= dt.hour < 9 else 0
        features["IS_LATE_NIGHT"] = 1 if dt.hour >= 20 or dt.hour < 5 else 0
        
        # Day type features
        features["IS_WEEKEND"] = 1 if dt.weekday() >= 5 else 0  # 5=Sat, 6=Sun
        
        # Long flight feature
        features["IS_LONG_FLIGHT"] = 1 if features.get('DISTANCE', 0) > 1500 else 0
            
        return features 