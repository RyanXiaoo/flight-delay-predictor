import os
import sys
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import altair as alt

# Add src directory to path for importing
sys.path.append('.')

from src.predictor import FlightDelayPredictor

# Set page configuration
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize predictor
@st.cache_resource
def load_predictor():
    model_path = 'models/flight_delay_model_tuned.pkl'
    if not os.path.exists(model_path):
        st.error(f"Error: Model not found at {model_path}")
        return None
    
    try:
        predictor = FlightDelayPredictor(model_path=model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Common airlines and major airports for dropdown options
AIRLINES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines"
}

MAJOR_AIRPORTS = {
    "ATL": "Atlanta (ATL)",
    "DFW": "Dallas/Fort Worth (DFW)",
    "DEN": "Denver (DEN)",
    "ORD": "Chicago O'Hare (ORD)",
    "LAX": "Los Angeles (LAX)",
    "CLT": "Charlotte (CLT)",
    "LAS": "Las Vegas (LAS)",
    "PHX": "Phoenix (PHX)",
    "MCO": "Orlando (MCO)",
    "SEA": "Seattle (SEA)",
    "MIA": "Miami (MIA)",
    "JFK": "New York JFK (JFK)",
    "EWR": "Newark (EWR)",
    "SFO": "San Francisco (SFO)",
    "BOS": "Boston (BOS)",
    "MSP": "Minneapolis (MSP)",
    "DTW": "Detroit (DTW)",
    "FLL": "Fort Lauderdale (FLL)",
    "PHL": "Philadelphia (PHL)",
    "BWI": "Baltimore (BWI)"
}

# Approximate flight distances and times for popular routes
ROUTE_DISTANCES = {
    ("ATL", "LAX"): {"distance": 1946, "time": 270},
    ("JFK", "LAX"): {"distance": 2475, "time": 360},
    ("ORD", "SFO"): {"distance": 1846, "time": 270},
    ("DFW", "JFK"): {"distance": 1391, "time": 195},
    ("MIA", "SEA"): {"distance": 2724, "time": 375},
    ("BOS", "ORD"): {"distance": 867, "time": 150},
    ("LAX", "JFK"): {"distance": 2475, "time": 330},
    ("ATL", "ORD"): {"distance": 606, "time": 120},
    ("DEN", "SFO"): {"distance": 967, "time": 150},
    ("LAS", "JFK"): {"distance": 2248, "time": 300},
}

# Default values for any other route
DEFAULT_SPEED = 500  # mph

def estimate_flight_parameters(origin, dest):
    """Estimate distance and flight time based on airports"""
    if (origin, dest) in ROUTE_DISTANCES:
        return ROUTE_DISTANCES[(origin, dest)]
    elif (dest, origin) in ROUTE_DISTANCES:
        return ROUTE_DISTANCES[(dest, origin)]
    else:
        # Very rough estimate for unknown routes
        # In a real app, you would use a proper distance calculator
        distance = 1000  # default miles
        time = distance / DEFAULT_SPEED * 60  # convert to minutes
        return {"distance": distance, "time": time}

def format_feature_name(feature_name):
    """Format feature names for display"""
    if feature_name.startswith("AIRLINE_"):
        # Extract airline name without prefix
        return feature_name.replace("AIRLINE_", "")
    elif feature_name.startswith("ORIGIN_"):
        return f"Origin: {feature_name.replace('ORIGIN_', '')}"
    elif feature_name.startswith("DEST_"):
        return f"Destination: {feature_name.replace('DEST_', '')}"
    elif feature_name.startswith("TIME_BLOCK_"):
        return f"Time: {feature_name.replace('TIME_BLOCK_', '').title()}"
    elif feature_name == "DAY_OF_WEEK":
        return "Day of Week"
    elif feature_name == "IS_HOLIDAY":
        return "Holiday"
    elif feature_name == "IS_WEEKEND":
        return "Weekend"
    elif feature_name == "IS_EARLY_MORNING":
        return "Early Morning"
    elif feature_name == "IS_LATE_NIGHT":
        return "Late Night"
    elif feature_name == "IS_LONG_FLIGHT":
        return "Long Flight"
    else:
        # Convert snake_case to Title Case
        return feature_name.replace("_", " ").title()

def main():
    predictor = load_predictor()
    if predictor is None:
        st.error("Could not load the prediction model. Please check the model file.")
        return
        
    # App Header
    st.title("✈️ Flight Delay Predictor")
    st.markdown("""
    This app predicts the likelihood of flight delays based on flight information.
    Enter your flight details below to get a prediction.
    """)
    
    # Create columns for form layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Information")
        
        # Airline selection
        airline = st.selectbox(
            "Airline",
            options=list(AIRLINES.keys()),
            format_func=lambda x: f"{x} - {AIRLINES[x]}"
        )
        
        # Origin and destination
        origin = st.selectbox(
            "Origin Airport",
            options=list(MAJOR_AIRPORTS.keys()),
            format_func=lambda x: MAJOR_AIRPORTS[x]
        )
        
        dest = st.selectbox(
            "Destination Airport",
            options=list(MAJOR_AIRPORTS.keys()),
            format_func=lambda x: MAJOR_AIRPORTS[x],
            index=min(1, len(MAJOR_AIRPORTS)-1)  # Default to second airport
        )
        
        # Get estimated flight parameters based on route
        flight_params = estimate_flight_parameters(origin, dest)
        
        # Flight details with defaults based on the route
        distance = st.number_input(
            "Flight Distance (miles)",
            min_value=100.0,
            max_value=5000.0,
            value=float(flight_params["distance"]),
            step=50.0
        )
        
        scheduled_time = st.number_input(
            "Scheduled Flight Time (minutes)",
            min_value=30.0,
            max_value=720.0,
            value=float(flight_params["time"]),
            step=15.0
        )
    
    with col2:
        st.subheader("Date and Time")
        
        # Date picker (default to tomorrow)
        tomorrow = datetime.now() + timedelta(days=1)
        departure_date = st.date_input(
            "Departure Date",
            value=tomorrow
        )
        
        # Time selection
        departure_time = st.time_input(
            "Departure Time",
            value=datetime.strptime("09:00", "%H:%M").time()
        )
        
        # Show selected date and time
        date_str = departure_date.strftime("%Y-%m-%d")
        time_str = departure_time.strftime("%H:%M")
        
        st.markdown(f"**Selected Departure:** {date_str} at {time_str}")
        
        # Show the selected route
        st.markdown(f"**Selected Route:** {origin} → {dest}")
        if origin == dest:
            st.warning("⚠️ Origin and destination airports are the same!")
    
    # Prediction button
    if st.button("Predict Flight Delay", type="primary"):
        with st.spinner("Predicting..."):
            # Prepare input for prediction
            flight_info = {
                'airline': airline,
                'origin': origin,
                'dest': dest,
                'distance': distance,
                'scheduled_time': scheduled_time,
                'departure_date': date_str,
                'departure_time': time_str
            }
            
            # Make prediction
            try:
                result = predictor.predict_delay(flight_info)
                
                # Display results in a nice format
                st.subheader("Prediction Results")
                
                # Create columns for displaying the results
                res_col1, res_col2, res_col3 = st.columns(3)
                
                # Format probability as percentage
                probability = result.get('probability', 0) * 100
                
                # Define risk color based on category
                risk_category = result.get('risk_category', 'Unknown')
                if risk_category == 'low':
                    risk_color = 'green'
                elif risk_category == 'medium':
                    risk_color = 'orange'
                else:
                    risk_color = 'red'
                
                with res_col1:
                    st.metric(
                        label="Delay Probability", 
                        value=f"{probability:.1f}%"
                    )
                    
                with res_col2:
                    st.metric(
                        label="Prediction", 
                        value="Likely Delayed" if result.get('is_delayed', False) else "On Time"
                    )
                    
                with res_col3:
                    st.markdown(
                        f"<h2 style='text-align: center; color: {risk_color};'>{risk_category.upper()}</h2>", 
                        unsafe_allow_html=True
                    )
                
                # Display feature importance if available
                if result.get('feature_importance'):
                    st.subheader("Key Factors Influencing Prediction")
                    
                    # Create a dataframe for the chart
                    importance_data = []
                    for feature, importance in result['feature_importance'].items():
                        importance_data.append({
                            'Feature': format_feature_name(feature),
                            'Importance': float(importance),
                        })
                    
                    importance_df = pd.DataFrame(importance_data)
                    
                    # Display as bar chart
                    chart = alt.Chart(importance_df).mark_bar().encode(
                        x=alt.X('Importance:Q', title='Relative Importance'),
                        y=alt.Y('Feature:N', sort='-x', title=None),
                        color=alt.Color('Importance:Q', 
                                       scale=alt.Scale(scheme='blues'),
                                       legend=None)
                    ).properties(
                        height=min(len(importance_data) * 40, 300)
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                
                # Visualization
                st.subheader("Probability Visualization")
                
                # Create probability chart
                chart_data = pd.DataFrame({
                    'category': ['Delay', 'On-time'],
                    'probability': [probability, 100 - probability]
                })
                
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('category:N', title=None),
                    y=alt.Y('probability:Q', title='Probability (%)'),
                    color=alt.Color('category:N', 
                                   scale=alt.Scale(range=['#FF5733', '#33FF57']), 
                                   legend=None)
                ).properties(
                    width=500,
                    height=300
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Add recommendations based on the prediction
                if result.get('is_delayed', False):
                    st.warning("""
                    ### Recommendations for Potential Delay
                    
                    Based on the prediction, this flight has a higher chance of delay. Consider:
                    - Allowing extra time for connections
                    - Checking with the airline before heading to the airport
                    - Having a backup plan for important meetings or events
                    - Setting up flight status alerts
                    """)
                
                # Add some explanation
                st.markdown("""
                ### Factors Affecting Flight Delays
                
                Common reasons for flight delays include:
                - Weather conditions at origin or destination
                - Air traffic congestion
                - Airline operational issues
                - Day of week and time of day
                - Previous flight delays (ripple effect)
                """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    # About section
    with st.expander("About this App"):
        st.markdown("""
        This Flight Delay Predictor uses a machine learning model trained on historical flight data
        to predict the likelihood of flight delays.
        
        The model considers various factors including:
        - Airline carrier
        - Origin and destination airports
        - Flight distance and scheduled duration
        - Date and time of departure
        
        **Note:** This is a demonstration app. For actual flight information, 
        please check with your airline or flight tracking services.
        """)

if __name__ == "__main__":
    main() 