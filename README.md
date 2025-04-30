# Flight Delay Predictor

A machine learning-based application that predicts the likelihood of flight delays based on various factors such as airline, origin/destination airports, flight distance, and departure time.

![Flight Delay Predictor](https://raw.githubusercontent.com/username/flight-delay-predictor/main/assets/app_screenshot.png)

## Features

-   Predicts flight delay probability based on flight information
-   User-friendly Streamlit web interface
-   Visualizes prediction results with intuitive graphics
-   Auto-calculates typical flight distances and times between major airports
-   Color-coded risk assessment (low, medium, high)

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/yourusername/flight-delay-predictor.git
    cd flight-delay-predictor
    ```

2. Install required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit web app:

    ```
    streamlit run app.py
    ```

2. Open your browser and navigate to http://localhost:8501

3. Enter your flight details:

    - Select airline
    - Choose origin and destination airports
    - Verify flight distance and scheduled time (auto-calculated)
    - Set departure date and time

4. Click "Predict Flight Delay" to see the results

## Project Structure

```
flight-delay-predictor/
├── app.py                  # Streamlit web application
├── src/                    # Core modules
│   ├── features.py         # Feature engineering
│   ├── model.py            # Delay prediction models
│   └── predictor.py        # Main prediction interface
├── models/                 # Trained models
│   └── flight_delay_model_tuned.pkl
├── data/                   # Data files
│   ├── flights_processed_v1.csv
│   ├── flights_weather_features.csv
│   └── weather_cache/      # Cached weather API responses
├── notebooks/              # Jupyter notebooks for data analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 06_weather_enrichment.ipynb
│   └── 07_model_development.ipynb
└── requirements.txt        # Project dependencies
```

## Data Processing Pipeline

This project follows a structured data processing pipeline:

1. **Data Collection**: Historical flight data combined with weather information
2. **Preprocessing**: Cleaning and preparing the flight data
3. **Feature Engineering**: Creating relevant features for the model
4. **Weather Enrichment**: Adding weather data from Open-Meteo API
5. **Model Development**: Training and tuning the prediction model
6. **Web Application**: User-friendly interface for making predictions

## Technologies Used

-   Python 3.9+
-   Streamlit for web interface
-   Pandas for data manipulation
-   Scikit-learn for machine learning
-   Altair for data visualization

## Future Improvements

-   Add real-time weather data integration
-   Incorporate flight network effects for more accurate predictions
-   Create a mobile app version
-   Add flight tracking capabilities
-   Include more historical data for improved model accuracy

## License

MIT License

## Acknowledgments

-   Data sourced from public flight delay datasets
-   Weather data provided by Open-Meteo API

## Data

The raw historical flight data is **not** included in this repository due to size constraints. You can download the dataset from Kaggle:

-   Flight Delay and Cancellation Dataset 2019–2023: [https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)

After downloading, place the relevant CSV files into a `data/` folder at the root of this project (e.g. `data/flights_processed_v1.csv`).
