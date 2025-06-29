#!/usr/bin/env python3
"""
California Housing Price Predictor

This script loads a pre-trained pipeline model (directly embedded) 
and makes predictions on new data.
"""

import joblib
import pandas as pd
import os
from pathlib import Path

# Constants
MODEL_FILENAME = "california_housing_lr_model.pkl"  # or .pkl
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME

def load_model():
    """Load the saved pipeline model from the predefined path"""
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def prepare_sample_data():
    """Prepare sample input data for prediction"""
    data = {
        'MedInc': [3.5, 5.0, 2.0],          # Median income in $10,000s
        'HouseAge': [25, 12, 40],            # House age in years
        'AveRooms': [5.0, 6.5, 3.0],         # Average rooms
        'AveBedrms': [1.0, 1.5, 0.8],        # Average bedrooms
        'Population': [1000, 500, 1200],     # Population
        'AveOccup': [2.5, 3.0, 4.0],         # Average occupancy
        'Latitude': [34.5, 36.0, 33.8],      # Latitude
        'Longitude': [-118.5, -119.0, -117.8], # Longitude
        'AgeCategory': ['Mid', 'New', 'Old']  # Categorical feature
    }
    return pd.DataFrame(data)

def make_predictions(model, input_data):
    """Make predictions and return results"""
    try:
        predictions = model.predict(input_data)
        results = input_data.copy()
        results['PredictedPrice'] = predictions * 100000  # Convert to dollars
        return results
    except Exception as e:
        print(f"Prediction error: {e}")
        raise

def display_results(results):
    """Display predictions in a user-friendly format"""
    print("\nCalifornia Housing Price Predictions")
    print("=" * 50)
    
    # Format numeric columns
    pd.set_option('display.float_format', lambda x: f'${x:,.2f}')
    
    # Display all results
    print("\nDetailed Predictions:")
    print(results[['MedInc', 'HouseAge', 'AgeCategory', 'AveRooms', 'PredictedPrice']])
    
    # Display summary
    print("\nSummary:")
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\nProperty #{i+1}:")
        print(f"  - Income: ${row['MedInc']*10000:,.0f}/year")
        print(f"  - {row['HouseAge']} year old house ({row['AgeCategory']})")
        print(f"  - {row['AveRooms']:.1f} rooms")
        print(f"  Predicted Value: {row['PredictedPrice']:,.2f}")

def main():
    print("California Housing Price Prediction Service")
    print("Loading model...")
    
    try:
        # Load model directly
        model = load_model()
        
        # Prepare sample data (replace with your actual data source)
        input_data = prepare_sample_data()
        
        # Make predictions
        results = make_predictions(model, input_data)
        
        # Display results
        display_results(results)
        
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
