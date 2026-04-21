"""
Flask API for Taxi Trip Duration Prediction
Run: python flask_api.py
Then call: curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{...}'
"""

from flask import Flask, request, jsonify
from modeling_pipeline import TaxiDurationPredictor
import os
from datetime import datetime

app = Flask(__name__)

# Initialize predictor
predictor = None

@app.before_request
def load_model():
    """Load model on first request."""
    global predictor
    if predictor is None:
        predictor = TaxiDurationPredictor()
        predictor.load_model('Ridge')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None and predictor.is_loaded
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction on taxi trip.
    
    Expected JSON input:
    {
        "pickup_datetime": "2016-03-15 10:30:00",
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7489,
        "dropoff_longitude": -73.9680,
        "passenger_count": 1,
        "vendor_id": 2,
        "store_and_fwd_flag": "N"
    }
    
    Returns:
    {
        "prediction": {
            "predicted_duration_seconds": 523,
            "predicted_duration_minutes": 8,
            "predicted_duration_hours_minutes": "0h 8m",
            "distance_km": 1.23
        },
        "status": "success"
    }
    """
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'pickup_datetime', 'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude', 'passenger_count',
            'vendor_id', 'store_and_fwd_flag'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing}'
            }), 400
        
        # Make prediction
        result = predictor.predict_single(
            pickup_dt=data['pickup_datetime'],
            pickup_lat=data['pickup_latitude'],
            pickup_lon=data['pickup_longitude'],
            dropoff_lat=data['dropoff_latitude'],
            dropoff_lon=data['dropoff_longitude'],
            passenger_count=int(data['passenger_count']),
            vendor_id=int(data['vendor_id']),
            store_and_fwd_flag=data.get('store_and_fwd_flag', 'N')
        )
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'predicted_duration_seconds': int(result['predicted_duration_seconds']),
                'predicted_duration_minutes': int(result['predicted_duration_minutes']),
                'predicted_duration_hours_minutes': result['predicted_duration_hours_minutes'],
                'distance_km': float(result['distance_km'])
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Make predictions on multiple trips.
    
    Expected JSON input:
    {
        "trips": [
            {
                "pickup_datetime": "2016-03-15 10:30:00",
                "pickup_latitude": 40.7580,
                ...
            },
            ...
        ]
    }
    """
    
    try:
        data = request.get_json()
        trips = data.get('trips', [])
        
        if not trips:
            return jsonify({
                'status': 'error',
                'message': 'No trips provided'
            }), 400
        
        # Make predictions
        predictions = []
        for trip in trips:
            try:
                result = predictor.predict_single(
                    pickup_dt=trip['pickup_datetime'],
                    pickup_lat=trip['pickup_latitude'],
                    pickup_lon=trip['pickup_longitude'],
                    dropoff_lat=trip['dropoff_latitude'],
                    dropoff_lon=trip['dropoff_longitude'],
                    passenger_count=int(trip['passenger_count']),
                    vendor_id=int(trip['vendor_id']),
                    store_and_fwd_flag=trip.get('store_and_fwd_flag', 'N')
                )
                
                predictions.append({
                    'status': 'success',
                    'prediction': {
                        'predicted_duration_seconds': int(result['predicted_duration_seconds']),
                        'predicted_duration_minutes': int(result['predicted_duration_minutes']),
                        'predicted_duration_hours_minutes': result['predicted_duration_hours_minutes'],
                        'distance_km': float(result['distance_km'])
                    }
                })
            except Exception as e:
                predictions.append({
                    'status': 'error',
                    'message': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'total_predictions': len(predictions),
            'predictions': predictions
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/info', methods=['GET'])
def info():
    """Get model information."""
    return jsonify({
        'model_name': 'Ridge Regression',
        'task': 'Taxi Trip Duration Prediction',
        'input_features': {
            'pickup_datetime': 'string (YYYY-MM-DD HH:MM:SS)',
            'pickup_latitude': 'float (40.5-40.9)',
            'pickup_longitude': 'float (-74.05 to -73.7)',
            'dropoff_latitude': 'float (40.5-40.9)',
            'dropoff_longitude': 'float (-74.05 to -73.7)',
            'passenger_count': 'int (0-10)',
            'vendor_id': 'int (1 or 2)',
            'store_and_fwd_flag': 'string (Y or N)'
        },
        'output_features': {
            'predicted_duration_seconds': 'int',
            'predicted_duration_minutes': 'int',
            'predicted_duration_hours_minutes': 'string (e.g., "0h 8m")',
            'distance_km': 'float'
        },
        'endpoints': [
            'GET  /health - Health check',
            'GET  /info - Model information',
            'POST /predict - Single prediction',
            'POST /predict-batch - Batch predictions'
        ]
    })


if __name__ == '__main__':
    print("="*70)
    print("NYC Taxi Trip Duration Predictor - Flask API")
    print("="*70)
    print("Loading model...")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
