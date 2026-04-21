# Deployment Guide - NYC Taxi Duration Predictor

## Overview

Your model has been packaged with **three deployment options**:

1. **Python Module** - Direct programmatic access
2. **Streamlit Web App** - Interactive web interface
3. **Flask REST API** - HTTP endpoints for integration

All options use the `TaxiDurationPredictor` wrapper class that handles:
- Raw user input (basic features only)
- Automatic feature engineering
- Preprocessing & scaling
- Model prediction
- Output interpretation

---

## Setup

### Prerequisites
```bash
pip install streamlit flask pandas numpy scikit-learn
```

### File Structure
```
Modeling pipeline/
├── modeling_pipeline.py        # Main pipeline with TaxiDurationPredictor class
├── Ridge_model.pkl             # Trained model (generated after running)
├── Ridge_preprocessor.pkl      # Preprocessing pipeline (generated)
├── Ridge_results.pkl           # Results/metrics (generated)
├── streamlit_app.py            # Streamlit web app
├── flask_api.py                # Flask API
└── DEPLOYMENT_GUIDE.md         # This file
```

---

## Option 1: Python Module (Direct Usage)

### Usage
```python
from modeling_pipeline import TaxiDurationPredictor
import pandas as pd

# Initialize predictor
predictor = TaxiDurationPredictor()
predictor.load_model('Ridge')

# Single prediction
result = predictor.predict_single(
    pickup_dt='2016-03-15 10:30:00',
    pickup_lat=40.7580,
    pickup_lon=-73.9855,
    dropoff_lat=40.7489,
    dropoff_lon=-73.9680,
    passenger_count=1,
    vendor_id=2,
    store_and_fwd_flag='N'
)

print(result)
# Output:
# {
#     'predicted_duration_seconds': 523,
#     'predicted_duration_minutes': 8,
#     'predicted_duration_hours_minutes': '0h 8m',
#     'pickup_datetime': '2016-03-15 10:30:00',
#     'passenger_count': 1,
#     'distance_km': 1.23
# }
```

### Batch Prediction
```python
# Multiple trips
raw_data = pd.DataFrame({
    'pickup_datetime': ['2016-03-15 10:30:00', '2016-03-15 14:15:00'],
    'pickup_latitude': [40.7580, 40.7600],
    'pickup_longitude': [-73.9855, -73.9800],
    'dropoff_latitude': [40.7489, 40.7400],
    'dropoff_longitude': [-73.9680, -73.9500],
    'passenger_count': [1, 2],
    'vendor_id': [2, 1],
    'store_and_fwd_flag': ['N', 'N']
})

predictions = predictor.predict(raw_data)
print(predictions)
```

---

## Option 2: Streamlit Web App

### Run
```bash
cd "c:\Users\pc\Downloads\Taxi driver project\Modeling pipeline"
streamlit run streamlit_app.py
```

### Features
- **Interactive UI** - No coding required
- **Real-time prediction** - Instant results
- **Visual feedback** - Clear metrics display
- **Location presets** - Quick NYC location selection

### User Flow
1. Click "Load Model" in sidebar
2. Enter pickup location (latitude/longitude)
3. Enter dropoff location
4. Select pickup date & time
5. Select passenger count & vendor
6. Click "Predict Duration"

### Output
- Predicted duration in seconds/minutes/hours
- Distance calculated
- Trip summary

### Example NYC Coordinates
```
Manhattan: (40.75, -73.97)
Times Square: (40.758, -73.986)
Central Park: (40.785, -73.968)
Brooklyn: (40.650, -73.950)
JFK Airport: (40.641, -73.778)
LaGuardia: (40.777, -73.874)
```

---

## Option 3: Flask REST API

### Run
```bash
cd "c:\Users\pc\Downloads\Taxi driver project\Modeling pipeline"
python flask_api.py
```

### Endpoints

#### 1. Health Check
```bash
GET http://localhost:5000/health

# Response:
{
    "status": "ok",
    "model_loaded": true
}
```

#### 2. Model Info
```bash
GET http://localhost:5000/info

# Response:
{
    "model_name": "Ridge Regression",
    "task": "Taxi Trip Duration Prediction",
    "input_features": {...},
    "output_features": {...},
    "endpoints": [...]
}
```

#### 3. Single Prediction
```bash
POST http://localhost:5000/predict
Content-Type: application/json

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

# Response:
{
    "status": "success",
    "prediction": {
        "predicted_duration_seconds": 523,
        "predicted_duration_minutes": 8,
        "predicted_duration_hours_minutes": "0h 8m",
        "distance_km": 1.23
    }
}
```

#### 4. Batch Predictions
```bash
POST http://localhost:5000/predict-batch
Content-Type: application/json

{
    "trips": [
        {"pickup_datetime": "2016-03-15 10:30:00", ...},
        {"pickup_datetime": "2016-03-15 14:15:00", ...}
    ]
}

# Response:
{
    "status": "success",
    "total_predictions": 2,
    "predictions": [
        {"status": "success", "prediction": {...}},
        {"status": "success", "prediction": {...}}
    ]
}
```

### Using cURL
```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2016-03-15 10:30:00",
    "pickup_latitude": 40.7580,
    "pickup_longitude": -73.9855,
    "dropoff_latitude": 40.7489,
    "dropoff_longitude": -73.9680,
    "passenger_count": 1,
    "vendor_id": 2,
    "store_and_fwd_flag": "N"
  }'
```

### Using Python Requests
```python
import requests
import json

url = 'http://localhost:5000/predict'

data = {
    'pickup_datetime': '2016-03-15 10:30:00',
    'pickup_latitude': 40.7580,
    'pickup_longitude': -73.9855,
    'dropoff_latitude': 40.7489,
    'dropoff_longitude': -73.9680,
    'passenger_count': 1,
    'vendor_id': 2,
    'store_and_fwd_flag': 'N'
}

response = requests.post(url, json=data)
result = response.json()

print(f"Duration: {result['prediction']['predicted_duration_hours_minutes']}")
```

### Using JavaScript/Node.js
```javascript
const data = {
    pickup_datetime: '2016-03-15 10:30:00',
    pickup_latitude: 40.7580,
    pickup_longitude: -73.9855,
    dropoff_latitude: 40.7489,
    dropoff_longitude: -73.9680,
    passenger_count: 1,
    vendor_id: 2,
    store_and_fwd_flag: 'N'
};

fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
})
.then(res => res.json())
.then(result => console.log(`Duration: ${result.prediction.predicted_duration_hours_minutes}`))
.catch(err => console.error(err));
```

---

## Input Specifications

### Required Fields

| Field | Type | Example | Valid Range |
|-------|------|---------|-------------|
| `pickup_datetime` | String | "2016-03-15 10:30:00" | YYYY-MM-DD HH:MM:SS |
| `pickup_latitude` | Float | 40.7580 | 40.5 - 40.9 |
| `pickup_longitude` | Float | -73.9855 | -74.05 to -73.7 |
| `dropoff_latitude` | Float | 40.7489 | 40.5 - 40.9 |
| `dropoff_longitude` | Float | -73.9680 | -74.05 to -73.7 |
| `passenger_count` | Integer | 1 | 0 - 10 |
| `vendor_id` | Integer | 2 | 1 or 2 |
| `store_and_fwd_flag` | String | "N" | "Y" or "N" |

---

## Output Specification

All methods return predictions with:

```python
{
    'predicted_duration_seconds': int,           # Total seconds
    'predicted_duration_minutes': int,           # Total minutes
    'predicted_duration_hours_minutes': str,     # Formatted "Xh Ym"
    'distance_km': float,                        # Calculated distance
    'pickup_datetime': str,                      # Input datetime
    'passenger_count': int                       # Input count
}
```

---

## Deployment Scenarios

### Scenario 1: Web Dashboard
**Use:** Streamlit
```bash
streamlit run streamlit_app.py
```
- Share via public URL
- No coding required for users
- Real-time updates

### Scenario 2: Mobile App Backend
**Use:** Flask API
```bash
python flask_api.py  # Or deploy to cloud (AWS, GCP, Azure)
```
- REST endpoints
- Scalable
- Easy integration

### Scenario 3: Batch Processing
**Use:** Python Module
```python
predictor = TaxiDurationPredictor()
predictor.load_model('Ridge')
predictions = predictor.predict(df_trips)
```
- Process large datasets
- Integrate into pipelines

### Scenario 4: Embedded in Larger System
**Use:** Python Module or Flask
```python
# As part of a larger taxi dispatch system
from taxi_dispatcher import get_available_cabs
from modeling_pipeline import TaxiDurationPredictor

predictor = TaxiDurationPredictor()
predictor.load_model('Ridge')

for cab in get_available_cabs():
    duration = predictor.predict_single(...)
    assign_cab_if_suitable(cab, duration)
```

---

## Troubleshooting

### Model Not Loading
```python
# Check if .pkl files exist
import os
print(os.listdir('.'))  # Should show Ridge_model.pkl, Ridge_preprocessor.pkl, Ridge_results.pkl
```

### Predictions Seem Wrong
- **Check coordinates**: Must be in NYC bounds (40.5-40.9 lat, -74.05 to -73.7 lon)
- **Check vendor_id**: Must be 1 or 2
- **Check datetime format**: Must be "YYYY-MM-DD HH:MM:SS"

### API Connection Issues
- Ensure Flask is running on correct port (5000)
- Check firewall settings
- For remote access, modify Flask config: `app.run(host='0.0.0.0')`

### Performance Issues
- Use batch endpoints for multiple predictions
- Consider caching frequent predictions
- Deploy API on server with GPU support (not needed for CPU inference)

---

## Production Checklist

- [ ] Verify all .pkl files exist
- [ ] Test all endpoints
- [ ] Set up monitoring (API response times)
- [ ] Add authentication if needed
- [ ] Set up error logging
- [ ] Configure CORS if cross-domain
- [ ] Document SLA (Service Level Agreement)
- [ ] Plan for model updates
- [ ] Set up automated backups
- [ ] Monitor prediction drift over time

---

## Next Steps

1. **Test locally** with one of the three options
2. **Choose deployment method** based on your use case
3. **Deploy to production** (Cloud, On-Premise, etc.)
4. **Monitor performance** and prediction quality
5. **Retrain model** quarterly with new data

---

**For questions or issues, check the main README or contact the data science team.**
