"""
Quick Start Examples - Using TaxiDurationPredictor in Different Scenarios
"""

import pandas as pd
from modeling_pipeline import TaxiDurationPredictor

# ============================================================================
# Example 1: Single Trip Prediction
# ============================================================================

def example_single_prediction():
    """Predict duration for a single trip."""
    print("=" * 70)
    print("EXAMPLE 1: Single Trip Prediction")
    print("=" * 70)
    
    predictor = TaxiDurationPredictor()
    predictor.load_model('Ridge')
    
    # User inputs (from Streamlit form, web form, API, etc.)
    result = predictor.predict_single(
        pickup_dt='2016-03-15 10:30:00',           # User enters date + time
        pickup_lat=40.7580,                        # User enters/clicks pickup
        pickup_lon=-73.9855,
        dropoff_lat=40.7489,                       # User enters/clicks dropoff
        dropoff_lon=-73.9680,
        passenger_count=1,                         # User enters
        vendor_id=2,                               # User selects
        store_and_fwd_flag='N'
    )
    
    print(f"Pickup: {result['pickup_datetime']}")
    print(f"Passengers: {result['passenger_count']}")
    print(f"Distance: {result['distance_km']:.2f} km")
    print(f"\n✓ Predicted Duration: {result['predicted_duration_hours_minutes']}")
    print(f"  ({result['predicted_duration_minutes']} minutes)")
    print()


# ============================================================================
# Example 2: Batch Processing (Multiple Trips)
# ============================================================================

def example_batch_prediction():
    """Process multiple trips at once."""
    print("=" * 70)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 70)
    
    predictor = TaxiDurationPredictor()
    predictor.load_model('Ridge')
    
    # User provides multiple trips (from CSV, database, API batch request)
    trips_data = pd.DataFrame({
        'pickup_datetime': [
            '2016-03-15 08:00:00',  # Morning trip
            '2016-03-15 17:30:00',  # Rush hour trip
            '2016-03-15 22:00:00',  # Late night trip
        ],
        'pickup_latitude': [40.7580, 40.7128, 40.7549],
        'pickup_longitude': [-73.9855, -74.0060, -73.9840],
        'dropoff_latitude': [40.7489, 40.7505, 40.7505],
        'dropoff_longitude': [-73.9680, -73.9935, -73.9935],
        'passenger_count': [1, 3, 2],
        'vendor_id': [2, 1, 2],
        'store_and_fwd_flag': ['N', 'N', 'Y']
    })
    
    # Get predictions
    results = predictor.predict(trips_data)
    
    print(results.to_string())
    print()


# ============================================================================
# Example 3: Integration with Dispatch System
# ============================================================================

def example_dispatch_system():
    """Example: Use predictions in a taxi dispatch decision."""
    print("=" * 70)
    print("EXAMPLE 3: Dispatch System Integration")
    print("=" * 70)
    
    predictor = TaxiDurationPredictor()
    predictor.load_model('Ridge')
    
    # Simulate incoming ride request
    ride_request = {
        'passenger_id': 'P123',
        'pickup': {'lat': 40.7580, 'lon': -73.9855, 'time': '2016-03-15 10:30:00'},
        'dropoff': {'lat': 40.7489, 'lon': -73.9680},
        'passengers': 1,
    }
    
    # Predict duration
    prediction = predictor.predict_single(
        pickup_dt=ride_request['pickup']['time'],
        pickup_lat=ride_request['pickup']['lat'],
        pickup_lon=ride_request['pickup']['lon'],
        dropoff_lat=ride_request['dropoff']['lat'],
        dropoff_lon=ride_request['dropoff']['lon'],
        passenger_count=ride_request['passengers'],
        vendor_id=2,
        store_and_fwd_flag='N'
    )
    
    # Business logic: Make dispatch decision
    predicted_mins = prediction['predicted_duration_minutes']
    distance_km = prediction['distance_km']
    
    print(f"Ride Request: {ride_request['passenger_id']}")
    print(f"  Distance: {distance_km:.2f} km")
    print(f"  Predicted Duration: {predicted_mins} minutes")
    
    # Decision logic
    if predicted_mins < 5:
        priority = "QUICK_TRIP"
        print(f"  → Decision: Assign to ANY available cab (quick)")
    elif predicted_mins < 20:
        priority = "STANDARD"
        print(f"  → Decision: Assign standard cab")
    else:
        priority = "LONG_TRIP"
        print(f"  → Decision: Use premium cab / multi-stop routing")
    
    print()


# ============================================================================
# Example 4: Popular Routes Analysis
# ============================================================================

def example_route_analysis():
    """Analyze predictions for common NYC routes."""
    print("=" * 70)
    print("EXAMPLE 4: Route Analysis")
    print("=" * 70)
    
    predictor = TaxiDurationPredictor()
    predictor.load_model('Ridge')
    
    routes = {
        'Times Square → JFK Airport': {
            'from': (40.7580, -73.9855),
            'to': (40.6413, -73.7781),
        },
        'Central Park → Brooklyn': {
            'from': (40.7829, -73.9654),
            'to': (40.6501, -73.9496),
        },
        'Grand Central → Manhattan Hotel': {
            'from': (40.7527, -73.9772),
            'to': (40.7505, -73.9934),
        }
    }
    
    print(f"{'Route':<40} {'Distance':>10} {'Duration':>12}")
    print("-" * 65)
    
    for route_name, coords in routes.items():
        result = predictor.predict_single(
            pickup_dt='2016-03-15 14:00:00',  # 2 PM (rush hour)
            pickup_lat=coords['from'][0],
            pickup_lon=coords['from'][1],
            dropoff_lat=coords['to'][0],
            dropoff_lon=coords['to'][1],
            passenger_count=1,
            vendor_id=2,
            store_and_fwd_flag='N'
        )
        
        print(f"{route_name:<40} {result['distance_km']:>9.2f} km  {result['predicted_duration_hours_minutes']:>12}")
    
    print()


# ============================================================================
# Example 5: Time-of-Day Impact Analysis
# ============================================================================

def example_time_impact():
    """Compare predictions for same trip at different times."""
    print("=" * 70)
    print("EXAMPLE 5: Time-of-Day Impact")
    print("=" * 70)
    
    predictor = TaxiDurationPredictor()
    predictor.load_model('Ridge')
    
    times = [
        '06:00:00',  # Early morning
        '08:30:00',  # Morning rush
        '12:00:00',  # Midday
        '17:30:00',  # Evening rush
        '22:00:00',  # Late night
    ]
    
    print(f"{'Time':<12} {'Duration (min)':>15} {'Impact':>15}")
    print("-" * 45)
    
    baseline_mins = None
    
    for time_str in times:
        result = predictor.predict_single(
            pickup_dt=f'2016-03-15 {time_str}',
            pickup_lat=40.7580,
            pickup_lon=-73.9855,
            dropoff_lat=40.7489,
            dropoff_lon=-73.9680,
            passenger_count=1,
            vendor_id=2,
            store_and_fwd_flag='N'
        )
        
        mins = result['predicted_duration_minutes']
        
        if baseline_mins is None:
            baseline_mins = mins
            impact = "Baseline"
        else:
            diff = mins - baseline_mins
            pct_diff = (diff / baseline_mins) * 100
            impact = f"{diff:+.0f}m ({pct_diff:+.0f}%)"
        
        print(f"{time_str:<12} {mins:>15} {impact:>15}")
    
    print()


# ============================================================================
# Example 6: Error Handling
# ============================================================================

def example_error_handling():
    """Handle common errors gracefully."""
    print("=" * 70)
    print("EXAMPLE 6: Error Handling")
    print("=" * 70)
    
    predictor = TaxiDurationPredictor()
    predictor.load_model('Ridge')
    
    # Scenario 1: Missing required fields
    print("Scenario 1: Missing required fields")
    try:
        incomplete_data = pd.DataFrame({
            'pickup_datetime': ['2016-03-15 10:00:00'],
            'pickup_latitude': [40.7580],
            # Missing other required fields
        })
        predictor.predict(incomplete_data)
    except ValueError as e:
        print(f"  ✓ Caught error: {e}\n")
    
    # Scenario 2: Invalid coordinates
    print("Scenario 2: Out-of-range coordinates")
    try:
        invalid_data = pd.DataFrame({
            'pickup_datetime': ['2016-03-15 10:00:00'],
            'pickup_latitude': [90.0],  # Invalid - out of range
            'pickup_longitude': [-73.9855],
            'dropoff_latitude': [40.7489],
            'dropoff_longitude': [-73.9680],
            'passenger_count': [1],
            'vendor_id': [2],
            'store_and_fwd_flag': ['N']
        })
        predictor.predict(invalid_data)
        print("  ⚠ Model still made prediction - check if reasonable\n")
    except ValueError as e:
        print(f"  ✓ Caught error: {e}\n")


# ============================================================================
# Example 7: Cost Estimation
# ============================================================================

def example_cost_estimation():
    """Use predictions for cost estimation."""
    print("=" * 70)
    print("EXAMPLE 7: Cost Estimation")
    print("=" * 70)
    
    predictor = TaxiDurationPredictor()
    predictor.load_model('Ridge')
    
    # NYC Taxi Rates (approximate):
    # Base fare: $2.50
    # Per mile: $2.50
    # Per minute: $0.50
    
    result = predictor.predict_single(
        pickup_dt='2016-03-15 14:00:00',
        pickup_lat=40.7580,
        pickup_lon=-73.9855,
        dropoff_lat=40.7489,
        dropoff_lon=-73.9680,
        passenger_count=1,
        vendor_id=2,
        store_and_fwd_flag='N'
    )
    
    # Calculate estimated cost
    base_fare = 2.50
    rate_per_mile = 2.50
    rate_per_minute = 0.50
    
    distance_miles = result['distance_km'] * 0.621371  # km to miles
    duration_minutes = result['predicted_duration_minutes']
    
    mileage_charge = distance_miles * rate_per_mile
    time_charge = duration_minutes * rate_per_minute
    subtotal = base_fare + mileage_charge + time_charge
    tax = subtotal * 0.08  # 8% tax
    total = subtotal + tax
    
    print(f"Trip Summary:")
    print(f"  Distance: {result['distance_km']:.2f} km ({distance_miles:.2f} mi)")
    print(f"  Duration: {result['predicted_duration_hours_minutes']}")
    print(f"\nEstimated Cost:")
    print(f"  Base Fare:      ${base_fare:.2f}")
    print(f"  Mileage Charge: ${mileage_charge:.2f} ({distance_miles:.2f} mi × ${rate_per_mile})")
    print(f"  Time Charge:    ${time_charge:.2f} ({duration_minutes} min × ${rate_per_minute})")
    print(f"  Subtotal:       ${subtotal:.2f}")
    print(f"  Tax (8%):       ${tax:.2f}")
    print(f"  TOTAL:          ${total:.2f}")
    print()


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == '__main__':
    print("\n")
    print("█" * 70)
    print("NYC TAXI DURATION PREDICTOR - USAGE EXAMPLES")
    print("█" * 70)
    print("\n")
    
    example_single_prediction()
    example_batch_prediction()
    example_dispatch_system()
    example_route_analysis()
    example_time_impact()
    example_error_handling()
    example_cost_estimation()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nFor more info, see DEPLOYMENT_GUIDE.md")
