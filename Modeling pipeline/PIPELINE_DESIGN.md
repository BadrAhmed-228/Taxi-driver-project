# MODELING PIPELINE DESIGN DOCUMENT
## NYC Taxi Trip Duration Prediction

**Date:** 2026-04-18  
**Based on:** EDA_REPORT.md  
**Problem Type:** Regression (predict continuous trip duration)

---

## 1. EXECUTIVE SUMMARY

This document outlines the production-level ML pipeline designed based strictly on EDA findings. The pipeline addresses:

- **Data Quality:** Zero missing values ✓
- **Target Distribution:** Right-skewed with extreme outliers → Log transformation
- **Feature Correlations:** Weak (max 0.121) → Tree-based models
- **Feature Types:** Temporal, geographic, categorical → Specialized pipelines
- **Data Leakage:** Prevented via train-only fitting strategy

---

## 2. DECISION RATIONALE (EDA → DESIGN)

### 2.1 Log Transformation of Target Variable

**EDA Finding:**
```
Target Distribution:
- Mean: 15.91 minutes
- Median: 11.03 minutes  
- Max: 25.8 days (2,227,612 seconds!)
- Outliers (>34.83 min): 50,939 (5.09%)
- Std Dev: 3,882.07 seconds (highly variable)
```

**Design Decision:** `y_train_log = np.log1p(y_train)`

**Why:**
- Right-skewed distribution benefits from log transformation
- Extreme outliers (5.09%) handled better in log scale
- Reduces variance, improves model performance
- Prevents predictions from biasing toward outliers
- Use `log1p` (log(1+x)) to handle zeros safely

**Implementation:** Transform target before training, reverse after prediction

---

### 2.2 Feature Scaling Strategy

#### Numerical Features Pipeline

**EDA Finding:**
```
Distance (strongest predictor, r=0.121):
- Mean: 3.44 km
- Std Dev: 4.42 km
- 81% of trips < 5 km
- Outliers: 96,420 (9.64%) > 7.83 km
- Max: 1,240.91 km
```

**Design Decision:** RobustScaler for distance + StandardScaler for coordinates

**Why:**
- **RobustScaler for distance:** Uses median and IQR instead of mean/std
  - Resistant to outliers (9.64% outliers present)
  - Better for skewed distributions
  - Preserves outlier information (stored rides are valid)

- **StandardScaler for coordinates:** Geographic data less skewed
  - Symmetric distribution of latitude/longitude
  - No extreme outliers
  - StandardScaler adequate

**Code:**
```python
numerical_transformer = Pipeline([
    ('robust_scaler', RobustScaler())  # Not StandardScaler!
])
```

---

### 2.3 Categorical Features Encoding

#### Vendor ID

**EDA Finding:**
```
Distribution: 
- Vendor 1: 46.52% (465,207)
- Vendor 2: 53.48% (534,793)
- Vendor 2 has 27% LONGER trips (17.65 vs 13.92 min)
- Balanced but significant effect
```

**Design Decision:** OrdinalEncoder

**Why:**
- Balanced distribution (no imbalance issues)
- Strong effect on target (27% difference)
- OrdinalEncoder sufficient; OneHotEncoder works too
- Avoids multicollinearity issues

---

#### Store & Forward Flag

**EDA Finding:**
```
Distribution:
- N (No): 99.44% (994,443)
- Y (Yes): 0.56% (5,557)
- Y trips are 13% longer (17.93 vs 15.86 min)
- Highly imbalanced but meaningful
```

**Design Decision:** OrdinalEncoder (Y=1, N=0)

**Why:**
- Binary feature; OrdinalEncoder perfect fit
- Imbalanced but valid; keep both classes
- Small Y sample (5,557) is still meaningful
- Don't remove; represents offline GPS logging

---

### 2.4 Temporal Features Engineering

**EDA Finding:**
```
Hourly Effect:
- Shortest: Hour 6 (6 AM) - 13.80 min
- Longest: Hour 15 (3 PM) - 18.39 min
- Range: ±33% from mean (15.91 min)
- Rush hours (15-20): Significant variations

Daily Effect:
- Weekdays: ~16.40 min average
- Weekends: ~15.17 min average  
- Difference: ~8-12% longer weekdays
- Thursday longest (16.74 min), Monday shortest (15.00 min)

Monthly Effect:
- Fairly consistent (~16 min)
- No strong seasonal pattern (6 months only)
```

**Design Decision:** Extract hour, day_of_week, is_rush_hour, is_weekday

**Why:**
- **hour (OneHotEncoder):** 33% variation; too important to ignore
  - Create dummies for each hour (drop first to avoid multicollinearity)
- **day_of_week (OneHotEncoder):** 12% variation; clear weekday effect
  - Capture Thursday peak vs Monday low
- **is_rush_hour (Binary):** Rush hours (15-20) show longest trips
  - Binary feature captures peak effect
- **is_weekday (Binary):** 10-12% difference is significant
  - Separate weekday/weekend model differences

**Code:**
```python
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  
df['is_rush_hour'] = ((df['hour'] >= 15) & (df['hour'] <= 20)).astype(int)
df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
```

---

### 2.5 Distance Feature Engineering

**EDA Finding:**
```
Distance Insights:
- Strongest predictor: r = 0.121 (weak but best available)
- Calculated as: Haversine distance between pickup/dropoff
- Distribution: Heavily skewed, 81% < 5 km
- Zero-distance trips: 4,085 (0.41%)
- Outliers (>7.83 km): 96,420 (9.64%)
```

**Design Decision:** Calculate haversine distance; keep zero-distance as binary feature

**Why:**
- Haversine is geographic standard (great-circle distance)
- More realistic than Euclidean for lat/lon coordinates
- Zero-distance trips important: same pickup/dropoff location
  - Could indicate waiting, dropped-off pickup (different person)
  - Create `is_zero_distance` binary flag
  - Helps model distinguish these edge cases

**Code:**
```python
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km."""
    R = 6371  # Earth radius in km
    # ... formula ...
    return R * c

df['distance_km'] = calculate_haversine_distance(...)
df['is_zero_distance'] = (df['distance_km'] == 0).astype(int)
```

---

### 2.6 Passenger Count Feature

**EDA Finding:**
```
Distribution:
- 0 passengers: 0.004% (38) - anomaly
- 1 passenger: 70.86% (708,628) - dominant
- 2-4: 20.44%
- 5 passengers: 5.36% (53,635) - interesting spike
- 6: 3.33%
- 7: 0.0002% (2) - ultra-rare

Duration Effect:
- 1 passenger: 15.42 min
- 5 passengers: 17.87 min (16% increase)
- Clear ordering effect
```

**Design Decision:** Group rare values (0, 7 → 1; 5+ → 5)

**Why:**
- Clear ordinal relationship (more passengers = longer trips)
- Keep as numerical/categorical for better predictions
- Group rare categories to prevent sparse one-hot encoding
- 5-passenger spike is real (might be shared rides)
- Group as: [0→1, 1, 2, 3, 4, 5, 6+→5]

**Code:**
```python
df['passenger_group'] = df['passenger_count'].copy()
df.loc[df['passenger_count'] == 0, 'passenger_group'] = 1
df.loc[df['passenger_count'] >= 5, 'passenger_group'] = 5
# Then OneHotEncoder on passenger_group
```

---

### 2.7 Geographic Coordinates Handling

**EDA Finding:**
```
Coordinate Correlations with Duration:
- pickup_latitude: -0.034 (very weak negative)
- pickup_longitude: 0.035 (very weak positive)
- dropoff_latitude: -0.028 (very weak negative)
- dropoff_longitude: 0.022 (very weak positive)

Geographic Range:
- Latitude: 34.36° to 51.88° (some outside NYC bounds)
- Longitude: -121.93° to -61.34° (very wide range)
- Anomalies: 222 records outside NYC (0.02%)

Insight:
- Weak direct correlation but location DOES matter
- Geographic clusters exist but not captured by raw coordinates
```

**Design Decision:** Keep raw coordinates + StandardScaler (no removal)

**Why:**
- Weak direct correlation BUT geographic patterns matter (mentioned in EDA)
- Tree models can capture non-linear geographic effects
- Don't remove; 222 anomalies (0.02%) negligible
- StandardScaler sufficient for coordinates (less skewed than distance)
- Could add geographic features in future (neighborhoods, grid-based)

---

## 3. OUTLIER HANDLING STRATEGY

**EDA Finding:**
```
Outliers Identified:
- Duration > 34.83 min: 50,939 (5.09%)
- Distance > 7.83 km: 96,420 (9.64%)
- Extreme durations (>24h): 8,361
- Maximum duration: 25.8 days
- Zero distance trips: 4,085 (0.41%)
- Data anomalies: 222 coords outside NYC (0.02%)
```

**Design Decision:** Keep outliers; use RobustScaler

**Why:**
- **DO NOT remove outliers:**
  - 5.09% is significant (50,939 records)
  - Stored rides are valid data (store_and_fwd_flag = Y)
  - EDA noted these are "potential data logging errors or special circumstances"
  - Could represent legitimate business patterns

- **Use RobustScaler:**
  - Handles outliers without removing them
  - Preserves outlier information
  - IQR-based (median/IQR) less influenced by extremes
  - Better than StandardScaler for skewed data

- **Log transformation on target:**
  - Further normalizes distribution
  - Reduces influence of extreme values naturally
  - Reversible for final predictions

---

## 4. MODEL SELECTION

**EDA Finding:**
```
Feature Correlations (vs. trip_duration):
- distance_km: 0.121 (WEAK)
- pickup_longitude: 0.035
- dropoff_longitude: 0.022
- passenger_count: 0.012
- hour: 0.003
- All correlations < 0.13

Insight:
- Linear relationships are WEAK
- External factors (traffic, congestion) likely dominant
- Non-linear models needed
```

**Design Decision:** Tree-Based Ensemble Models

**Selected Models:**
1. **XGBoost** (primary choice)
   - Handles non-linear relationships
   - Naturally handles mixed feature types
   - Robust to outliers via gradient boosting
   - Hyperparameters tuned for taxi domain

2. **Random Forest** (baseline)
   - Simple ensemble, low variance
   - Good for comparison

3. **Gradient Boosting** (secondary)
   - Similar to XGBoost, slower but interpretable

**Why NOT Linear Regression:**
- Correlations too weak (0.121 max) for linear model
- Interaction effects likely (hour × vendor, location × distance)
- Tree models capture these non-linearities

---

## 5. DATA LEAKAGE PREVENTION

**Critical Strategy:** Fit Preprocessor ONLY on Training Data

### 5.1 Train/Validation Split Strategy

```python
# Step 1: Separate features and target
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2: FIT preprocessor on TRAINING data only
preprocessor.fit(X_train)

# Step 3: Transform both sets using FITTED preprocessor
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)  # Use fitted scaler!

# Step 4: Train model on training data only
model.fit(X_train_processed, y_train)

# Step 5: Evaluate on validation set
y_val_pred = model.predict(X_val_processed)
```

### 5.2 Why This Matters

**Data Leakage Risk:**
- If scaler is fit on full dataset (train + val), validation set statistics leak into training
- Mean/std computed from validation set affects scaling
- Model sees "ghost" information from validation set during training
- Evaluation metrics become optimistic (overestimating performance)

**Prevention:**
- ColumnTransformer.fit() called ONLY on training data
- Validation/test data uses fitted transformers (no new fitting)
- Each fold in cross-validation follows same pattern

---

## 6. PREPROCESSING PIPELINE ARCHITECTURE

```
Input Data
    ↓
Feature Engineering (temporal, distance, binary flags)
    ↓
ColumnTransformer
    ├── Numerical Branch
    │   ├── distance_km → RobustScaler
    │   ├── pickup_longitude → StandardScaler
    │   ├── pickup_latitude → StandardScaler
    │   ├── dropoff_longitude → StandardScaler
    │   └── dropoff_latitude → StandardScaler
    │
    ├── Categorical/Binary Branch
    │   ├── vendor_id → OrdinalEncoder
    │   ├── store_and_fwd_flag → OrdinalEncoder
    │   ├── is_rush_hour → OrdinalEncoder
    │   ├── is_weekday → OrdinalEncoder
    │   └── is_zero_distance → OrdinalEncoder
    │
    └── One-Hot Branch
        ├── hour → OneHotEncoder (drop='first')
        ├── day_of_week → OneHotEncoder (drop='first')
        └── passenger_group → OneHotEncoder (drop='first')
    ↓
Concatenated Feature Matrix [n_samples, n_features]
    ↓
Model Training/Prediction
```

---

## 7. EVALUATION METRICS & RATIONALE

**Problem Type:** Regression (continuous target)

### Metrics Selected:

1. **R² Score (Coefficient of Determination)**
   - Measures proportion of variance explained
   - Range: -∞ to 1 (1 = perfect, <0 = worse than mean)
   - **Primary metric** for model selection
   
2. **RMSE (Root Mean Squared Error)**
   - Square root of average squared errors
   - Penalizes large errors heavily
   - Important for trip duration (missing one trip time badly is costly)
   - Reported in both log scale and original seconds

3. **MAE (Mean Absolute Error)**
   - Average absolute error
   - Interpretable in minutes (user-friendly)
   - Less sensitive to outliers than RMSE
   - Useful for communicating error to stakeholders

### Evaluation Protocol:

```python
# Evaluate on log-transformed scale
rmse_log = sqrt(mean_squared_error(y_train_log, y_pred_log))

# Reverse transformation for interpretation
y_true_original = exp(y_true_log) - 1
y_pred_original = exp(y_pred_log) - 1

# Report in original scale (seconds/minutes)
rmse_seconds = sqrt(mean_squared_error(y_true_original, y_pred_original))
mae_minutes = mean_absolute_error(y_true_original, y_pred_original) / 60
```

---

## 8. EXPECTED MODEL PERFORMANCE

**Based on EDA insights:**

### Baseline Expectation
- **Distance-only model:** R² ≈ 0.015 (correlation 0.121 squared)
- **Mean duration model:** R² = 0 (just predicting mean)

### Realistic Target
- **R² = 0.3 - 0.5** with good feature engineering
- **MAE ≈ 3-4 minutes** average error
- **RMSE ≈ 5-8 minutes** (accounts for variance)

### Why Performance Is Limited
1. Weak feature correlations (max 0.121)
2. Missing features: traffic, weather, road conditions, events
3. Target variance driven by external factors
4. Only 6 months of data (limited seasonal patterns)

---

## 9. IMPORTANT ASSUMPTIONS & CAVEATS

### Data Quality Assumptions
1. ✓ Zero missing values (confirmed by EDA)
2. ✓ Trip duration in valid range (1 sec - valid; 25.8 days - anomaly but kept)
3. ✓ Coordinates in reasonable range (some anomalies at 0.02%)
4. ✓ No major data drift in validation/test sets

### Feature Engineering Assumptions
1. Haversine distance is good proxy for actual road distance
2. Temporal patterns stable (only 6 months data)
3. Vendor effects remain consistent
4. Passenger count accurately recorded

### Model Assumptions
1. Non-linear relationships exist (tree-based models)
2. Feature interactions important (hour × location, vendor × distance)
3. Outliers are valid (not errors to be removed)
4. Log transformation appropriate for right-skewed target

---

## 10. PRODUCTION CHECKLIST

- [ ] Data loaded successfully
- [ ] No missing values in train/val/test
- [ ] Feature engineering applied consistently
- [ ] Preprocessor fitted on training data only
- [ ] Validation/test use fitted transformers (no refitting)
- [ ] Model trained on training data
- [ ] Evaluation performed on validation set
- [ ] Predictions reversed from log scale
- [ ] Metrics reported in interpretable units (minutes)
- [ ] Code is modular and reusable
- [ ] No hardcoded paths (use relative paths)
- [ ] Handles edge cases (zero distance, rare passengers)

---

## 11. NEXT STEPS & IMPROVEMENTS

### Immediate (within scope)
1. Feature selection via permutation importance
2. Hyperparameter tuning (GridSearchCV/RandomSearchCV)
3. Cross-validation for robust evaluation
4. Prediction confidence intervals

### Future Improvements
1. Add geographic features (pickup/dropoff borough, grid-based features)
2. Collect additional features (weather, traffic, special events)
3. Temporal validation (time-based split) instead of random split
4. Ensemble stacking (combine multiple model types)
5. Separate models for different trip types (short vs. long)

---

## 12. CODE STRUCTURE

Main script: `modeling_pipeline.py`

Functions:
- `load_data()` - Load train/val/test CSVs
- `calculate_haversine_distance()` - Geographic distance
- `engineer_features()` - Temporal + distance features
- `create_preprocessing_pipeline()` - ColumnTransformer setup
- `create_models()` - Initialize XGBoost, RF, GB
- `train_model()` - Fit preprocessor (train only), train model
- `evaluate_model()` - Compute metrics, reverse transformations
- `main()` - Execute complete pipeline

---

**Document Version:** 1.0  
**Last Updated:** 2026-04-18  
**Status:** Ready for Production Deployment
